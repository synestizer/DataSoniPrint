"""
app.py — Flask web backend for DataSoniPrint.

Endpoints:
    GET  /              → main UI
    POST /upload        → upload data file, return column info
    POST /preview       → quick line chart PNG of a column
    POST /process       → run sonification pipeline, return download links
    GET  /download/<id>/<type> → download WAV / STL / settings JSON
"""

import io
import json
import os
import secrets
import time
from pathlib import Path

from flask import (Flask, request, jsonify, send_file,
                   render_template, session)

from processing import (load_file, column_stats, process_file,
                         generate_preview_png, supported_extension)

app = Flask(__name__,
            template_folder="templates",
            static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024 * 1024  # 1 GB max upload

# In-memory store for processed results (keyed by session job ID).
# In production, swap for Redis or file-based storage.
_results = {}

# Auto-cleanup: keep results for max 30 minutes
MAX_RESULT_AGE = 1800


def _cleanup_old_results():
    now = time.time()
    expired = [k for k, v in _results.items() if now - v["ts"] > MAX_RESULT_AGE]
    for k in expired:
        del _results[k]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload data file, return column list + stats."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    if not supported_extension(f.filename):
        return jsonify({"error": "Unsupported file format. Accepted: CSV, HDF5, NetCDF, GRIB, ASDF"}), 400

    try:
        contents = f.read()
        headers, columns, row_count = load_file(contents, f.filename)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    stats = column_stats(columns, headers)

    job_id = secrets.token_hex(12)
    _results[job_id] = {
        "ts": time.time(),
        "file_bytes": contents,
        "filename": f.filename,
        "headers": headers,
        "columns_cache": columns,
    }
    _cleanup_old_results()

    session["job_id"] = job_id

    return jsonify({
        "job_id": job_id,
        "filename": f.filename,
        "row_count": row_count,
        "headers": headers,
        "stats": stats,
    })


@app.route("/preview", methods=["POST"])
def preview():
    """Generate a quick line chart preview of a column."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    job_id = data.get("job_id") or session.get("job_id")
    if not job_id or job_id not in _results:
        return jsonify({"error": "No file uploaded or session expired"}), 400

    job = _results[job_id]
    columns = job.get("columns_cache")
    hdrs = job.get("headers")

    if columns is None or hdrs is None:
        return jsonify({"error": "Data not found, please re-upload"}), 400

    column = data.get("column")
    try:
        png_bytes = generate_preview_png(columns, hdrs, selected_column=column)
    except Exception as e:
        return jsonify({"error": f"Preview failed: {e}"}), 500

    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        download_name="preview.png",
    )


@app.route("/preview-effects", methods=["POST"])
def preview_effects():
    """Generate a spectrogram preview showing sonification effects with spread parameter."""
    from processing import sonify_columns, compute_spectrogram, generate_spectrogram_preview_png
    
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    job_id = data.get("job_id") or session.get("job_id")
    if not job_id or job_id not in _results:
        return jsonify({"error": "No file uploaded or session expired"}), 400

    job = _results[job_id]
    columns = job.get("columns_cache")
    hdrs = job.get("headers")

    if columns is None or hdrs is None:
        return jsonify({"error": "Data not found, please re-upload"}), 400

    col_names = data.get("columns", [])
    spread = data.get("spread", 0.35)
    playback_sr = 44100

    try:
        # Sonify with the given spread value
        audio = sonify_columns(columns, col_names, spread=spread, playback_sr=playback_sr)
        
        # Compute spectrogram
        times, freqs, spec_db = compute_spectrogram(audio, sr=playback_sr)
        
        # Generate preview
        png_bytes = generate_spectrogram_preview_png(times, freqs, spec_db)
    except Exception as e:
        return jsonify({"error": f"Effects preview failed: {e}"}), 500

    return send_file(
        io.BytesIO(png_bytes),
        mimetype="image/png",
        download_name="effects_preview.png",
    )


@app.route("/process", methods=["POST"])
def process():
    """Run the sonification pipeline with user parameters."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    job_id = data.get("job_id") or session.get("job_id")
    if not job_id or job_id not in _results:
        return jsonify({"error": "No file uploaded or session expired"}), 400

    job = _results[job_id]
    file_bytes = job.get("file_bytes")
    if file_bytes is None:
        return jsonify({"error": "File data not found, please re-upload"}), 400

    params = {
        "columns": data.get("columns", []),
        "column": data.get("column"),
        "spread": data.get("spread", 0.35),
        "speed": data.get("speed", 0.5),
        "volume": data.get("volume", 0.7),
    }

    try:
        result = process_file(file_bytes, job["filename"], params)
    except Exception as e:
        return jsonify({"error": f"Processing failed: {e}"}), 500

    # Free cached columns to save memory now that we have outputs
    job.pop("columns_cache", None)

    job["wav"] = result["wav"]
    job["stl"] = result["stl"]
    job["settings"] = result["settings"]
    job["ts"] = time.time()

    return jsonify({
        "job_id": job_id,
        "settings": result["settings"],
        "headers": result["headers"],
        "ready": True,
    })


@app.route("/download/<job_id>/<file_type>")
def download(job_id, file_type):
    """Download a processed output file."""
    if job_id not in _results:
        return jsonify({"error": "Job not found or expired"}), 404

    job = _results[job_id]
    base = os.path.splitext(job.get("filename", "data"))[0]

    if file_type == "wav":
        data = job.get("wav")
        if not data:
            return jsonify({"error": "WAV not ready"}), 404
        return send_file(
            io.BytesIO(data),
            mimetype="audio/wav",
            as_attachment=True,
            download_name=f"{base}_sonified.wav",
        )
    elif file_type == "stl":
        data = job.get("stl")
        if not data:
            return jsonify({"error": "STL not ready"}), 404
        return send_file(
            io.BytesIO(data),
            mimetype="application/sla",
            as_attachment=True,
            download_name=f"{base}_terrain.stl",
        )
    elif file_type == "settings":
        settings = job.get("settings")
        if not settings:
            return jsonify({"error": "Settings not ready"}), 404
        return send_file(
            io.BytesIO(json.dumps(settings, indent=2).encode()),
            mimetype="application/json",
            as_attachment=True,
            download_name=f"{base}_settings.json",
        )
    else:
        return jsonify({"error": f"Unknown file type: {file_type}"}), 400


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DataSoniPrint Web — http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
