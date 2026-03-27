"""
processing.py — Headless data processing engine for DataSoniPrint web.

Extracts all computation from the pygame/pyo GUI code into pure functions
that take numpy arrays + parameters and return file-like outputs.

No GUI, no audio server — just numpy math → WAV bytes / STL bytes / JSON.
"""

import csv
import io
import json
import math
import struct
import wave
import numpy as np


# ─── CSV loading ─────────────────────────────────────────────────────────────

def load_csv(file_stream):
    """Load CSV from an in-memory file stream.

    Returns:
        headers: list of column names (numeric columns only)
        columns: dict mapping header → numpy float64 array
        row_count: int
    """
    text = file_stream.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8-sig")

    reader = csv.reader(io.StringIO(text))
    raw_headers = next(reader, None)
    if raw_headers is None:
        raise ValueError("CSV is empty")

    rows = list(reader)
    if not rows:
        raise ValueError("CSV has headers but no data rows")

    headers = []
    columns = {}
    for ci, h in enumerate(raw_headers):
        h = h.strip() or f"col_{ci}"
        vals = []
        num_count = 0
        for row in rows:
            if ci < len(row):
                cell = row[ci].strip()
                try:
                    vals.append(float(cell))
                    num_count += 1
                except ValueError:
                    vals.append(float("nan"))
            else:
                vals.append(float("nan"))
        if num_count > len(rows) * 0.5:
            headers.append(h)
            columns[h] = np.array(vals, dtype=np.float64)

    if not headers:
        raise ValueError("No numeric columns found in CSV")

    return headers, columns, len(rows)


# ─── Column stats ────────────────────────────────────────────────────────────

def column_stats(columns, headers):
    """Return per-column min/max/mean/count."""
    stats = {}
    for h in headers:
        col = columns[h]
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            stats[h] = {
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
                "mean": float(np.mean(valid)),
                "count": int(len(valid)),
            }
    return stats


# ─── FM Synthesis (core sonification) ────────────────────────────────────────

def sonify_column(data, spread=0.35, playback_sr=44100):
    """Convert a numeric column to FM-synthesized audio.

    Args:
        data: numpy array of values (may contain NaN)
        spread: 0.0–1.0 spread slider value
        playback_sr: output sample rate

    Returns:
        audio: numpy float64 array, peak ≤ 0.75
    """
    data = np.nan_to_num(data.copy(), nan=0.0)

    peak = np.max(np.abs(data))
    normed = data / peak if peak > 0 else data

    # upsample: each data point → ~0.01s of audio
    samples_per_point = max(1, int(playback_sr * 0.01))
    out_len = len(normed) * samples_per_point
    orig_t = np.linspace(0, 1, len(normed))
    new_t = np.linspace(0, 1, out_len)
    normed_up = np.interp(new_t, orig_t, normed)

    # FM synthesis: value → instantaneous frequency
    semi_range = 2 + spread * 46  # 2–48 semitones
    center_hz = 440.0
    freq_array = center_hz * np.power(2.0, normed_up * semi_range / 12.0)
    phase_inc = freq_array / playback_sr
    phase = np.cumsum(phase_inc)
    audio = np.sin(2.0 * np.pi * phase) * 0.75

    return audio


def apply_speed(audio, speed, playback_sr=44100):
    """Resample audio to change playback speed without changing pitch mapping.

    speed: multiplier (e.g. 2.0 = twice as fast = half as many samples)
    """
    if speed == 1.0:
        return audio
    new_len = max(1, int(len(audio) / speed))
    old_t = np.linspace(0, 1, len(audio))
    new_t = np.linspace(0, 1, new_len)
    return np.interp(new_t, old_t, audio)


def apply_volume(audio, volume):
    """Scale amplitude."""
    return audio * volume


# ─── WAV export ──────────────────────────────────────────────────────────────

def audio_to_wav_bytes(audio, sample_rate=44100):
    """Convert float64 audio array to in-memory WAV bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)
    return buf.read()


# ─── Spectrogram ─────────────────────────────────────────────────────────────

def compute_spectrogram(audio, sr=44100, nperseg=1024, overlap_frac=0.75):
    """Compute STFT spectrogram.

    Returns:
        times: 1D array of frame center times (seconds)
        freqs: 1D array of frequency bins (Hz)
        spec_db: 2D array (n_freqs × n_frames), power in dB
    """
    step = int(nperseg * (1 - overlap_frac))
    n_frames = (len(audio) - nperseg) // step + 1

    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)

    spec = np.zeros((len(freqs), n_frames))
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * step
        chunk = audio[start:start + nperseg] * window
        spec[:, i] = np.abs(np.fft.rfft(chunk))
        times[i] = start / sr

    spec_db = 20 * np.log10(spec + 1e-30)
    return times, freqs, spec_db


# ─── STL export ──────────────────────────────────────────────────────────────

def spectrogram_to_stl_bytes(times, freqs, spec_db,
                              freq_max=2000, time_downsample=4,
                              freq_downsample=2, base_thickness=2.0,
                              height_scale=15.0, width_mm=150, depth_mm=80):
    """Generate 3D-printable STL mesh from spectrogram, return as bytes.

    The spectrogram becomes a terrain surface:
    - X = time, Y = frequency, Z = power (height)
    - Solid base + side walls for printability
    """
    from stl import mesh as stl_mesh

    freq_mask = freqs <= freq_max
    spec = spec_db[freq_mask, :]
    spec = spec[::freq_downsample, ::time_downsample]
    n_freq, n_time = spec.shape

    # normalize height to [0, 1]
    vmin = np.percentile(spec, 2)
    vmax = np.percentile(spec, 99.5)
    height = np.clip((spec - vmin) / (vmax - vmin + 1e-10), 0, 1)

    x_scale = width_mm / max(n_time - 1, 1)
    y_scale = depth_mm / max(n_freq - 1, 1)

    # top surface vertices
    top_verts = np.zeros((n_freq, n_time, 3))
    for iy in range(n_freq):
        for ix in range(n_time):
            top_verts[iy, ix] = [
                ix * x_scale, iy * y_scale,
                base_thickness + height[iy, ix] * height_scale
            ]

    # bottom surface (flat)
    bot_verts = np.zeros((n_freq, n_time, 3))
    for iy in range(n_freq):
        for ix in range(n_time):
            bot_verts[iy, ix] = [ix * x_scale, iy * y_scale, 0.0]

    triangles = []

    def add_quad(v0, v1, v2, v3):
        triangles.append([v0, v1, v2])
        triangles.append([v0, v2, v3])

    # top surface
    for iy in range(n_freq - 1):
        for ix in range(n_time - 1):
            add_quad(top_verts[iy, ix], top_verts[iy, ix + 1],
                     top_verts[iy + 1, ix + 1], top_verts[iy + 1, ix])

    # bottom surface (reverse winding)
    for iy in range(n_freq - 1):
        for ix in range(n_time - 1):
            add_quad(bot_verts[iy, ix], bot_verts[iy + 1, ix],
                     bot_verts[iy + 1, ix + 1], bot_verts[iy, ix + 1])

    # side walls
    for ix in range(n_time - 1):
        add_quad(bot_verts[0, ix], bot_verts[0, ix + 1],
                 top_verts[0, ix + 1], top_verts[0, ix])
    for ix in range(n_time - 1):
        add_quad(bot_verts[-1, ix], top_verts[-1, ix],
                 top_verts[-1, ix + 1], bot_verts[-1, ix + 1])
    for iy in range(n_freq - 1):
        add_quad(bot_verts[iy, 0], top_verts[iy, 0],
                 top_verts[iy + 1, 0], bot_verts[iy + 1, 0])
    for iy in range(n_freq - 1):
        add_quad(bot_verts[iy, -1], bot_verts[iy + 1, -1],
                 top_verts[iy + 1, -1], top_verts[iy, -1])

    tri_array = np.array(triangles)
    m = stl_mesh.Mesh(np.zeros(len(tri_array), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(tri_array):
        m.vectors[i] = tri

    buf = io.BytesIO()
    m.save("output.stl", fh=buf)
    buf.seek(0)
    return buf.read()


# ─── Full pipeline ───────────────────────────────────────────────────────────

def process_csv(file_stream, params):
    """Run the complete DataSoniPrint pipeline on a CSV upload.

    Args:
        file_stream: file-like object with CSV data
        params: dict with keys:
            column: str — which column to sonify
            spread: float 0–1
            speed: float 0–1 slider value (maps to 0.25x–4x)
            volume: float 0–1

    Returns:
        dict with keys:
            wav: bytes — WAV audio file
            stl: bytes — STL mesh file
            settings: dict — parameters used
            stats: dict — per-column statistics
            headers: list — column names
    """
    SR = 44100

    # --- load ---
    headers, columns, row_count = load_csv(file_stream)

    # --- pick column ---
    col_name = params.get("column", headers[0])
    if col_name not in columns:
        col_name = headers[0]

    spread = float(params.get("spread", 0.35))
    speed_slider = float(params.get("speed", 0.5))
    volume = float(params.get("volume", 0.7))

    speed_multiplier = 0.25 * (16 ** speed_slider)  # 0.25x–4x

    # --- sonify ---
    audio = sonify_column(columns[col_name], spread=spread, playback_sr=SR)
    audio = apply_speed(audio, speed_multiplier, playback_sr=SR)
    audio = apply_volume(audio, volume)

    # --- WAV ---
    wav_bytes = audio_to_wav_bytes(audio, sample_rate=SR)

    # --- spectrogram + STL ---
    times, freqs, spec_db = compute_spectrogram(audio, sr=SR)
    stl_bytes = spectrogram_to_stl_bytes(times, freqs, spec_db)

    # --- settings ---
    stats = column_stats(columns, headers)
    settings = {
        "source_file": "uploaded.csv",
        "row_count": row_count,
        "column": col_name,
        "spread": spread,
        "speed_slider": speed_slider,
        "speed_multiplier": round(speed_multiplier, 4),
        "volume": volume,
        "sample_rate": SR,
        "audio_samples": len(audio),
        "audio_duration_sec": round(len(audio) / SR, 2),
        "stl_dimensions_mm": "150 x 80 x 17",
        "column_stats": stats,
    }

    return {
        "wav": wav_bytes,
        "stl": stl_bytes,
        "settings": settings,
        "stats": stats,
        "headers": headers,
    }
