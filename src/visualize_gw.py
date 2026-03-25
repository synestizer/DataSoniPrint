#!/usr/bin/env python3
"""
visualize_gw.py — Spectrogram, waveform, and 3D-printable STL from LIGO strain data.

Usage:
    # Spectrogram + waveform plot (saved as PNG)
    python src/visualize_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5

    # Focus on a segment
    python src/visualize_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --start 148 --duration 60

    # Export 3D-printable STL (spectrogram as terrain)
    python src/visualize_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --stl output.stl

    # All at once
    python src/visualize_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --all --stl gw_sculpture.stl

Requires: matplotlib, numpy, h5py, numpy-stl
"""

import argparse
import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_strain(path, start_sec=None, duration_sec=None):
    """Load strain + metadata from GWOSC HDF5."""
    import h5py

    with h5py.File(path, "r") as f:
        sr = int(f["strain/Strain"].shape[0] / f["meta/Duration"][()])
        det = f["meta/Detector"][()].decode() if hasattr(f["meta/Detector"][()], "decode") else str(f["meta/Detector"][()])
        utc = f["meta/UTCstart"][()].decode() if hasattr(f["meta/UTCstart"][()], "decode") else str(f["meta/UTCstart"][()])
        total_dur = int(f["meta/Duration"][()])
        dq = f["quality/simple/DQmask"][:]

        if start_sec is None:
            # find best good segment
            good = (dq > 0).astype(int)
            best_s, best_l = 0, 0
            rs = None
            for i in range(len(good)):
                if good[i] and rs is None:
                    rs = i
                elif not good[i] and rs is not None:
                    if (i - rs) > best_l:
                        best_s, best_l = rs, i - rs
                    rs = None
            if rs is not None and (len(good) - rs) > best_l:
                best_s, best_l = rs, len(good) - rs
            start_sec = best_s
            if duration_sec is None:
                duration_sec = best_l
            print(f"  Auto-selected: offset={start_sec}s, duration={duration_sec}s")
        elif duration_sec is None:
            duration_sec = total_dur - start_sec

        i0 = start_sec * sr
        i1 = i0 + duration_sec * sr
        strain = f["strain/Strain"][i0:i1]

    nan_count = np.sum(~np.isfinite(strain))
    if nan_count > 0:
        print(f"  {nan_count} NaN samples → 0")
        strain = np.nan_to_num(strain, nan=0.0)

    return strain, {
        "detector": det, "utc_start": utc, "sample_rate": sr,
        "total_duration": total_dur, "segment_start": start_sec,
        "segment_duration": duration_sec, "dq": dq,
    }


def compute_spectrogram(strain, sr, nperseg=1024, overlap_frac=0.75):
    """Compute spectrogram using short-time FFT."""
    step = int(nperseg * (1 - overlap_frac))
    n_frames = (len(strain) - nperseg) // step + 1

    window = np.hanning(nperseg)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / sr)

    spec = np.zeros((len(freqs), n_frames))
    times = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * step
        chunk = strain[start:start + nperseg] * window
        spec[:, i] = np.abs(np.fft.rfft(chunk))
        times[i] = start / sr

    # convert to dB (avoid log(0))
    spec_db = 20 * np.log10(spec + 1e-30)

    return times, freqs, spec_db


def plot_spectrogram(strain, info, output_path=None, freq_max=2000, show=True):
    """Plot waveform + spectrogram + RMS energy."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    sr = info["sample_rate"]
    duration = info["segment_duration"]
    t_offset = info["segment_start"]

    print(f"  Computing spectrogram ({len(strain):,} samples)...")
    times, freqs, spec_db = compute_spectrogram(strain, sr, nperseg=2048)

    # clip frequency range
    freq_mask = freqs <= freq_max
    freqs_plot = freqs[freq_mask]
    spec_plot = spec_db[freq_mask, :]

    # compute RMS per second
    rms_window = sr
    n_rms = len(strain) // rms_window
    rms = np.array([np.sqrt(np.mean(strain[i * rms_window:(i + 1) * rms_window] ** 2))
                     for i in range(n_rms)])
    rms_times = np.arange(n_rms) + t_offset

    # create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), height_ratios=[1, 3, 1],
                              sharex=True)
    fig.suptitle(f"LIGO {info['detector']} — {info['utc_start']}  "
                 f"(t={t_offset}–{t_offset + duration}s)",
                 fontsize=13, fontweight="bold")

    # waveform
    ax_wave = axes[0]
    t_wave = np.linspace(t_offset, t_offset + duration, len(strain))
    # downsample for plotting if too many points
    if len(strain) > 50000:
        ds = len(strain) // 50000
        ax_wave.plot(t_wave[::ds], strain[::ds], color="#3a86a8", linewidth=0.3)
    else:
        ax_wave.plot(t_wave, strain, color="#3a86a8", linewidth=0.3)
    ax_wave.set_ylabel("Strain")
    ax_wave.set_title("Waveform", fontsize=10, loc="left")
    ax_wave.ticklabel_format(axis="y", style="scientific", scilimits=(-18, -18))

    # spectrogram
    ax_spec = axes[1]
    extent = [times[0] + t_offset, times[-1] + t_offset, freqs_plot[0], freqs_plot[-1]]
    vmin = np.percentile(spec_plot, 5)
    vmax = np.percentile(spec_plot, 99)
    im = ax_spec.imshow(spec_plot, aspect="auto", origin="lower", extent=extent,
                         cmap="inferno", vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Spectrogram", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax_spec, label="Power (dB)", pad=0.01, fraction=0.02)

    # RMS energy
    ax_rms = axes[2]
    ax_rms.fill_between(rms_times, rms, color="#e07a5f", alpha=0.7)
    ax_rms.plot(rms_times, rms, color="#c44536", linewidth=0.8)
    ax_rms.set_ylabel("RMS")
    ax_rms.set_xlabel("Time (s)")
    ax_rms.set_title("RMS Energy per second", fontsize=10, loc="left")
    ax_rms.ticklabel_format(axis="y", style="scientific", scilimits=(-18, -18))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return times, freqs, spec_db


def spectrogram_to_stl(times, freqs, spec_db, output_path,
                        freq_max=2000, time_downsample=4, freq_downsample=2,
                        base_thickness=2.0, height_scale=15.0,
                        width_mm=150, depth_mm=80):
    """
    Convert spectrogram to a 3D-printable STL mesh.

    The spectrogram becomes a terrain surface:
    - X axis = time
    - Y axis = frequency
    - Z axis = power (height)

    A solid base is added underneath for printability.
    """
    from stl import mesh as stl_mesh

    # clip frequency range
    freq_mask = freqs <= freq_max
    spec = spec_db[freq_mask, :]
    freqs_c = freqs[freq_mask]

    # downsample for manageable mesh size
    spec = spec[::freq_downsample, ::time_downsample]
    n_freq, n_time = spec.shape
    print(f"  STL grid: {n_time} x {n_freq} ({n_time * n_freq:,} vertices)")

    # normalize height to 0–1 range
    vmin = np.percentile(spec, 2)
    vmax = np.percentile(spec, 99.5)
    height = (spec - vmin) / (vmax - vmin + 1e-10)
    height = np.clip(height, 0, 1)

    # scale to mm
    x_scale = width_mm / max(n_time - 1, 1)
    y_scale = depth_mm / max(n_freq - 1, 1)

    # build vertex grid: top surface
    top_verts = np.zeros((n_freq, n_time, 3))
    for iy in range(n_freq):
        for ix in range(n_time):
            top_verts[iy, ix] = [
                ix * x_scale,
                iy * y_scale,
                base_thickness + height[iy, ix] * height_scale
            ]

    # bottom surface (flat base)
    bot_verts = np.zeros((n_freq, n_time, 3))
    for iy in range(n_freq):
        for ix in range(n_time):
            bot_verts[iy, ix] = [ix * x_scale, iy * y_scale, 0.0]

    # generate triangles
    triangles = []

    def add_quad(v0, v1, v2, v3):
        """Add two triangles for a quad (v0-v1-v2-v3)."""
        triangles.append([v0, v1, v2])
        triangles.append([v0, v2, v3])

    # top surface
    for iy in range(n_freq - 1):
        for ix in range(n_time - 1):
            add_quad(
                top_verts[iy, ix], top_verts[iy, ix + 1],
                top_verts[iy + 1, ix + 1], top_verts[iy + 1, ix]
            )

    # bottom surface (reverse winding)
    for iy in range(n_freq - 1):
        for ix in range(n_time - 1):
            add_quad(
                bot_verts[iy, ix], bot_verts[iy + 1, ix],
                bot_verts[iy + 1, ix + 1], bot_verts[iy, ix + 1]
            )

    # side walls — front (iy=0)
    for ix in range(n_time - 1):
        add_quad(
            bot_verts[0, ix], bot_verts[0, ix + 1],
            top_verts[0, ix + 1], top_verts[0, ix]
        )
    # back (iy=max)
    for ix in range(n_time - 1):
        add_quad(
            bot_verts[-1, ix], top_verts[-1, ix],
            top_verts[-1, ix + 1], bot_verts[-1, ix + 1]
        )
    # left (ix=0)
    for iy in range(n_freq - 1):
        add_quad(
            bot_verts[iy, 0], top_verts[iy, 0],
            top_verts[iy + 1, 0], bot_verts[iy + 1, 0]
        )
    # right (ix=max)
    for iy in range(n_freq - 1):
        add_quad(
            bot_verts[iy, -1], bot_verts[iy + 1, -1],
            top_verts[iy + 1, -1], top_verts[iy, -1]
        )

    # build STL mesh
    tri_array = np.array(triangles)
    m = stl_mesh.Mesh(np.zeros(len(tri_array), dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(tri_array):
        m.vectors[i] = tri

    m.save(output_path)
    print(f"  STL saved: {output_path}")
    print(f"  Dimensions: {width_mm}mm x {depth_mm}mm x {base_thickness + height_scale:.0f}mm")
    print(f"  Triangles: {len(tri_array):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize & 3D-print LIGO gravitational wave data")
    parser.add_argument("hdf5_file", help="Path to GWOSC HDF5 file")
    parser.add_argument("--start", type=int, default=None,
                        help="Start time in seconds (default: auto-find good segment)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration in seconds (default: entire good segment)")
    parser.add_argument("--all", action="store_true",
                        help="Use entire file including gaps")
    parser.add_argument("--freq-max", type=float, default=2000,
                        help="Max frequency for spectrogram/STL (default: 2000 Hz)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open plot window (save only)")
    parser.add_argument("--png", type=str, default=None,
                        help="Save spectrogram plot to PNG file")
    parser.add_argument("--stl", type=str, default=None,
                        help="Export spectrogram as 3D-printable STL")
    parser.add_argument("--stl-width", type=float, default=150,
                        help="STL width in mm (time axis, default: 150)")
    parser.add_argument("--stl-depth", type=float, default=80,
                        help="STL depth in mm (frequency axis, default: 80)")
    parser.add_argument("--stl-height", type=float, default=15,
                        help="STL max peak height in mm (default: 15)")
    parser.add_argument("--stl-base", type=float, default=2.0,
                        help="STL base thickness in mm (default: 2)")
    args = parser.parse_args()

    # load
    if args.all:
        strain, info = load_strain(args.hdf5_file, start_sec=0, duration_sec=None)
    else:
        strain, info = load_strain(args.hdf5_file, args.start, args.duration)

    print(f"  {info['detector']} | {info['utc_start']} | "
          f"{len(strain):,} samples | {info['segment_duration']}s")

    # default: save PNG next to the HDF5 file
    png_path = args.png
    if png_path is None and args.stl is None:
        base = os.path.splitext(os.path.basename(args.hdf5_file))[0]
        png_path = f"{base}_spectrogram.png"

    # compute spectrogram
    times, freqs, spec_db = plot_spectrogram(
        strain, info,
        output_path=png_path,
        freq_max=args.freq_max,
        show=not args.no_show,
    )

    # STL export
    if args.stl:
        spectrogram_to_stl(
            times, freqs, spec_db,
            output_path=args.stl,
            freq_max=args.freq_max,
            width_mm=args.stl_width,
            depth_mm=args.stl_depth,
            height_scale=args.stl_height,
            base_thickness=args.stl_base,
        )


if __name__ == "__main__":
    main()
