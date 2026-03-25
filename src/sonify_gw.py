#!/usr/bin/env python3
"""
sonify_gw.py — Sonify LIGO/GWOSC gravitational wave strain data.

Two modes:
  1. Direct playback: pitch-shift the raw strain into audible range
  2. Spectral sonification: map time-frequency content to sine waves

Usage:
    python src/sonify_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5
    python src/sonify_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --mode spectral
    python src/sonify_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --start 200 --duration 30
    python src/sonify_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --speed 1 --bandpass 20 500
    python src/sonify_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --wav output.wav
    python src/sonify_gw.py ~/Downloads/H-H1_GWOSC_*.hdf5 --mode spectral --wav spectral.wav

Keys during playback:
    SPACE — pause / resume
    Q/ESC — quit
"""

import argparse
import os
import sys
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_gw_strain(path, start_sec=None, duration_sec=None):
    """Load strain data from GWOSC HDF5 file, auto-finding good segments."""
    import h5py

    with h5py.File(path, "r") as f:
        sample_rate = int(f["strain/Strain"].shape[0] / f["meta/Duration"][()])
        detector = f["meta/Detector"][()].decode() if hasattr(f["meta/Detector"][()], "decode") else str(f["meta/Detector"][()])
        utc_start = f["meta/UTCstart"][()].decode() if hasattr(f["meta/UTCstart"][()], "decode") else str(f["meta/UTCstart"][()])
        total_duration = int(f["meta/Duration"][()])

        dq = f["quality/simple/DQmask"][:]

        # find longest good segment if no start given
        if start_sec is None:
            good = (dq > 0).astype(int)
            best_start, best_len = 0, 0
            run_start = None
            for i in range(len(good)):
                if good[i] and run_start is None:
                    run_start = i
                elif not good[i] and run_start is not None:
                    if (i - run_start) > best_len:
                        best_start, best_len = run_start, i - run_start
                    run_start = None
            if run_start is not None and (len(good) - run_start) > best_len:
                best_start, best_len = run_start, len(good) - run_start
            start_sec = best_start
            if duration_sec is None:
                duration_sec = min(best_len, 60)  # default: 60s max
            print(f"  Auto-selected good segment: offset={start_sec}s, duration={duration_sec}s")
        elif duration_sec is None:
            duration_sec = 30

        # read strain
        i0 = start_sec * sample_rate
        i1 = i0 + duration_sec * sample_rate
        strain = f["strain/Strain"][i0:i1]

    # replace NaN with 0
    nan_count = np.sum(~np.isfinite(strain))
    if nan_count > 0:
        print(f"  Warning: {nan_count} NaN samples replaced with 0")
        strain = np.nan_to_num(strain, nan=0.0)

    info = {
        "detector": detector,
        "utc_start": utc_start,
        "sample_rate": sample_rate,
        "total_duration": total_duration,
        "segment_start": start_sec,
        "segment_duration": duration_sec,
        "samples": len(strain),
    }
    return strain, info


def bandpass(data, low, high, sample_rate):
    """Simple FFT bandpass filter."""
    n = len(data)
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    mask = (freqs >= low) & (freqs <= high)
    fft[~mask] = 0
    return np.fft.irfft(fft, n=n)


def normalize(data, target_peak=0.8):
    """Normalize to target peak amplitude."""
    peak = np.max(np.abs(data))
    if peak == 0:
        return data
    return data * (target_peak / peak)


def save_wav(path, audio, sample_rate=44100):
    """Save float64 audio array to 16-bit WAV file."""
    import wave
    import struct

    # clip to [-1, 1] and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    duration = len(audio) / sample_rate
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  WAV saved: {path}")
    print(f"  {duration:.1f}s, {sample_rate} Hz, 16-bit mono, {size_mb:.1f} MB")


def play_direct(strain, info, speed=1.0, bp_low=20, bp_high=2000, wav_path=None):
    """
    Direct playback: the raw strain IS the audio waveform.
    LIGO data at 4096 Hz is already in the audible range.
    We bandpass filter and normalize, then play through pyo.
    If wav_path is set, saves to WAV file instead of playing.
    """
    sr = info["sample_rate"]
    print(f"\n  Direct {'export' if wav_path else 'playback'} mode")
    print(f"  Detector: {info['detector']}, UTC: {info['utc_start']}")
    print(f"  {info['samples']:,} samples @ {sr} Hz = {info['segment_duration']}s")
    print(f"  Bandpass: {bp_low}–{bp_high} Hz")

    # bandpass filter
    filtered = bandpass(strain, bp_low, bp_high, sr)

    # normalize to audible level
    audio = normalize(filtered, target_peak=0.7)

    print(f"  Peak amplitude after processing: {np.max(np.abs(audio)):.4f}")

    # boot pyo server at the LIGO sample rate (or resample)
    playback_sr = 44100
    if sr != playback_sr:
        # resample using linear interpolation
        orig_time = np.linspace(0, 1, len(audio))
        new_len = int(len(audio) * playback_sr / sr)
        new_time = np.linspace(0, 1, new_len)
        audio = np.interp(new_time, orig_time, audio)
        print(f"  Resampled {sr} → {playback_sr} Hz ({new_len:,} samples)")

    # WAV export mode — write file and return
    if wav_path:
        save_wav(wav_path, audio, playback_sr)
        return

    from pyo import Server, DataTable, TableRead, pa_get_default_output

    dev = pa_get_default_output()
    s = Server(sr=playback_sr, duplex=0, audio="portaudio")
    s.setOutputDevice(dev)
    s.boot()
    s.start()

    # load into pyo DataTable
    table = DataTable(size=len(audio), init=audio.tolist())
    effective_speed = speed
    reader = TableRead(table, freq=effective_speed / (len(audio) / playback_sr),
                       loop=False, mul=0.9).out()

    playback_duration = len(audio) / playback_sr / speed
    print(f"  Playing {playback_duration:.1f}s (speed={speed}x)")
    print(f"  Press Ctrl+C to stop\n")

    # optional pygame window
    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((500, 100))
        pygame.display.set_caption(f"GW Sonifier — {info['detector']}")
        font = pygame.font.Font(None, 28)
        has_pygame = True
    except Exception:
        has_pygame = False

    t0 = time.time()
    try:
        while time.time() - t0 < playback_duration + 0.5:
            if has_pygame:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if ev.type == pygame.KEYDOWN:
                        if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                            raise KeyboardInterrupt

                elapsed = time.time() - t0
                pct = min(elapsed / playback_duration, 1.0)
                screen.fill((15, 15, 25))
                label = font.render(
                    f"{info['detector']} strain  {elapsed:.1f}s / {playback_duration:.1f}s",
                    True, (140, 200, 255))
                screen.blit(label, (15, 15))
                pygame.draw.rect(screen, (40, 40, 60), (15, 55, 470, 12))
                pygame.draw.rect(screen, (80, 160, 255), (15, 55, int(470 * pct), 12))
                # mini waveform
                chunk_i = int(pct * (len(audio) - 200))
                for x in range(200):
                    y = int(40 + audio[min(chunk_i + x * 2, len(audio) - 1)] * 25)
                    pygame.draw.line(screen, (60, 120, 80), (150 + x, 85), (150 + x, y))
                pygame.display.flip()

            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        s.stop()
        if has_pygame:
            pygame.quit()
        print("  Done.")


def play_spectral(strain, info, speed=1.0, window_sec=0.5, freq_range=(80, 3000),
                   wav_path=None):
    """
    Spectral sonification: break strain into windows, compute RMS energy
    and dominant frequency, map to sine wave parameters.
    If wav_path is set, renders offline to WAV instead of live playback.
    """
    from pyo import Server, Sine, Fader, pa_get_default_output

    sr = info["sample_rate"]
    window = int(window_sec * sr)
    n_windows = len(strain) // window

    print(f"\n  Spectral sonification mode")
    print(f"  {n_windows} windows of {window_sec}s each")

    # compute per-window features
    events = []
    for i in range(n_windows):
        chunk = strain[i * window:(i + 1) * window]
        if not np.any(np.isfinite(chunk)):
            continue

        rms = np.sqrt(np.mean(chunk ** 2))
        # spectral centroid as dominant frequency indicator
        fft_mag = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(len(chunk), d=1.0 / sr)
        total = np.sum(fft_mag)
        if total > 0:
            centroid = np.sum(freqs * fft_mag) / total
        else:
            centroid = 0

        events.append({
            "rms": rms,
            "centroid": centroid,
            "window_index": i,
        })

    if not events:
        print("  No valid windows found.")
        return

    # scale to audio parameters
    rms_vals = [e["rms"] for e in events]
    cent_vals = [e["centroid"] for e in events]
    rms_min, rms_max = min(rms_vals), max(rms_vals)
    cent_min, cent_max = min(cent_vals), max(cent_vals)

    def log_map(val, vmin, vmax, fmin, fmax):
        if vmax == vmin:
            return math.sqrt(fmin * fmax)
        t = max(0, min(1, (val - vmin) / (vmax - vmin)))
        return math.exp(math.log(max(fmin, 1)) + t * (math.log(max(fmax, 1)) - math.log(max(fmin, 1))))

    def lin_map(val, vmin, vmax, omin, omax):
        if vmax == vmin:
            return (omin + omax) / 2
        t = max(0, min(1, (val - vmin) / (vmax - vmin)))
        return omin + t * (omax - omin)

    print(f"  RMS range: {rms_min:.4e} – {rms_max:.4e}")
    print(f"  Centroid range: {cent_min:.1f} – {cent_max:.1f} Hz")
    print(f"  Freq output: {freq_range[0]}–{freq_range[1]} Hz")

    note_dur = window_sec / speed

    # WAV export mode — render offline
    if wav_path:
        playback_sr = 44100
        samples_per_note = int(note_dur * playback_sr)
        total_samples = samples_per_note * len(events)
        output = np.zeros(total_samples, dtype=np.float64)

        fade_in = int(min(0.01, note_dur * 0.1) * playback_sr)
        fade_out = int(min(0.03, note_dur * 0.2) * playback_sr)

        for idx, e in enumerate(events):
            freq = log_map(e["centroid"], cent_min, cent_max,
                           freq_range[0], freq_range[1])
            amp = lin_map(e["rms"], rms_min, rms_max, 0.02, 0.5)

            t = np.arange(samples_per_note) / playback_sr
            tone = np.sin(2 * np.pi * freq * t) * amp

            # apply fade envelope
            env = np.ones(samples_per_note)
            if fade_in > 0:
                env[:fade_in] = np.linspace(0, 1, fade_in)
            if fade_out > 0:
                env[-fade_out:] = np.linspace(1, 0, fade_out)
            tone *= env

            start = idx * samples_per_note
            output[start:start + samples_per_note] += tone

        output = normalize(output, target_peak=0.7)
        save_wav(wav_path, output, playback_sr)
        return

    # Live playback
    from pyo import Server, Sine, Fader, pa_get_default_output

    dev = pa_get_default_output()
    s = Server(duplex=0, audio="portaudio")
    s.setOutputDevice(dev)
    s.boot()
    s.start()

    try:
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((500, 120))
        pygame.display.set_caption(f"GW Spectral — {info['detector']}")
        font = pygame.font.Font(None, 26)
        has_pygame = True
    except Exception:
        has_pygame = False

    print(f"  Playing {len(events)} events ({note_dur:.3f}s each, speed={speed}x)")
    print(f"  Press Ctrl+C to stop\n")

    try:
        for idx, e in enumerate(events):
            if has_pygame:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_q, pygame.K_ESCAPE):
                        raise KeyboardInterrupt

            freq = log_map(e["centroid"], cent_min, cent_max,
                           freq_range[0], freq_range[1])
            amp = lin_map(e["rms"], rms_min, rms_max, 0.02, 0.5)

            fader = Fader(fadein=float(min(0.01, note_dur * 0.1)),
                          fadeout=float(min(0.03, note_dur * 0.2)),
                          dur=float(note_dur), mul=float(amp))
            sine = Sine(freq=freq, mul=fader).out()
            fader.play()

            if has_pygame:
                pct = idx / max(len(events) - 1, 1)
                screen.fill((15, 15, 25))
                txt = font.render(
                    f"{idx+1}/{len(events)}  f={freq:.0f}Hz  a={amp:.3f}  rms={e['rms']:.2e}",
                    True, (160, 220, 160))
                screen.blit(txt, (15, 15))
                pygame.draw.rect(screen, (40, 40, 60), (15, 50, 470, 10))
                pygame.draw.rect(screen, (100, 180, 255), (15, 50, int(470 * pct), 10))
                # frequency bar
                f_pct = (freq - freq_range[0]) / (freq_range[1] - freq_range[0])
                pygame.draw.rect(screen, (60, 60, 80), (15, 75, 470, 20))
                pygame.draw.rect(screen, (200, 120, 80), (15, 75, int(470 * f_pct), 20))
                freq_label = font.render(f"freq", True, (180, 180, 180))
                screen.blit(freq_label, (15, 100))
                pygame.display.flip()

            time.sleep(note_dur)
            sine.stop()
            fader.stop()

    except KeyboardInterrupt:
        pass
    finally:
        s.stop()
        if has_pygame:
            pygame.quit()
        print("  Done.")


def main():
    parser = argparse.ArgumentParser(description="Sonify LIGO gravitational wave data")
    parser.add_argument("hdf5_file", help="Path to GWOSC HDF5 file")
    parser.add_argument("--mode", choices=["direct", "spectral"], default="direct",
                        help="Sonification mode (default: direct)")
    parser.add_argument("--start", type=int, default=None,
                        help="Start time in seconds from file start (default: auto-find good segment)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Duration in seconds (default: 60 for direct, all for spectral)")
    parser.add_argument("--all", action="store_true",
                        help="Use entire file including data gaps (gaps become silence)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default: 1.0)")
    parser.add_argument("--bandpass", nargs=2, type=float, default=[20, 2000],
                        metavar=("LOW", "HIGH"),
                        help="Bandpass filter range in Hz (default: 20 2000)")
    parser.add_argument("--window", type=float, default=0.5,
                        help="Window size in seconds for spectral mode (default: 0.5)")
    parser.add_argument("--wav", type=str, default=None,
                        help="Export sonification to WAV file instead of playing")
    args = parser.parse_args()

    if args.all:
        strain, info = load_gw_strain(args.hdf5_file, start_sec=0,
                                       duration_sec=None)
    else:
        strain, info = load_gw_strain(args.hdf5_file, args.start, args.duration)

    if args.mode == "direct":
        play_direct(strain, info, speed=args.speed,
                    bp_low=args.bandpass[0], bp_high=args.bandpass[1],
                    wav_path=args.wav)
    else:
        play_spectral(strain, info, speed=args.speed, window_sec=args.window,
                      wav_path=args.wav)


if __name__ == "__main__":
    main()
