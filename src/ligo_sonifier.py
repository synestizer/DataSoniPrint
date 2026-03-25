#!/usr/bin/env python3
"""
ligo_sonifier.py — Standalone LIGO gravitational wave data sonification.

A visual + audio app that turns LIGO strain data into sound.
The DATA SPREAD slider controls how the strain maps to pitch:
  - Low spread: data compressed into a narrow melodic range (tonal, musical)
  - High spread: data spans a wide frequency range (raw, textural)

Usage:
    python src/ligo_sonifier.py
    python src/ligo_sonifier.py ~/Downloads/H-H1_GWOSC_*.hdf5
    python src/ligo_sonifier.py ~/Downloads/H-H1_GWOSC_*.hdf5 --duration 120
"""

import argparse
import os
import sys
import time
import math
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sonify_gw import load_gw_strain, bandpass, normalize

# ─── Audio engine ────────────────────────────────────────────────────────────
try:
    from pyo import (Server, SigTo, Sine, Fader, Freeverb, Delay, Noise,
                     Biquad, Pan, DataTable, TableRead, pa_list_devices,
                     pa_get_default_output)
except ImportError:
    print("pyo is required:  pip install pyo")
    sys.exit(1)

import pygame

# ─── Constants ───────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1100, 700
MARGIN = 40
FPS = 40
DEFAULT_HDF5 = os.path.expanduser(
    "~/Downloads/H-H1_GWOSC_O4a_4KHZ_R1-1368195072-4096.hdf5")

# Slider geometry
SL_W, SL_H, KNOB_R = 700, 22, 10
SL_X = MARGIN

# Colors
BG        = (10, 10, 18)
ACCENT    = (80, 180, 255)
ACCENT2   = (255, 140, 60)
ACCENT3   = (100, 255, 180)
DIM       = (50, 50, 65)
TXT       = (220, 220, 230)
TXT_DIM   = (140, 140, 160)


# ─── App state ───────────────────────────────────────────────────────────────
class State:
    strain = None          # raw strain array (float64)
    info = None            # metadata dict
    audio = None           # processed audio (float64, 44100 Hz)
    playback_sr = 44100

    # playback
    playing = False
    play_start = 0.0
    play_duration = 0.0
    reader = None
    table = None

    # sliders (0–1 normalized)
    spread = 0.35          # data spread: 0 = very melodic, 1 = raw wide
    volume = 0.7           # master volume
    speed = 0.5            # playback speed (maps to 0.25x–4x)
    reverb_mix = 0.3       # reverb wet/dry
    bp_low_norm = 0.0      # bandpass low  (maps 20–500 Hz)
    bp_high_norm = 1.0     # bandpass high (maps 500–2000 Hz)

    dragging = None        # which slider is being dragged


ST = State()


def speed_from_slider(v):
    """Map 0–1 slider to 0.25x–4x playback speed (log scale)."""
    return 0.25 * (16 ** v)   # 0→0.25, 0.5→1.0, 1.0→4.0


def bp_low_from_slider(v):
    return 20 + v * 480      # 20–500 Hz


def bp_high_from_slider(v):
    return 500 + v * 1500    # 500–2000 Hz


def spread_freq_range(spread_val):
    """Convert spread 0–1 to (center_hz, semitone_range).
    Low spread → tight cluster around a center note (melodic).
    High spread → wide frequency sweep (raw texture).
    """
    center = 440                           # A4
    semi_range = 2 + spread_val * 46       # 2 to 48 semitones (4 octaves)
    low = center * (2 ** (-semi_range / 24))
    high = center * (2 ** (semi_range / 24))
    return max(low, 30), min(high, 8000)


# ─── Audio processing ────────────────────────────────────────────────────────
def process_strain():
    """Re-process the raw strain with current slider values.
    Returns processed audio array at playback_sr."""
    if ST.strain is None:
        return None

    sr = ST.info["sample_rate"]
    lo = bp_low_from_slider(ST.bp_low_norm)
    hi = bp_high_from_slider(ST.bp_high_norm)
    if lo >= hi:
        hi = lo + 50

    filtered = bandpass(ST.strain, lo, hi, sr)
    audio = normalize(filtered, target_peak=0.75)

    # Apply spread: remap audio amplitude to frequency-modulated version
    # At spread=0 the waveform is compressed (melodic hum),
    # at spread=1 it passes through mostly unchanged (raw texture).
    spread = ST.spread
    if spread < 1.0:
        # Non-linear compression: raise amplitude toward center
        # Using soft clipping / power compression
        compress = 1.0 + (1.0 - spread) * 4.0   # exponent: 1 (raw) to 5 (very compressed)
        sign = np.sign(audio)
        audio = sign * (np.abs(audio) ** (1.0 / compress))
        audio = normalize(audio, target_peak=0.75)

    # Resample to playback rate
    if sr != ST.playback_sr:
        orig_t = np.linspace(0, 1, len(audio))
        new_len = int(len(audio) * ST.playback_sr / sr)
        new_t = np.linspace(0, 1, new_len)
        audio = np.interp(new_t, orig_t, audio)

    return audio


# ─── pyo server & playback ───────────────────────────────────────────────────
server = None
VOL_SIG = None
REVERB = None


def boot_audio():
    global server, VOL_SIG, REVERB
    dev = pa_get_default_output()
    server = Server(sr=ST.playback_sr, duplex=0, buffersize=4096, audio="portaudio")
    server.setOutputDevice(dev)
    server.boot()
    server.start()
    VOL_SIG = SigTo(value=ST.volume, time=0.08, init=ST.volume)


def start_playback():
    """Process strain and start playing through pyo."""
    stop_playback()

    audio = process_strain()
    if audio is None or len(audio) == 0:
        return
    ST.audio = audio

    spd = speed_from_slider(ST.speed)

    ST.table = DataTable(size=len(audio), init=audio.tolist())
    freq = spd / (len(audio) / ST.playback_sr)

    # Play through reverb
    ST.reader = TableRead(ST.table, freq=freq, loop=True, mul=0.9 * VOL_SIG)
    reverb_amt = ST.reverb_mix
    dry = ST.reader * (1.0 - reverb_amt * 0.5)
    wet = Freeverb(ST.reader, size=0.88, damp=0.5, bal=1.0, mul=reverb_amt * 0.6)
    dry.out()
    wet.out()

    ST.play_duration = len(audio) / ST.playback_sr / spd
    ST.play_start = time.time()
    ST.playing = True


def stop_playback():
    if ST.reader:
        try:
            ST.reader.stop()
        except Exception:
            pass
    ST.reader = None
    ST.table = None
    ST.playing = False


def restart_playback():
    """Restart playback with updated parameters (called when sliders change)."""
    if ST.playing:
        threading.Thread(target=start_playback, daemon=True).start()


# ─── Waveform cache for display ─────────────────────────────────────────────
_wave_cache = {"spread": None, "points": None}


def get_waveform_points(rect, n_points=400):
    """Downsample audio for waveform display. Cache by spread value."""
    if ST.audio is None:
        return []
    if _wave_cache["spread"] == ST.spread and _wave_cache["points"] is not None:
        return _wave_cache["points"]

    audio = ST.audio
    step = max(1, len(audio) // n_points)
    samples = audio[::step][:n_points]
    x0, y_center = rect.x, rect.centery
    w, h = rect.width, rect.height
    points = []
    for i, s in enumerate(samples):
        x = x0 + int(i / len(samples) * w)
        y = int(y_center - s * h * 0.45)
        points.append((x, y))

    _wave_cache["spread"] = ST.spread
    _wave_cache["points"] = points
    return points


# ─── UI ──────────────────────────────────────────────────────────────────────
SLIDERS = []   # populated in main


def draw_slider(screen, font, label, y, val, color, tag):
    """Draw one slider. Returns (tag, rect) for hit testing."""
    # label
    lbl = font.render(label, True, color)
    screen.blit(lbl, (SL_X, y - 22))
    # track
    rect = pygame.Rect(SL_X, y, SL_W, SL_H)
    pygame.draw.rect(screen, DIM, rect, border_radius=4)
    # fill
    fill_w = int(SL_W * val)
    pygame.draw.rect(screen, color, (SL_X, y, fill_w, SL_H), border_radius=4)
    # knob
    kx = SL_X + fill_w
    ky = y + SL_H // 2
    pygame.draw.circle(screen, (240, 240, 240), (kx, ky), KNOB_R)
    # value text
    screen.blit(font.render(f"{val:.2f}", True, TXT_DIM), (SL_X + SL_W + 12, y))
    return (tag, rect)


def draw(screen, fonts):
    font, small = fonts
    screen.fill(BG)

    # ── Title bar ────────────────────────────────────────────────────────
    title = "LIGO Gravitational Wave Sonifier"
    screen.blit(font.render(title, True, ACCENT), (MARGIN, 14))
    if ST.info:
        det = ST.info["detector"]
        utc = ST.info["utc_start"]
        dur = ST.info["segment_duration"]
        meta = f"{det}  |  {utc}  |  {dur}s segment  |  {ST.info['sample_rate']} Hz"
        screen.blit(small.render(meta, True, TXT_DIM), (MARGIN, 44))

    # ── Waveform display ─────────────────────────────────────────────────
    wave_rect = pygame.Rect(MARGIN, 70, WIDTH - MARGIN * 2, 180)
    pygame.draw.rect(screen, (18, 18, 28), wave_rect, border_radius=6)
    pygame.draw.rect(screen, DIM, wave_rect, width=1, border_radius=6)
    # center line
    pygame.draw.line(screen, (35, 35, 50),
                     (wave_rect.x, wave_rect.centery),
                     (wave_rect.right, wave_rect.centery))

    pts = get_waveform_points(wave_rect)
    if len(pts) > 1:
        pygame.draw.lines(screen, ACCENT, False, pts, 2)

    # playback cursor
    if ST.playing:
        elapsed = time.time() - ST.play_start
        spd = speed_from_slider(ST.speed)
        loop_dur = len(ST.audio) / ST.playback_sr / spd if ST.audio is not None else 1
        pct = (elapsed % loop_dur) / loop_dur if loop_dur > 0 else 0
        cx = wave_rect.x + int(pct * wave_rect.width)
        pygame.draw.line(screen, ACCENT2, (cx, wave_rect.y), (cx, wave_rect.bottom), 2)

    # ── Spread info ──────────────────────────────────────────────────────
    flo, fhi = spread_freq_range(ST.spread)
    spread_desc = f"Freq range: {flo:.0f}–{fhi:.0f} Hz"
    if ST.spread < 0.2:
        spread_desc += "  (very melodic)"
    elif ST.spread < 0.5:
        spread_desc += "  (melodic)"
    elif ST.spread < 0.8:
        spread_desc += "  (textural)"
    else:
        spread_desc += "  (raw)"

    # ── Sliders ──────────────────────────────────────────────────────────
    sy = 290
    gap = 58

    SLIDERS.clear()
    SLIDERS.append(draw_slider(screen, small,
        f"DATA SPREAD — {spread_desc}", sy, ST.spread, ACCENT, "spread"))
    sy += gap
    SLIDERS.append(draw_slider(screen, small,
        f"VOLUME", sy, ST.volume, (200, 200, 210), "volume"))
    sy += gap
    spd = speed_from_slider(ST.speed)
    SLIDERS.append(draw_slider(screen, small,
        f"SPEED  ({spd:.2f}x)", sy, ST.speed, ACCENT2, "speed"))
    sy += gap
    SLIDERS.append(draw_slider(screen, small,
        f"REVERB", sy, ST.reverb_mix, ACCENT3, "reverb"))
    sy += gap
    lo_hz = bp_low_from_slider(ST.bp_low_norm)
    SLIDERS.append(draw_slider(screen, small,
        f"BANDPASS LOW  ({lo_hz:.0f} Hz)", sy, ST.bp_low_norm, (180, 130, 255), "bp_low"))
    sy += gap
    hi_hz = bp_high_from_slider(ST.bp_high_norm)
    SLIDERS.append(draw_slider(screen, small,
        f"BANDPASS HIGH  ({hi_hz:.0f} Hz)", sy, ST.bp_high_norm, (180, 130, 255), "bp_high"))

    # ── Buttons ──────────────────────────────────────────────────────────
    btn_y = sy + gap + 10
    # Play / Stop
    play_rect = pygame.Rect(MARGIN, btn_y, 160, 40)
    play_color = ACCENT2 if ST.playing else (40, 110, 170)
    play_border = (255, 180, 100) if ST.playing else ACCENT
    pygame.draw.rect(screen, play_color, play_rect, border_radius=6)
    pygame.draw.rect(screen, play_border, play_rect, width=2, border_radius=6)
    play_label = "■  Stop" if ST.playing else "▶  Play"
    screen.blit(font.render(play_label, True, TXT), (play_rect.x + 30, play_rect.y + 8))

    # Reprocess
    reproc_rect = pygame.Rect(MARGIN + 180, btn_y, 200, 40)
    pygame.draw.rect(screen, (40, 70, 60), reproc_rect, border_radius=6)
    pygame.draw.rect(screen, ACCENT3, reproc_rect, width=2, border_radius=6)
    screen.blit(font.render("↻  Reprocess", True, TXT), (reproc_rect.x + 25, reproc_rect.y + 8))

    # Status
    if ST.playing and ST.audio is not None:
        elapsed = time.time() - ST.play_start
        spd_val = speed_from_slider(ST.speed)
        loop_dur = len(ST.audio) / ST.playback_sr / spd_val
        pct = (elapsed % loop_dur) / loop_dur * 100 if loop_dur > 0 else 0
        stat = f"Playing  {elapsed:.1f}s  ({pct:.0f}% of loop)"
        screen.blit(small.render(stat, True, ACCENT2), (MARGIN + 400, btn_y + 12))
    elif ST.strain is not None:
        screen.blit(small.render("Ready — press Play", True, TXT_DIM), (MARGIN + 400, btn_y + 12))

    # ── Key hints ────────────────────────────────────────────────────────
    hints = "SPACE: play/stop    R: reprocess    Q/ESC: quit"
    screen.blit(small.render(hints, True, (80, 80, 100)), (MARGIN, HEIGHT - 28))

    pygame.display.flip()
    return play_rect, reproc_rect


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LIGO Gravitational Wave Sonifier")
    parser.add_argument("hdf5_file", nargs="?", default=DEFAULT_HDF5,
                        help="Path to GWOSC HDF5 file")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────────────
    if not os.path.isfile(args.hdf5_file):
        print(f"File not found: {args.hdf5_file}")
        sys.exit(1)

    print(f"Loading {os.path.basename(args.hdf5_file)}...")
    ST.strain, ST.info = load_gw_strain(args.hdf5_file, args.start, args.duration)
    print(f"  {ST.info['detector']} | {ST.info['samples']:,} samples "
          f"@ {ST.info['sample_rate']} Hz | {ST.info['segment_duration']}s")

    # initial processing
    ST.audio = process_strain()
    _wave_cache["spread"] = None   # reset cache

    # ── Boot audio ───────────────────────────────────────────────────────
    boot_audio()

    # ── Pygame ───────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LIGO Sonifier")
    font = pygame.font.SysFont("DejaVuSans", 22)
    small = pygame.font.SysFont("DejaVuSans", 15)
    clock = pygame.time.Clock()

    running = True
    needs_reprocess = False    # flag to defer reprocessing until mouse-up

    while running:
        play_rect, reproc_rect = draw(screen, (font, small))

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    if ST.playing:
                        stop_playback()
                    else:
                        threading.Thread(target=start_playback, daemon=True).start()
                elif ev.key == pygame.K_r:
                    ST.audio = process_strain()
                    _wave_cache["spread"] = None
                    if ST.playing:
                        restart_playback()

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                # buttons
                if play_rect.collidepoint(mx, my):
                    if ST.playing:
                        stop_playback()
                    else:
                        threading.Thread(target=start_playback, daemon=True).start()
                    continue
                if reproc_rect.collidepoint(mx, my):
                    ST.audio = process_strain()
                    _wave_cache["spread"] = None
                    if ST.playing:
                        restart_playback()
                    continue
                # slider hit test
                for tag, rect in SLIDERS:
                    if rect.collidepoint(mx, my):
                        ST.dragging = tag
                        break

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if ST.dragging:
                    # defer reprocess until release to avoid stutter
                    if ST.dragging in ("spread", "bp_low", "bp_high"):
                        needs_reprocess = True
                    ST.dragging = None

            elif ev.type == pygame.MOUSEMOTION and ST.dragging:
                mx, _ = ev.pos
                val = (mx - SL_X) / float(SL_W)
                val = max(0.0, min(1.0, val))

                if ST.dragging == "spread":
                    ST.spread = val
                    _wave_cache["spread"] = None
                elif ST.dragging == "volume":
                    ST.volume = val
                    if VOL_SIG:
                        VOL_SIG.time = 0.05
                        VOL_SIG.setValue(val)
                elif ST.dragging == "speed":
                    ST.speed = val
                elif ST.dragging == "reverb":
                    ST.reverb_mix = val
                elif ST.dragging == "bp_low":
                    ST.bp_low_norm = val
                elif ST.dragging == "bp_high":
                    ST.bp_high_norm = val

        # Deferred reprocess (only on mouse release)
        if needs_reprocess:
            needs_reprocess = False
            ST.audio = process_strain()
            _wave_cache["spread"] = None
            if ST.playing:
                restart_playback()

        clock.tick(FPS)

    # ── Shutdown ─────────────────────────────────────────────────────────
    stop_playback()
    if server:
        server.stop()
        server.shutdown()
    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
