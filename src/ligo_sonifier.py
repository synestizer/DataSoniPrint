#!/usr/bin/env python3
"""
DataSoniPrint — LIGO gravitational wave sonification & 3D visualization.

Three output panels:
  1. Raw strain waveform graph
  2. Spectrogram with playback cursor
  3. 3D terrain preview (exportable as STL for 3D printing)

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
from visualize_gw import compute_spectrogram, spectrogram_to_stl

try:
    from pyo import (Server, SigTo, Freeverb, DataTable, TableRead,
                     pa_get_default_output)
except ImportError:
    sys.exit("pyo is required:  pip install pyo")

import pygame

# ─── Layout ──────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1500, 950
MARGIN = 30
FPS = 40
DEFAULT_HDF5 = os.path.expanduser(
    "~/Downloads/H-H1_GWOSC_O4a_4KHZ_R1-1368195072-4096.hdf5")

# Panel geometry
PANEL_W = WIDTH - MARGIN * 2           # 1440
WAVE_H = 150
VIS_H = 250
WAVE_Y = 70
VIS_Y = WAVE_Y + WAVE_H + 20          # 240
SPEC_W = (PANEL_W - 20) // 2           # 710
P3D_W = PANEL_W - SPEC_W - 20          # 710
SPEC_X = MARGIN
P3D_X = MARGIN + SPEC_W + 20

# Slider geometry
SL_Y0 = VIS_Y + VIS_H + 30            # 520
SL_W, SL_H, KNOB_R = 900, 20, 9
SL_X = MARGIN
SL_GAP = 46

# Colors
BG       = (10, 10, 18)
PANEL_BG = (18, 18, 28)
ACCENT   = (80, 180, 255)
ACCENT2  = (255, 140, 60)
ACCENT3  = (100, 255, 180)
DIM      = (50, 50, 65)
TXT      = (220, 220, 230)
TXT_DIM  = (140, 140, 160)


# ─── Inferno colormap (256 entries) ──────────────────────────────────────────
def _build_inferno_lut():
    """Build a 256-entry inferno-like colormap via control-point interpolation."""
    pts = [
        (0.00,   0,   0,   4),
        (0.13,  40,  11,  84),
        (0.25,  87,  16, 110),
        (0.38, 137,  30,  93),
        (0.50, 188,  55,  84),
        (0.63, 227,  89,  51),
        (0.75, 249, 142,   9),
        (0.88, 252, 201,  38),
        (1.00, 252, 255, 164),
    ]
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        for j in range(len(pts) - 1):
            if pts[j][0] <= t <= pts[j + 1][0]:
                f = (t - pts[j][0]) / (pts[j + 1][0] - pts[j][0])
                lut[i] = [int(pts[j][k] + f * (pts[j + 1][k] - pts[j][k]))
                          for k in (1, 2, 3)]
                break
    return lut


INFERNO = _build_inferno_lut()


# ─── App state ───────────────────────────────────────────────────────────────
class State:
    strain = None          # raw strain (float64)
    info = None            # metadata dict
    audio = None           # processed audio (float64, 44100 Hz)
    playback_sr = 44100

    # playback
    playing = False
    play_start = 0.0
    reader = None
    table = None

    # sliders (0–1 normalized)
    spread = 0.35
    volume = 0.7
    speed = 0.5
    reverb_mix = 0.3
    bp_low_norm = 0.0
    bp_high_norm = 1.0
    dragging = None

    # visualization surfaces
    spec_surface = None     # pygame Surface — spectrogram panel
    preview_3d = None       # pygame Surface — 3D wireframe panel

    # spectrogram data (for STL export)
    spec_times = None
    spec_freqs = None
    spec_db = None

    # STL export
    stl_busy = False
    stl_status = ""


ST = State()


# ─── Slider helpers ──────────────────────────────────────────────────────────
def speed_from_slider(v):
    return 0.25 * (16 ** v)          # 0→0.25x, 0.5→1x, 1→4x


def bp_low_from_slider(v):
    return 10 + v * 490              # 10–500 Hz


def bp_high_from_slider(v):
    return 200 + v * 3800            # 200–4000 Hz


def spread_freq_range(spread_val):
    center = 440
    semi_range = 2 + spread_val * 46
    low = center * (2 ** (-semi_range / 24))
    high = center * (2 ** (semi_range / 24))
    return max(low, 30), min(high, 8000)


# ─── Audio processing ────────────────────────────────────────────────────────
def process_strain():
    """Bandpass → normalize → spread compression → resample."""
    if ST.strain is None:
        return None

    sr = ST.info["sample_rate"]
    lo = bp_low_from_slider(ST.bp_low_norm)
    hi = bp_high_from_slider(ST.bp_high_norm)
    if lo >= hi:
        hi = lo + 50

    filtered = bandpass(ST.strain, lo, hi, sr)
    audio = normalize(filtered, target_peak=0.75)

    spread = ST.spread
    if spread < 1.0:
        compress = 1.0 + (1.0 - spread) * 4.0
        sign = np.sign(audio)
        audio = sign * (np.abs(audio) ** (1.0 / compress))
        audio = normalize(audio, target_peak=0.75)

    if sr != ST.playback_sr:
        orig_t = np.linspace(0, 1, len(audio))
        new_len = int(len(audio) * ST.playback_sr / sr)
        new_t = np.linspace(0, 1, new_len)
        audio = np.interp(new_t, orig_t, audio)

    return audio


# ─── Visual computation ──────────────────────────────────────────────────────
def recompute_visuals():
    """Recompute spectrogram surface + 3D wireframe preview from current audio."""
    if ST.audio is None:
        return

    freq_max = 2000
    spec_inner_w = SPEC_W - 8
    spec_inner_h = VIS_H - 24
    p3d_inner_w = P3D_W - 8
    p3d_inner_h = VIS_H - 24

    # ── Spectrogram ─────────────────────────────────────────────────────
    print("  Computing spectrogram...")
    times, freqs, spec_db = compute_spectrogram(
        ST.audio, ST.playback_sr, nperseg=1024, overlap_frac=0.75)

    ST.spec_times = times
    ST.spec_freqs = freqs
    ST.spec_db = spec_db

    freq_mask = freqs <= freq_max
    spec_clip = spec_db[freq_mask, :]

    # Normalize → 0-255 index → inferno color
    vmin = np.percentile(spec_clip, 5)
    vmax = np.percentile(spec_clip, 99)
    norm = np.clip((spec_clip - vmin) / (vmax - vmin + 1e-10), 0, 1)
    indices = (norm * 255).astype(np.uint8)
    rgb = INFERNO[indices][::-1, :, :]       # flip so low freq is at bottom

    # surfarray expects (width, height, 3)
    rgb_t = np.ascontiguousarray(rgb.transpose(1, 0, 2))
    raw_surf = pygame.surfarray.make_surface(rgb_t)
    ST.spec_surface = pygame.transform.smoothscale(
        raw_surf, (spec_inner_w, spec_inner_h))

    # ── 3D wireframe ────────────────────────────────────────────────────
    print("  Computing 3D preview...")
    spec_3d = spec_db[freq_mask, :]
    gx, gy = 60, 35
    step_t = max(1, spec_3d.shape[1] // gx)
    step_f = max(1, spec_3d.shape[0] // gy)
    grid = spec_3d[::step_f, ::step_t]
    ny, nx = grid.shape

    gmin = np.percentile(grid, 5)
    gmax = np.percentile(grid, 99)
    h = np.clip((grid - gmin) / (gmax - gmin + 1e-10), 0, 1)

    p3d_surf = pygame.Surface((p3d_inner_w, p3d_inner_h))
    p3d_surf.fill(PANEL_BG)

    # isometric projection
    cx_3d = p3d_inner_w * 0.48
    cy_3d = p3d_inner_h * 0.72
    sx = p3d_inner_w * 0.4 / max(nx, 1)
    sy = p3d_inner_h * 0.22 / max(ny, 1)
    sz = p3d_inner_h * 0.38
    ang = math.radians(30)
    ca, sa = math.cos(ang), math.sin(ang)

    def proj(ix, iy, hv):
        x3 = (ix - nx / 2) * sx
        y3 = (iy - ny / 2) * sy
        z3 = hv * sz
        return int(cx_3d + x3 * ca - y3 * ca), int(cy_3d - x3 * sa - y3 * sa - z3)

    # draw wireframe back-to-front
    for iy in range(ny - 1, -1, -1):
        for ix in range(nx - 1):
            ci = int(h[iy, ix] * 255)
            col = tuple(int(c) for c in INFERNO[min(255, max(0, ci))])
            p0 = proj(ix, iy, h[iy, ix])
            p1 = proj(ix + 1, iy, h[iy, ix + 1])
            pygame.draw.line(p3d_surf, col, p0, p1, 1)
        if iy < ny - 1:
            for ix in range(nx):
                ci = int(h[iy, ix] * 255)
                col = tuple(int(c) for c in INFERNO[min(255, max(0, ci))])
                p0 = proj(ix, iy, h[iy, ix])
                p1 = proj(ix, iy + 1, h[iy + 1, ix])
                pygame.draw.line(p3d_surf, col, p0, p1, 1)

    ST.preview_3d = p3d_surf
    print("  Visuals ready.")


# ─── pyo server & playback ───────────────────────────────────────────────────
server = None
VOL_SIG = None
SPEED_SIG = None


def _base_freq():
    """Base table-read frequency for 1x speed."""
    if ST.audio is not None and len(ST.audio) > 0:
        return 1.0 / (len(ST.audio) / ST.playback_sr)
    return 1.0


def boot_audio():
    global server, VOL_SIG, SPEED_SIG
    dev = pa_get_default_output()
    server = Server(sr=ST.playback_sr, duplex=0, buffersize=4096, audio="portaudio")
    server.setOutputDevice(dev)
    server.boot()
    server.start()
    VOL_SIG = SigTo(value=ST.volume, time=0.08, init=ST.volume)
    spd = speed_from_slider(ST.speed)
    SPEED_SIG = SigTo(value=_base_freq() * spd, time=0.15,
                      init=_base_freq() * spd)


def start_playback():
    stop_playback()
    audio = ST.audio
    if audio is None or len(audio) == 0:
        return

    spd = speed_from_slider(ST.speed)
    ST.table = DataTable(size=len(audio), init=audio.tolist())
    bf = _base_freq()
    SPEED_SIG.setValue(bf * spd)

    ST.reader = TableRead(ST.table, freq=SPEED_SIG, loop=True, mul=0.9 * VOL_SIG)
    rev_amt = ST.reverb_mix
    dry = ST.reader * (1.0 - rev_amt * 0.5)
    wet = Freeverb(ST.reader, size=0.88, damp=0.5, bal=1.0, mul=rev_amt * 0.6)
    dry.out()
    wet.out()

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
    if ST.playing:
        threading.Thread(target=start_playback, daemon=True).start()


# ─── Waveform cache ─────────────────────────────────────────────────────────
_wave_cache = {"key": None, "points": None}


def get_waveform_points(rect, data, n_points=600):
    """Downsample array for waveform display, cached by data identity."""
    if data is None or len(data) == 0:
        return []
    key = id(data)
    if _wave_cache["key"] == key and _wave_cache["points"] is not None:
        return _wave_cache["points"]

    step = max(1, len(data) // n_points)
    samples = data[::step][:n_points]
    x0, yc = rect.x, rect.centery
    w, h2 = rect.width, rect.height
    points = [(x0 + int(i / len(samples) * w),
               int(yc - s * h2 * 0.45))
              for i, s in enumerate(samples)]

    _wave_cache["key"] = key
    _wave_cache["points"] = points
    return points


# ─── STL export ──────────────────────────────────────────────────────────────
def export_stl_threaded():
    if ST.spec_times is None or ST.stl_busy:
        return
    ST.stl_busy = True
    ST.stl_status = "Exporting STL..."

    def _do():
        try:
            export_dir = os.path.expanduser("~/DataSoniPrint/exports")
            os.makedirs(export_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out = os.path.join(export_dir, f"gw_terrain_{ts}.stl")
            spectrogram_to_stl(ST.spec_times, ST.spec_freqs, ST.spec_db,
                               output_path=out, freq_max=2000)
            ST.stl_status = f"Saved: {os.path.basename(out)}"
        except ImportError:
            ST.stl_status = "Error: pip install numpy-stl"
        except Exception as e:
            ST.stl_status = f"Error: {e}"
        finally:
            ST.stl_busy = False

    threading.Thread(target=_do, daemon=True).start()


# ─── UI drawing ──────────────────────────────────────────────────────────────
SLIDERS = []


def draw_slider(screen, font, label, y, val, color, tag):
    """Draw one slider, return (tag, rect) for hit testing."""
    lbl = font.render(label, True, color)
    screen.blit(lbl, (SL_X, y - 20))
    rect = pygame.Rect(SL_X, y, SL_W, SL_H)
    pygame.draw.rect(screen, DIM, rect, border_radius=4)
    fill_w = int(SL_W * val)
    pygame.draw.rect(screen, color, (SL_X, y, fill_w, SL_H), border_radius=4)
    kx = SL_X + fill_w
    ky = y + SL_H // 2
    pygame.draw.circle(screen, (240, 240, 240), (kx, ky), KNOB_R)
    screen.blit(font.render(f"{val:.2f}", True, TXT_DIM), (SL_X + SL_W + 12, y))
    return (tag, rect)


def draw(screen, fonts):
    font, small, tiny = fonts
    screen.fill(BG)

    # ── Title bar ────────────────────────────────────────────────────────
    screen.blit(font.render(
        "DataSoniPrint \u2014 LIGO Gravitational Wave Sonifier",
        True, ACCENT), (MARGIN, 12))
    if ST.info:
        det = ST.info["detector"]
        utc = ST.info["utc_start"]
        dur = ST.info["segment_duration"]
        meta = f"{det}  |  {utc}  |  {dur}s segment  |  {ST.info['sample_rate']} Hz"
        screen.blit(small.render(meta, True, TXT_DIM), (MARGIN, 42))

    # ── Panel 1: Raw strain waveform ─────────────────────────────────────
    wave_rect = pygame.Rect(MARGIN, WAVE_Y, PANEL_W, WAVE_H)
    pygame.draw.rect(screen, PANEL_BG, wave_rect, border_radius=6)
    pygame.draw.rect(screen, DIM, wave_rect, width=1, border_radius=6)
    screen.blit(tiny.render("RAW STRAIN WAVEFORM", True, TXT_DIM),
                (MARGIN + 8, WAVE_Y + 4))
    # center line
    pygame.draw.line(screen, (30, 30, 45),
                     (wave_rect.x, wave_rect.centery),
                     (wave_rect.right, wave_rect.centery))
    # waveform
    pts = get_waveform_points(wave_rect, ST.strain)
    if len(pts) > 1:
        pygame.draw.lines(screen, ACCENT, False, pts, 2)
    # playback cursor
    if ST.playing and ST.audio is not None:
        elapsed = time.time() - ST.play_start
        spd = speed_from_slider(ST.speed)
        loop_dur = len(ST.audio) / ST.playback_sr / spd
        pct = (elapsed % loop_dur) / loop_dur if loop_dur > 0 else 0
        cx = wave_rect.x + int(pct * wave_rect.width)
        pygame.draw.line(screen, ACCENT2,
                         (cx, wave_rect.y), (cx, wave_rect.bottom), 2)

    # ── Panel 2: Spectrogram ─────────────────────────────────────────────
    spec_rect = pygame.Rect(SPEC_X, VIS_Y, SPEC_W, VIS_H)
    pygame.draw.rect(screen, PANEL_BG, spec_rect, border_radius=6)
    pygame.draw.rect(screen, DIM, spec_rect, width=1, border_radius=6)
    screen.blit(tiny.render("SPECTROGRAM", True, TXT_DIM),
                (SPEC_X + 8, VIS_Y + 4))

    spec_inner = pygame.Rect(SPEC_X + 4, VIS_Y + 20, SPEC_W - 8, VIS_H - 24)
    if ST.spec_surface is not None:
        screen.blit(ST.spec_surface, spec_inner.topleft)
        # playback cursor on spectrogram
        if ST.playing and ST.audio is not None:
            elapsed = time.time() - ST.play_start
            spd = speed_from_slider(ST.speed)
            loop_dur = len(ST.audio) / ST.playback_sr / spd
            pct = (elapsed % loop_dur) / loop_dur if loop_dur > 0 else 0
            cx = spec_inner.x + int(pct * spec_inner.width)
            pygame.draw.line(screen, (255, 255, 255),
                             (cx, spec_inner.y), (cx, spec_inner.bottom), 2)
        # frequency axis labels
        screen.blit(tiny.render("2 kHz", True, TXT_DIM),
                    (SPEC_X + SPEC_W - 48, VIS_Y + 20))
        screen.blit(tiny.render("0 Hz", True, TXT_DIM),
                    (SPEC_X + SPEC_W - 40, VIS_Y + VIS_H - 16))
    else:
        screen.blit(small.render("Processing...", True, TXT_DIM),
                    (SPEC_X + SPEC_W // 2 - 40, VIS_Y + VIS_H // 2))

    # ── Panel 3: 3D terrain preview ──────────────────────────────────────
    p3d_rect = pygame.Rect(P3D_X, VIS_Y, P3D_W, VIS_H)
    pygame.draw.rect(screen, PANEL_BG, p3d_rect, border_radius=6)
    pygame.draw.rect(screen, DIM, p3d_rect, width=1, border_radius=6)
    screen.blit(tiny.render("3D TERRAIN PREVIEW  (STL-exportable)", True, TXT_DIM),
                (P3D_X + 8, VIS_Y + 4))

    if ST.preview_3d is not None:
        screen.blit(ST.preview_3d, (P3D_X + 4, VIS_Y + 20))
    else:
        screen.blit(small.render("Processing...", True, TXT_DIM),
                    (P3D_X + P3D_W // 2 - 40, VIS_Y + VIS_H // 2))

    # ── Spread info ──────────────────────────────────────────────────────
    flo, fhi = spread_freq_range(ST.spread)
    spread_lbl = f"Freq range: {flo:.0f}\u2013{fhi:.0f} Hz"
    if ST.spread < 0.2:
        spread_lbl += "  (very melodic)"
    elif ST.spread < 0.5:
        spread_lbl += "  (melodic)"
    elif ST.spread < 0.8:
        spread_lbl += "  (textural)"
    else:
        spread_lbl += "  (raw)"

    # ── Sliders ──────────────────────────────────────────────────────────
    sy = SL_Y0
    SLIDERS.clear()
    SLIDERS.append(draw_slider(screen, small,
        f"DATA SPREAD \u2014 {spread_lbl}", sy, ST.spread, ACCENT, "spread"))
    sy += SL_GAP
    SLIDERS.append(draw_slider(screen, small,
        "VOLUME", sy, ST.volume, (200, 200, 210), "volume"))
    sy += SL_GAP
    spd = speed_from_slider(ST.speed)
    SLIDERS.append(draw_slider(screen, small,
        f"SPEED  ({spd:.2f}x)", sy, ST.speed, ACCENT2, "speed"))
    sy += SL_GAP
    SLIDERS.append(draw_slider(screen, small,
        "REVERB", sy, ST.reverb_mix, ACCENT3, "reverb"))
    sy += SL_GAP
    lo_hz = bp_low_from_slider(ST.bp_low_norm)
    SLIDERS.append(draw_slider(screen, small,
        f"BANDPASS LOW  ({lo_hz:.0f} Hz)", sy, ST.bp_low_norm,
        (180, 130, 255), "bp_low"))
    sy += SL_GAP
    hi_hz = bp_high_from_slider(ST.bp_high_norm)
    SLIDERS.append(draw_slider(screen, small,
        f"BANDPASS HIGH  ({hi_hz:.0f} Hz)", sy, ST.bp_high_norm,
        (180, 130, 255), "bp_high"))

    # ── Buttons ──────────────────────────────────────────────────────────
    btn_y = sy + SL_GAP

    # Play / Stop
    play_rect = pygame.Rect(MARGIN, btn_y, 160, 40)
    play_col = ACCENT2 if ST.playing else (40, 110, 170)
    play_brd = (255, 180, 100) if ST.playing else ACCENT
    pygame.draw.rect(screen, play_col, play_rect, border_radius=6)
    pygame.draw.rect(screen, play_brd, play_rect, width=2, border_radius=6)
    play_lbl = "\u25a0  Stop" if ST.playing else "\u25b6  Play"
    screen.blit(font.render(play_lbl, True, TXT),
                (play_rect.x + 30, play_rect.y + 8))

    # Reprocess
    reproc_rect = pygame.Rect(MARGIN + 180, btn_y, 200, 40)
    pygame.draw.rect(screen, (40, 70, 60), reproc_rect, border_radius=6)
    pygame.draw.rect(screen, ACCENT3, reproc_rect, width=2, border_radius=6)
    screen.blit(font.render("\u21bb  Reprocess", True, TXT),
                (reproc_rect.x + 20, reproc_rect.y + 8))

    # Export STL
    stl_rect = pygame.Rect(MARGIN + 400, btn_y, 200, 40)
    stl_col = (60, 50, 70) if not ST.stl_busy else (40, 40, 50)
    pygame.draw.rect(screen, stl_col, stl_rect, border_radius=6)
    pygame.draw.rect(screen, (180, 130, 255), stl_rect, width=2, border_radius=6)
    stl_text = "Exporting..." if ST.stl_busy else "Export STL"
    screen.blit(font.render(stl_text, True, TXT),
                (stl_rect.x + 35, stl_rect.y + 8))

    # Status line
    if ST.stl_status:
        screen.blit(small.render(ST.stl_status, True, (180, 130, 255)),
                    (MARGIN + 620, btn_y + 12))
    elif ST.playing and ST.audio is not None:
        elapsed = time.time() - ST.play_start
        spd_val = speed_from_slider(ST.speed)
        loop_dur = len(ST.audio) / ST.playback_sr / spd_val
        pct = (elapsed % loop_dur) / loop_dur * 100 if loop_dur > 0 else 0
        stat = f"Playing  {elapsed:.1f}s  ({pct:.0f}% of loop)"
        screen.blit(small.render(stat, True, ACCENT2), (MARGIN + 620, btn_y + 12))
    elif ST.strain is not None:
        screen.blit(small.render("Ready \u2014 press Play", True, TXT_DIM),
                    (MARGIN + 620, btn_y + 12))

    # Key hints
    hints = "SPACE: play/stop    R: reprocess    E: export STL    Q/ESC: quit"
    screen.blit(tiny.render(hints, True, (80, 80, 100)), (MARGIN, HEIGHT - 25))

    pygame.display.flip()
    return play_rect, reproc_rect, stl_rect


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DataSoniPrint — LIGO Sonifier")
    parser.add_argument("hdf5_file", nargs="?", default=DEFAULT_HDF5,
                        help="Path to GWOSC HDF5 file")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_file):
        sys.exit(f"File not found: {args.hdf5_file}")

    print(f"Loading {os.path.basename(args.hdf5_file)}...")
    ST.strain, ST.info = load_gw_strain(args.hdf5_file, args.start, args.duration)
    print(f"  {ST.info['detector']} | {ST.info['samples']:,} samples "
          f"@ {ST.info['sample_rate']} Hz | {ST.info['segment_duration']}s")

    # initial processing
    ST.audio = process_strain()
    _wave_cache["key"] = None

    # boot audio
    boot_audio()

    # pygame init
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DataSoniPrint \u2014 LIGO Sonifier")
    font = pygame.font.SysFont("DejaVuSans", 22)
    small = pygame.font.SysFont("DejaVuSans", 15)
    tiny = pygame.font.SysFont("DejaVuSans", 12)
    clock = pygame.time.Clock()

    # compute initial visuals
    recompute_visuals()

    running = True
    needs_reprocess = False

    while running:
        play_rect, reproc_rect, stl_rect = draw(screen, (font, small, tiny))

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
                        threading.Thread(target=start_playback,
                                         daemon=True).start()
                elif ev.key == pygame.K_r:
                    ST.audio = process_strain()
                    _wave_cache["key"] = None
                    recompute_visuals()
                    if ST.playing:
                        restart_playback()
                elif ev.key == pygame.K_e:
                    export_stl_threaded()

            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx, my = ev.pos
                if play_rect.collidepoint(mx, my):
                    if ST.playing:
                        stop_playback()
                    else:
                        threading.Thread(target=start_playback,
                                         daemon=True).start()
                    continue
                if reproc_rect.collidepoint(mx, my):
                    ST.audio = process_strain()
                    _wave_cache["key"] = None
                    recompute_visuals()
                    if ST.playing:
                        restart_playback()
                    continue
                if stl_rect.collidepoint(mx, my):
                    export_stl_threaded()
                    continue
                for tag, rect in SLIDERS:
                    if rect.collidepoint(mx, my):
                        ST.dragging = tag
                        break

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if ST.dragging:
                    if ST.dragging in ("spread", "bp_low", "bp_high"):
                        needs_reprocess = True
                    ST.dragging = None

            elif ev.type == pygame.MOUSEMOTION and ST.dragging:
                mx, _ = ev.pos
                val = max(0.0, min(1.0, (mx - SL_X) / float(SL_W)))
                if ST.dragging == "spread":
                    ST.spread = val
                elif ST.dragging == "volume":
                    ST.volume = val
                    if VOL_SIG:
                        VOL_SIG.time = 0.05
                        VOL_SIG.setValue(val)
                elif ST.dragging == "speed":
                    ST.speed = val
                    if SPEED_SIG:
                        SPEED_SIG.setValue(_base_freq() * speed_from_slider(val))
                elif ST.dragging == "reverb":
                    ST.reverb_mix = val
                elif ST.dragging == "bp_low":
                    ST.bp_low_norm = val
                elif ST.dragging == "bp_high":
                    ST.bp_high_norm = val

        # deferred reprocess (on slider release)
        if needs_reprocess:
            needs_reprocess = False
            ST.audio = process_strain()
            _wave_cache["key"] = None
            recompute_visuals()
            if ST.playing:
                restart_playback()

        clock.tick(FPS)

    # shutdown
    stop_playback()
    if server:
        server.stop()
        server.shutdown()
    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
