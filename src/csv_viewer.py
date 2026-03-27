#!/usr/bin/env python3
"""
csv_viewer.py — Quick visual + audio preview of any CSV data file.

Opens a CSV, auto-detects numeric columns, shows a multi-line graph,
lets you pick which column to sonify, and save/load settings presets.

Usage:
    python src/csv_viewer.py data.csv
    python src/csv_viewer.py data.csv --preset my_settings.json
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import threading
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from pyo import (Server, SigTo, DataTable, TableRead, Freeverb,
                     pa_get_default_output)
except ImportError:
    sys.exit("pyo is required:  pip install pyo")

import pygame

# ─── Layout ──────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1400, 850
MARGIN = 30
FPS = 40

GRAPH_Y = 70
GRAPH_H = 380
GRAPH_W = WIDTH - MARGIN * 2 - 220     # leave room for legend sidebar

LEGEND_X = MARGIN + GRAPH_W + 15
LEGEND_W = 190

SL_Y0 = GRAPH_Y + GRAPH_H + 30
SL_W, SL_H, KNOB_R = 800, 20, 9
SL_X = MARGIN
SL_GAP = 44

BG       = (10, 10, 18)
PANEL_BG = (18, 18, 28)
ACCENT   = (80, 180, 255)
ACCENT2  = (255, 140, 60)
ACCENT3  = (100, 255, 180)
DIM      = (50, 50, 65)
TXT      = (220, 220, 230)
TXT_DIM  = (140, 140, 160)

# Distinct colors for up to 12 columns
COL_PALETTE = [
    (80, 180, 255), (255, 140, 60), (100, 255, 180), (255, 100, 100),
    (180, 130, 255), (255, 220, 80), (80, 255, 255), (255, 80, 200),
    (160, 255, 80), (255, 180, 140), (80, 140, 255), (200, 200, 200),
]


# ─── CSV loading ─────────────────────────────────────────────────────────────
def load_csv(path):
    """Load CSV, return (headers, columns_dict, row_count).
    columns_dict maps header_name → numpy array of floats (NaN for non-numeric).
    """
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        raw_headers = next(reader, None)
        if raw_headers is None:
            sys.exit("CSV is empty")
        rows = list(reader)

    if not rows:
        sys.exit("CSV has headers but no data rows")

    # Detect numeric columns
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
        # Only include columns that are mostly numeric (>50%)
        if num_count > len(rows) * 0.5:
            headers.append(h)
            columns[h] = np.array(vals)

    if not headers:
        sys.exit("No numeric columns found in CSV")

    return headers, columns, len(rows)


# ─── State ───────────────────────────────────────────────────────────────────
class State:
    filepath = ""
    headers = []
    columns = {}
    row_count = 0

    # Which columns are visible (toggled in legend)
    visible = {}        # header → bool

    # Sonification column (index into headers)
    sonify_col = 0

    # Sliders (0–1)
    spread = 0.35
    volume = 0.7
    speed = 0.5
    reverb_mix = 0.2

    # Audio
    audio = None
    playing = False
    play_start = 0.0
    playback_sr = 44100
    reader = None
    table = None

    dragging = None

    # Graph scroll/zoom
    x_offset = 0.0     # 0–1: scroll position
    x_zoom = 1.0        # 1.0 = show all, >1 = zoomed in


ST = State()


# ─── Settings save/load ─────────────────────────────────────────────────────
def save_settings(path):
    """Save current slider values and column visibility to JSON."""
    data = {
        "spread": ST.spread,
        "volume": ST.volume,
        "speed": ST.speed,
        "reverb_mix": ST.reverb_mix,
        "sonify_col": ST.sonify_col,
        "visible": ST.visible,
        "x_zoom": ST.x_zoom,
        "x_offset": ST.x_offset,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Settings saved: {path}")
    return path


def load_settings(path):
    """Load settings from JSON and apply to state."""
    if not os.path.isfile(path):
        print(f"  Settings file not found: {path}")
        return False
    with open(path, "r") as f:
        data = json.load(f)
    ST.spread = data.get("spread", ST.spread)
    ST.volume = data.get("volume", ST.volume)
    ST.speed = data.get("speed", ST.speed)
    ST.reverb_mix = data.get("reverb_mix", ST.reverb_mix)
    ST.sonify_col = data.get("sonify_col", ST.sonify_col)
    saved_vis = data.get("visible", {})
    for h in ST.headers:
        if h in saved_vis:
            ST.visible[h] = saved_vis[h]
    ST.x_zoom = data.get("x_zoom", ST.x_zoom)
    ST.x_offset = data.get("x_offset", ST.x_offset)
    print(f"  Settings loaded: {path}")
    return True


def default_settings_path():
    """Default settings file next to the CSV."""
    base = os.path.splitext(ST.filepath)[0]
    return base + "_settings.json"


# ─── Audio helpers ───────────────────────────────────────────────────────────
def speed_from_slider(v):
    return 0.25 * (16 ** v)


def process_column():
    """Convert the selected column to FM-synthesized audio."""
    if not ST.headers:
        return None

    col_name = ST.headers[ST.sonify_col % len(ST.headers)]
    data = ST.columns[col_name].copy()

    # Replace NaN with 0
    data = np.nan_to_num(data, nan=0.0)

    # Normalize to [-1, +1]
    peak = np.max(np.abs(data))
    if peak > 0:
        normed = data / peak
    else:
        normed = data

    # Resample to playback rate (data points → audio samples)
    # Each data point gets enough samples for ~0.01s at playback sr
    samples_per_point = max(1, int(ST.playback_sr * 0.01))
    out_len = len(normed) * samples_per_point
    orig_t = np.linspace(0, 1, len(normed))
    new_t = np.linspace(0, 1, out_len)
    normed_up = np.interp(new_t, orig_t, normed)

    # FM synthesis: value → pitch
    semi_range = 2 + ST.spread * 46
    center_hz = 440.0
    freq_array = center_hz * np.power(2.0, normed_up * semi_range / 12.0)
    phase_inc = freq_array / ST.playback_sr
    phase = np.cumsum(phase_inc)
    audio = np.sin(2.0 * np.pi * phase) * 0.75

    return audio


# ─── pyo audio ───────────────────────────────────────────────────────────────
server = None
VOL_SIG = None
SPEED_SIG = None
REVERB_MIX_SIG = None
_audio_chain = []


def _base_freq():
    if ST.audio is not None and len(ST.audio) > 0:
        return 1.0 / (len(ST.audio) / ST.playback_sr)
    return 1.0


def boot_audio():
    global server, VOL_SIG, SPEED_SIG, REVERB_MIX_SIG
    dev = pa_get_default_output()
    server = Server(sr=ST.playback_sr, duplex=0, buffersize=4096, audio="portaudio")
    server.setOutputDevice(dev)
    server.boot()
    server.start()
    VOL_SIG = SigTo(value=ST.volume, time=0.05, init=ST.volume)
    SPEED_SIG = SigTo(value=1.0, time=0.12, init=1.0)
    REVERB_MIX_SIG = SigTo(value=ST.reverb_mix * 0.6, time=0.1,
                            init=ST.reverb_mix * 0.6)


def start_playback():
    stop_playback()
    ST.audio = process_column()
    if ST.audio is None or len(ST.audio) == 0:
        return

    spd = speed_from_slider(ST.speed)
    ST.table = DataTable(size=len(ST.audio), init=ST.audio.tolist())
    bf = _base_freq()
    SPEED_SIG.setValue(bf * spd)

    reader = TableRead(ST.table, freq=SPEED_SIG, loop=True, mul=VOL_SIG)
    reverb = Freeverb(reader, size=0.88, damp=0.5, bal=1.0, mul=REVERB_MIX_SIG)
    reader.out()
    reverb.out()

    ST.reader = reader
    _audio_chain.clear()
    _audio_chain.extend([reader, reverb])

    ST.play_start = time.time()
    ST.playing = True


def stop_playback():
    for obj in _audio_chain:
        try:
            obj.stop()
        except Exception:
            pass
    _audio_chain.clear()
    ST.reader = None
    ST.table = None
    ST.playing = False


# ─── Graph rendering ────────────────────────────────────────────────────────
def draw_graph(screen, fonts):
    _, small, tiny = fonts
    graph_rect = pygame.Rect(MARGIN, GRAPH_Y, GRAPH_W, GRAPH_H)
    pygame.draw.rect(screen, PANEL_BG, graph_rect, border_radius=6)
    pygame.draw.rect(screen, DIM, graph_rect, width=1, border_radius=6)

    # Title
    screen.blit(tiny.render("DATA OVERVIEW", True, TXT_DIM),
                (MARGIN + 8, GRAPH_Y + 4))

    if not ST.headers:
        return

    inner = pygame.Rect(MARGIN + 5, GRAPH_Y + 20, GRAPH_W - 10, GRAPH_H - 30)

    # Determine visible data range (zoom/scroll)
    n = ST.row_count
    visible_n = max(10, int(n / ST.x_zoom))
    start_idx = int(ST.x_offset * max(0, n - visible_n))
    end_idx = min(n, start_idx + visible_n)

    # Draw grid lines
    for i in range(5):
        gy = inner.y + int(i / 4 * inner.height)
        pygame.draw.line(screen, (25, 25, 38), (inner.x, gy), (inner.right, gy))

    # X-axis labels
    for i in range(6):
        xi = start_idx + int(i / 5 * (end_idx - start_idx))
        px = inner.x + int(i / 5 * inner.width)
        screen.blit(tiny.render(str(xi), True, TXT_DIM), (px, inner.bottom + 2))

    # Draw each visible column
    for ci, h in enumerate(ST.headers):
        if not ST.visible.get(h, True):
            continue

        col = ST.columns[h]
        segment = col[start_idx:end_idx]
        valid = segment[np.isfinite(segment)]
        if len(valid) == 0:
            continue

        vmin, vmax = np.min(valid), np.max(valid)
        if vmin == vmax:
            vmax = vmin + 1

        color = COL_PALETTE[ci % len(COL_PALETTE)]
        pts = []
        for i, v in enumerate(segment):
            if not np.isfinite(v):
                continue
            x = inner.x + int(i / max(1, len(segment) - 1) * inner.width)
            t = (v - vmin) / (vmax - vmin)
            y = inner.bottom - int(t * inner.height)
            pts.append((x, y))

        if len(pts) > 1:
            pygame.draw.lines(screen, color, False, pts, 2)

    # Playback cursor
    if ST.playing and ST.audio is not None:
        elapsed = time.time() - ST.play_start
        spd = speed_from_slider(ST.speed)
        loop_dur = len(ST.audio) / ST.playback_sr / spd
        pct = (elapsed % loop_dur) / loop_dur if loop_dur > 0 else 0
        # Map pct to visible range
        data_idx = pct * ST.row_count
        if start_idx <= data_idx <= end_idx:
            cx = inner.x + int((data_idx - start_idx) /
                               max(1, end_idx - start_idx) * inner.width)
            pygame.draw.line(screen, ACCENT2,
                             (cx, graph_rect.y), (cx, graph_rect.bottom), 2)


# ─── Legend sidebar ──────────────────────────────────────────────────────────
_legend_rects = []


def draw_legend(screen, fonts):
    _, small, tiny = fonts
    _legend_rects.clear()

    leg_rect = pygame.Rect(LEGEND_X, GRAPH_Y, LEGEND_W, GRAPH_H)
    pygame.draw.rect(screen, PANEL_BG, leg_rect, border_radius=6)
    pygame.draw.rect(screen, DIM, leg_rect, width=1, border_radius=6)
    screen.blit(tiny.render("COLUMNS", True, TXT_DIM),
                (LEGEND_X + 8, GRAPH_Y + 4))
    screen.blit(tiny.render("click=toggle  right=sonify", True, (70, 70, 90)),
                (LEGEND_X + 8, GRAPH_Y + 18))

    y = GRAPH_Y + 36
    for ci, h in enumerate(ST.headers):
        color = COL_PALETTE[ci % len(COL_PALETTE)]
        vis = ST.visible.get(h, True)

        row_rect = pygame.Rect(LEGEND_X + 4, y, LEGEND_W - 8, 22)
        _legend_rects.append((ci, h, row_rect))

        # Highlight sonify column
        if ci == ST.sonify_col:
            pygame.draw.rect(screen, (30, 40, 55), row_rect, border_radius=3)
            pygame.draw.rect(screen, ACCENT2, row_rect, width=1, border_radius=3)

        # Color swatch
        swatch_col = color if vis else (40, 40, 50)
        pygame.draw.rect(screen, swatch_col,
                         (LEGEND_X + 10, y + 5, 12, 12), border_radius=2)

        # Column name
        txt_col = TXT if vis else (60, 60, 70)
        label = h[:18]
        screen.blit(tiny.render(label, True, txt_col), (LEGEND_X + 28, y + 4))

        # Stats
        col_data = ST.columns[h]
        valid = col_data[np.isfinite(col_data)]
        if len(valid) > 0:
            stats_txt = f"{np.min(valid):.1f}–{np.max(valid):.1f}"
            screen.blit(tiny.render(stats_txt, True, (60, 60, 80)),
                        (LEGEND_X + 28, y + 15))
            y += 34
        else:
            y += 24

        if y > GRAPH_Y + GRAPH_H - 10:
            break


# ─── Sliders ─────────────────────────────────────────────────────────────────
SLIDERS = []


def draw_slider(screen, font, label, y, val, color, tag):
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


# ─── Main draw ───────────────────────────────────────────────────────────────
def draw(screen, fonts):
    font, small, tiny = fonts
    screen.fill(BG)

    # Title
    fname = os.path.basename(ST.filepath)
    screen.blit(font.render(
        f"DataSoniPrint — CSV Viewer:  {fname}", True, ACCENT),
        (MARGIN, 12))
    info = f"{ST.row_count} rows  |  {len(ST.headers)} numeric columns"
    son_col = ST.headers[ST.sonify_col] if ST.headers else "—"
    info += f"  |  Sonify: {son_col}"
    screen.blit(small.render(info, True, TXT_DIM), (MARGIN, 42))

    # Graph & legend
    draw_graph(screen, fonts)
    draw_legend(screen, fonts)

    # Sliders
    sy = SL_Y0
    SLIDERS.clear()
    semi = 2 + ST.spread * 46
    SLIDERS.append(draw_slider(screen, small,
        f"DATA SPREAD  ({semi:.0f} semitones)", sy, ST.spread, ACCENT, "spread"))
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

    # Zoom slider
    sy += SL_GAP
    SLIDERS.append(draw_slider(screen, small,
        f"ZOOM  ({ST.x_zoom:.1f}x)", sy, min(1.0, (ST.x_zoom - 1) / 19),
        (180, 180, 120), "zoom"))

    # Scroll slider (only when zoomed)
    if ST.x_zoom > 1.05:
        sy += SL_GAP
        SLIDERS.append(draw_slider(screen, small,
            f"SCROLL  ({ST.x_offset:.0%})", sy, ST.x_offset,
            (120, 160, 180), "scroll"))

    # Buttons
    btn_y = sy + SL_GAP + 10

    play_rect = pygame.Rect(MARGIN, btn_y, 160, 40)
    play_col = ACCENT2 if ST.playing else (40, 110, 170)
    pygame.draw.rect(screen, play_col, play_rect, border_radius=6)
    pygame.draw.rect(screen, ACCENT if not ST.playing else (255, 180, 100),
                     play_rect, width=2, border_radius=6)
    play_lbl = "\u25a0  Stop" if ST.playing else "\u25b6  Play"
    screen.blit(font.render(play_lbl, True, TXT),
                (play_rect.x + 30, play_rect.y + 8))

    save_rect = pygame.Rect(MARGIN + 180, btn_y, 220, 40)
    pygame.draw.rect(screen, (40, 55, 70), save_rect, border_radius=6)
    pygame.draw.rect(screen, (80, 160, 220), save_rect, width=2, border_radius=6)
    screen.blit(font.render("\U0001F4BE  Save Settings", True, TXT),
                (save_rect.x + 20, save_rect.y + 8))

    load_rect = pygame.Rect(MARGIN + 420, btn_y, 220, 40)
    pygame.draw.rect(screen, (50, 50, 40), load_rect, border_radius=6)
    pygame.draw.rect(screen, (200, 180, 80), load_rect, width=2, border_radius=6)
    screen.blit(font.render("\U0001F4C2  Load Settings", True, TXT),
                (load_rect.x + 20, load_rect.y + 8))

    # Status
    if ST.playing and ST.audio is not None:
        elapsed = time.time() - ST.play_start
        spd_val = speed_from_slider(ST.speed)
        loop_dur = len(ST.audio) / ST.playback_sr / spd_val
        pct = (elapsed % loop_dur) / loop_dur * 100 if loop_dur > 0 else 0
        stat = f"Playing  {elapsed:.1f}s  ({pct:.0f}%)"
        screen.blit(small.render(stat, True, ACCENT2), (MARGIN + 660, btn_y + 12))

    # Key hints
    hints = "SPACE: play/stop    S: save settings    L: load settings    Q/ESC: quit"
    screen.blit(tiny.render(hints, True, (80, 80, 100)), (MARGIN, HEIGHT - 25))

    pygame.display.flip()
    return play_rect, save_rect, load_rect


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DataSoniPrint CSV Viewer")
    parser.add_argument("csv_file", help="Path to CSV data file")
    parser.add_argument("--preset", type=str, default=None,
                        help="Load settings from JSON preset file")
    args = parser.parse_args()

    if not os.path.isfile(args.csv_file):
        sys.exit(f"File not found: {args.csv_file}")

    ST.filepath = args.csv_file
    print(f"Loading {os.path.basename(args.csv_file)}...")
    ST.headers, ST.columns, ST.row_count = load_csv(args.csv_file)
    print(f"  {ST.row_count} rows, columns: {ST.headers}")

    # Default: all visible
    ST.visible = {h: True for h in ST.headers}

    # Print quick stats
    for h in ST.headers:
        col = ST.columns[h]
        valid = col[np.isfinite(col)]
        if len(valid) > 0:
            print(f"    {h}: {np.min(valid):.4g} – {np.max(valid):.4g}  "
                  f"(mean {np.mean(valid):.4g}, n={len(valid)})")

    # Load preset if given
    if args.preset:
        load_settings(args.preset)

    # Process initial audio
    ST.audio = process_column()

    # Boot audio
    boot_audio()

    # Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"DataSoniPrint — {os.path.basename(args.csv_file)}")
    font = pygame.font.SysFont("DejaVuSans", 22)
    small = pygame.font.SysFont("DejaVuSans", 15)
    tiny = pygame.font.SysFont("DejaVuSans", 12)
    clock = pygame.time.Clock()

    running = True
    needs_reprocess = False

    while running:
        play_rect, save_rect, load_rect = draw(screen, (font, small, tiny))

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
                        start_playback()
                elif ev.key == pygame.K_s:
                    save_settings(default_settings_path())
                elif ev.key == pygame.K_l:
                    if load_settings(default_settings_path()):
                        needs_reprocess = True

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos

                if ev.button == 1:
                    # Buttons
                    if play_rect.collidepoint(mx, my):
                        if ST.playing:
                            stop_playback()
                        else:
                            start_playback()
                        continue
                    if save_rect.collidepoint(mx, my):
                        save_settings(default_settings_path())
                        continue
                    if load_rect.collidepoint(mx, my):
                        if load_settings(default_settings_path()):
                            needs_reprocess = True
                        continue

                    # Legend clicks — toggle visibility
                    for ci, h, rect in _legend_rects:
                        if rect.collidepoint(mx, my):
                            ST.visible[h] = not ST.visible.get(h, True)
                            break

                    # Slider hit test
                    for tag, rect in SLIDERS:
                        if rect.collidepoint(mx, my):
                            ST.dragging = tag
                            break

                elif ev.button == 3:
                    # Right-click legend → set as sonify column
                    for ci, h, rect in _legend_rects:
                        if rect.collidepoint(mx, my):
                            ST.sonify_col = ci
                            needs_reprocess = True
                            break

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                if ST.dragging:
                    if ST.dragging == "spread":
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
                        VOL_SIG.setValue(val)
                elif ST.dragging == "speed":
                    ST.speed = val
                    if SPEED_SIG and ST.audio is not None:
                        bf = _base_freq()
                        SPEED_SIG.setValue(bf * speed_from_slider(val))
                elif ST.dragging == "reverb":
                    ST.reverb_mix = val
                    if REVERB_MIX_SIG:
                        REVERB_MIX_SIG.setValue(val * 0.6)
                elif ST.dragging == "zoom":
                    ST.x_zoom = 1.0 + val * 19.0   # 1x–20x
                elif ST.dragging == "scroll":
                    ST.x_offset = val

            elif ev.type == pygame.MOUSEWHEEL:
                # Scroll wheel → zoom
                ST.x_zoom = max(1.0, min(20.0, ST.x_zoom + ev.y * 0.5))

        # Deferred reprocess
        if needs_reprocess:
            needs_reprocess = False
            ST.audio = process_column()
            if ST.playing:
                start_playback()

        clock.tick(FPS)

    # Shutdown
    stop_playback()
    if server:
        server.stop()
        server.shutdown()
    pygame.quit()
    print("Done.")


if __name__ == "__main__":
    main()
