#!/usr/bin/env python3
"""
gw_explorer.py — Multi-dimensional LIGO data explorer & sonifier.

Single-window app: large data visualization (waveform, spectrogram, 3D scatter)
on the left with controls (dropdowns, sliders, buttons) on the right.
Changes in the dropdowns update the 3D scatter immediately.

Usage:
    python src/gw_explorer.py
    python src/gw_explorer.py ~/Downloads/H-H1_GWOSC_*.hdf5
    python src/gw_explorer.py ~/Downloads/H-H1_GWOSC_*.hdf5 --duration 120
"""

import argparse
import os
import sys
import time
import math
import threading
import wave as wavmod
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sonify_gw import load_gw_strain, bandpass, normalize

try:
    from pyo import (Server, SigTo, Sine, Freeverb, Biquad, Pan,
                     DataTable, Phasor, Pointer, pa_get_default_output,
                     pa_list_devices)
except ImportError:
    sys.exit("pyo is required:  pip install pyo")

try:
    from scipy.signal import fftconvolve
except ImportError:
    fftconvolve = None

import pygame

# ─── Constants ───────────────────────────────────────────────────────────────
DEFAULT_HDF5 = os.path.expanduser(
    "~/Downloads/H-H1_GWOSC_O4a_4KHZ_R1-1368195072-4096.hdf5")
PLAYBACK_SR = 48000
EXPORT_DIR = os.path.expanduser("~/DataSoniPrint/exports")

# Single window: data viz on left, controls on right
WIN_W, WIN_H = 1920, 1200
DATA_W = 1340          # left panel width for data visualization
CTRL_W = WIN_W - DATA_W  # right panel width for controls (580)
FPS = 30

# Colors
BG       = (10, 10, 18)
PANEL_BG = (18, 18, 28)
CTRL_BG  = (14, 14, 22)
ACCENT   = (80, 180, 255)
ACCENT2  = (255, 140, 60)
ACCENT3  = (100, 255, 180)
DIM      = (50, 50, 65)
TXT      = (220, 220, 230)
TXT_DIM  = (140, 140, 160)
BTN_BG   = (35, 45, 60)
BTN_HI   = (50, 65, 85)
BTN_REC  = (100, 30, 30)
BTN_REC_HI = (140, 45, 45)
DD_BG    = (25, 25, 40)
DD_HI    = (40, 40, 60)
DD_OPEN  = (30, 30, 50)
SLD_BG   = (25, 25, 40)
SLD_FILL = (60, 140, 220)
SLD_KNOB = (100, 200, 255)

# ─── Data dimension extractors ───────────────────────────────────────────────
DATA_DIMS = [
    "time",
    "strain",
    "inst_frequency",
    "rms_energy",
    "spectral_centroid",
    "delta_strain",
]

SONIC_PARAMS = [
    "(none)",
    "pitch",
    "volume",
    "cutoff",
    "release",
    "panning",
    "note_length",
]


def extract_dimension(name, strain, sr):
    """Extract a named dimension from raw strain, return array normalized to [0,1]."""
    n = len(strain)

    if name == "time":
        return np.linspace(0, 1, n)

    if name == "strain":
        peak = np.max(np.abs(strain))
        if peak > 0:
            return (strain / peak + 1) / 2
        return np.full(n, 0.5)

    if name == "inst_frequency":
        from scipy.signal import hilbert
        analytic = hilbert(strain)
        inst_phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(inst_phase) / (2 * np.pi / sr)
        inst_freq = np.append(inst_freq, inst_freq[-1])
        inst_freq = np.clip(inst_freq, 0, sr / 2)
        fmax = np.percentile(np.abs(inst_freq), 99)
        if fmax > 0:
            return np.clip(inst_freq / fmax, 0, 1)
        return np.full(n, 0.5)

    if name == "rms_energy":
        win = min(512, n // 4)
        if win < 2:
            return np.full(n, 0.5)
        kernel = np.ones(win) / win
        rms = np.sqrt(np.convolve(strain ** 2, kernel, mode='same'))
        rmax = np.max(rms)
        if rmax > 0:
            return rms / rmax
        return np.full(n, 0.5)

    if name == "spectral_centroid":
        win = 1024
        hop = win // 2
        n_frames = max(1, (n - win) // hop)
        centroids = np.zeros(n_frames)
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        for i in range(n_frames):
            chunk = strain[i * hop:i * hop + win]
            mag = np.abs(np.fft.rfft(chunk))
            total = np.sum(mag)
            if total > 0:
                centroids[i] = np.sum(freqs * mag) / total
        t_frames = np.linspace(0, 1, n_frames)
        t_full = np.linspace(0, 1, n)
        full = np.interp(t_full, t_frames, centroids)
        fmax = np.max(full)
        if fmax > 0:
            return full / fmax
        return np.full(n, 0.5)

    if name == "delta_strain":
        delta = np.abs(np.diff(strain))
        delta = np.append(delta, delta[-1])
        dmax = np.percentile(delta, 99)
        if dmax > 0:
            return np.clip(delta / dmax, 0, 1)
        return np.full(n, 0.5)

    return np.full(n, 0.5)


# ─── Sonification engine ────────────────────────────────────────────────────
def render_audio(strain, sr, mapping, spread=0.61, reverb_mix=0.3,
                 speed_slider=0.39, bp_low=10, bp_high=4000):
    """Render audio offline using the axis→parameter mapping."""
    filtered = bandpass(strain, bp_low, bp_high, sr)
    normed = normalize(filtered, target_peak=1.0)

    out_sr = PLAYBACK_SR
    if sr != out_sr:
        orig_t = np.linspace(0, 1, len(normed))
        new_len = int(len(normed) * out_sr / sr)
        new_t = np.linspace(0, 1, new_len)
        normed = np.interp(new_t, orig_t, normed)
        strain_rs = np.interp(new_t, orig_t, strain[:len(orig_t)])
    else:
        strain_rs = strain
        new_len = len(strain)
    n = len(normed)

    dims = {}
    for param, dim_name in mapping.items():
        if dim_name and dim_name != "(none)":
            dims[param] = extract_dimension(dim_name, strain_rs, out_sr)

    semi_range = 2 + spread * 46
    center_hz = 440.0

    if "pitch" in dims:
        pitch_mod = dims["pitch"] * 2 - 1
    else:
        pitch_mod = normed

    freq_array = center_hz * np.power(2.0, pitch_mod * semi_range / 12.0)
    phase = np.cumsum(freq_array / out_sr)
    audio = np.sin(2.0 * np.pi * phase) * 0.75

    if "volume" in dims:
        audio *= 0.2 + dims["volume"] * 0.8

    if "cutoff" in dims:
        cutoff_norm = dims["cutoff"]
        block = 256
        n_blocks = n // block
        result = np.copy(audio)
        for b in range(n_blocks):
            i0, i1 = b * block, (b + 1) * block
            co = np.mean(cutoff_norm[i0:i1])
            alpha = 0.02 + co * 0.98
            for i in range(i0 + 1, i1):
                result[i] = alpha * audio[i] + (1 - alpha) * result[i - 1]
        audio = result

    if "release" in dims:
        release_norm = dims["release"]
        env = np.ones(n)
        block = 1024
        for b in range(n // block):
            i0, i1 = b * block, (b + 1) * block
            rel = np.mean(release_norm[i0:i1])
            decay = 0.9990 + rel * 0.0009
            env[i0:i1] = np.power(decay, np.arange(block))
        audio *= env

    if "panning" in dims:
        pan_pos = dims["panning"]
        left = audio * np.cos(pan_pos * np.pi / 2)
        right = audio * np.sin(pan_pos * np.pi / 2)
    else:
        left = audio
        right = audio

    if "note_length" in dims:
        nl = dims["note_length"]
        grain_env = np.ones(n)
        pos = 0
        while pos < n:
            local_nl = nl[min(pos, n - 1)]
            grain_size = int(50 + local_nl * 1950)
            grain_size = min(grain_size, n - pos)
            grain_env[pos:pos + grain_size] *= np.hanning(grain_size)
            pos += grain_size
        left *= grain_env
        right *= grain_env

    speed = 0.25 * (16 ** speed_slider)
    if abs(speed - 1.0) > 0.01:
        out_len = int(n / speed)
        out_t = np.linspace(0, n - 1, out_len)
        src_idx = np.arange(n)
        left = np.interp(out_t, src_idx, left)
        right = np.interp(out_t, src_idx, right)

    if reverb_mix > 0 and fftconvolve is not None:
        reverb_len = int(out_sr * 1.5)
        impulse = np.exp(-np.linspace(0, 6, reverb_len))
        impulse[0] = 0
        for d in [0.02, 0.04, 0.07, 0.12, 0.18]:
            idx = int(d * out_sr)
            if idx < reverb_len:
                impulse[idx] += 0.5
        impulse /= np.max(np.abs(impulse))
        wet_l = fftconvolve(left, impulse, mode='full')[:len(left)]
        wet_r = fftconvolve(right, impulse, mode='full')[:len(right)]
        wmax_l = np.max(np.abs(wet_l))
        wmax_r = np.max(np.abs(wet_r))
        if wmax_l > 0:
            wet_l = wet_l / wmax_l * 0.75
        if wmax_r > 0:
            wet_r = wet_r / wmax_r * 0.75
        left = left * (1 - reverb_mix * 0.5) + wet_l * reverb_mix * 0.6
        right = right * (1 - reverb_mix * 0.5) + wet_r * reverb_mix * 0.6

    return np.column_stack([np.clip(left, -1, 1), np.clip(right, -1, 1)]), out_sr


def save_stereo_wav(path, stereo_audio, sr):
    pcm = (np.clip(stereo_audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wavmod.open(path, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.flatten().tobytes())


def compute_stats(strain, info):
    return [
        ("Detector", info["detector"]),
        ("UTC start", info["utc_start"]),
        ("Duration", f"{info['segment_duration']}s"),
        ("Sample rate", f"{info['sample_rate']} Hz"),
        ("Samples", f"{info['samples']:,}"),
        ("Strain min", f"{strain.min():.3e}"),
        ("Strain max", f"{strain.max():.3e}"),
        ("Strain std", f"{strain.std():.3e}"),
        ("Strain RMS", f"{np.sqrt(np.mean(strain**2)):.3e}"),
    ]


# ─── Inferno LUT ─────────────────────────────────────────────────────────────
def _build_inferno():
    pts = [
        (0.00, 0, 0, 4), (0.13, 40, 11, 84), (0.25, 87, 16, 110),
        (0.38, 137, 30, 93), (0.50, 188, 55, 84), (0.63, 227, 89, 51),
        (0.75, 249, 142, 9), (0.88, 252, 201, 38), (1.00, 252, 255, 164)]
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

INFERNO = _build_inferno()


# ─── Pygame UI widgets ──────────────────────────────────────────────────────
class PgDropdown:
    """Custom dropdown widget drawn in pygame."""

    def __init__(self, x, y, w, h, options, selected=0, label="", font=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected = selected
        self.label = label
        self.font = font
        self.open = False
        self.hover_idx = -1
        self._item_h = h

    @property
    def value(self):
        return self.options[self.selected]

    @value.setter
    def value(self, v):
        if v in self.options:
            self.selected = self.options.index(v)

    def draw(self, screen):
        # Label
        if self.label and self.font:
            lbl = self.font.render(self.label, True, TXT_DIM)
            screen.blit(lbl, (self.rect.x, self.rect.y - 15))

        # Main box
        bg = DD_HI if self.open else DD_BG
        pygame.draw.rect(screen, bg, self.rect, border_radius=3)
        pygame.draw.rect(screen, DIM, self.rect, width=1, border_radius=3)

        # Selected text
        if self.font:
            txt = self.font.render(self.value, True, ACCENT)
            screen.blit(txt, (self.rect.x + 6, self.rect.y + 4))
        # Arrow
        ax = self.rect.right - 14
        ay = self.rect.centery
        pygame.draw.polygon(screen, TXT_DIM,
                            [(ax - 4, ay - 3), (ax + 4, ay - 3), (ax, ay + 3)])

    def draw_dropdown(self, screen):
        """Draw the open dropdown list (call AFTER all other draws)."""
        if not self.open:
            return
        n = len(self.options)
        list_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                self.rect.width, n * self._item_h)
        pygame.draw.rect(screen, DD_OPEN, list_rect, border_radius=3)
        pygame.draw.rect(screen, DIM, list_rect, width=1, border_radius=3)
        for i, opt in enumerate(self.options):
            item_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self._item_h,
                                    self.rect.width, self._item_h)
            if i == self.hover_idx:
                pygame.draw.rect(screen, DD_HI, item_rect)
            col = ACCENT if i == self.selected else TXT
            if self.font:
                txt = self.font.render(opt, True, col)
                screen.blit(txt, (item_rect.x + 6, item_rect.y + 4))

    def handle_event(self, ev):
        """Return True if selection changed."""
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.open:
                n = len(self.options)
                list_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                        self.rect.width, n * self._item_h)
                if list_rect.collidepoint(ev.pos):
                    idx = (ev.pos[1] - self.rect.bottom) // self._item_h
                    if 0 <= idx < n:
                        old = self.selected
                        self.selected = idx
                        self.open = False
                        return old != idx
                self.open = False
            elif self.rect.collidepoint(ev.pos):
                self.open = True
        elif ev.type == pygame.MOUSEMOTION and self.open:
            n = len(self.options)
            list_rect = pygame.Rect(self.rect.x, self.rect.bottom,
                                    self.rect.width, n * self._item_h)
            if list_rect.collidepoint(ev.pos):
                self.hover_idx = (ev.pos[1] - self.rect.bottom) // self._item_h
            else:
                self.hover_idx = -1
        return False


class PgSlider:
    """Horizontal slider widget."""

    def __init__(self, x, y, w, h, lo=0.0, hi=1.0, val=0.5, label="", font=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.lo, self.hi = lo, hi
        self.val = val
        self.label = label
        self.font = font
        self.dragging = False

    def draw(self, screen):
        if self.label and self.font:
            lbl = self.font.render(f"{self.label}: {self.val:.2f}", True, TXT_DIM)
            screen.blit(lbl, (self.rect.x, self.rect.y - 15))
        # Track
        track = pygame.Rect(self.rect.x, self.rect.centery - 3,
                            self.rect.width, 6)
        pygame.draw.rect(screen, SLD_BG, track, border_radius=3)
        # Fill
        frac = (self.val - self.lo) / (self.hi - self.lo) if self.hi > self.lo else 0
        fill = pygame.Rect(track.x, track.y, int(track.width * frac), 6)
        pygame.draw.rect(screen, SLD_FILL, fill, border_radius=3)
        # Knob
        kx = track.x + int(track.width * frac)
        pygame.draw.circle(screen, SLD_KNOB, (kx, track.centery), 8)

    def handle_event(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            knob_rect = pygame.Rect(self.rect.x - 8, self.rect.y - 4,
                                    self.rect.width + 16, self.rect.height + 8)
            if knob_rect.collidepoint(ev.pos):
                self.dragging = True
                self._update_val(ev.pos[0])
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
            self.dragging = False
        elif ev.type == pygame.MOUSEMOTION and self.dragging:
            self._update_val(ev.pos[0])

    def _update_val(self, mx):
        frac = (mx - self.rect.x) / self.rect.width
        frac = max(0, min(1, frac))
        self.val = self.lo + frac * (self.hi - self.lo)


class PgButton:
    """Clickable button widget."""

    def __init__(self, x, y, w, h, text="", font=None, color=BTN_BG, hover=BTN_HI):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.color = color
        self.hover = hover
        self._hovered = False

    def draw(self, screen):
        bg = self.hover if self._hovered else self.color
        pygame.draw.rect(screen, bg, self.rect, border_radius=4)
        pygame.draw.rect(screen, DIM, self.rect, width=1, border_radius=4)
        if self.font:
            txt = self.font.render(self.text, True, TXT)
            screen.blit(txt, (self.rect.x + (self.rect.width - txt.get_width()) // 2,
                              self.rect.y + (self.rect.height - txt.get_height()) // 2))

    def handle_event(self, ev):
        """Return True if clicked."""
        if ev.type == pygame.MOUSEMOTION:
            self._hovered = self.rect.collidepoint(ev.pos)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                return True
        return False


# ─── Main App ────────────────────────────────────────────────────────────────
class GWExplorer:
    """Single-window app with data viz on the left and controls on the right."""

    def __init__(self, strain, info):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("GW Explorer")
        self.font = pygame.font.SysFont("DejaVuSans", 32)
        self.small = pygame.font.SysFont("DejaVuSans", 24)
        self.tiny = pygame.font.SysFont("DejaVuSans", 20)
        self.clock = pygame.time.Clock()

        self.strain = strain
        self.info = info
        self.dims_cache = {}
        self.running = True
        self.status = "Ready"

        # 3D scatter state
        self.rot_x = -30
        self.rot_y = 45
        self.dragging_3d = False
        self.drag_start = (0, 0)
        self.scatter_zoom = 1.0   # scroll-zoom for 3D scatter

        # Zoom state: fraction of total samples visible (1.0 = all)
        self.zoom = 1.0          # 1.0 = full view, smaller = zoomed in
        self.zoom_center = 0.5   # center of zoom window (0–1)
        self.min_zoom = 0.005    # max zoom: ~0.5% of data

        # Playback
        self.playing = False
        self.play_start = 0.0
        self.play_duration = 0.0
        self._play_phase = 0.0
        self._last_tick_time = 0.0
        self._stop_pending = None
        self._last_spread = 0.0
        # Real-time pyo objects (set during playback)
        self._speed_sig = None
        self._reverb_sig = None
        self._master_vol = None
        self._freq_table = None
        self._vol_table = None
        self._data_dur = 1.0
        self._last_speed_val = -1.0
        self._last_reverb_val = -1.0

        # Build spectrogram
        self.spec_surf = None
        self._build_spectrogram()

        # Build UI widgets
        self._build_widgets()

    def _build_spectrogram(self):
        from visualize_gw import compute_spectrogram
        sr = self.info["sample_rate"]
        filtered = bandpass(self.strain, 10, 4000, sr)
        normed = normalize(filtered, target_peak=1.0)
        times, freqs, spec_db = compute_spectrogram(normed, sr, nperseg=512, overlap_frac=0.75)
        freq_max = min(2000, sr / 2)
        freq_mask = freqs <= freq_max
        spec_clip = spec_db[freq_mask, :]
        vmin = np.percentile(spec_clip, 5)
        vmax = np.percentile(spec_clip, 99)
        norm = np.clip((spec_clip - vmin) / (vmax - vmin + 1e-10), 0, 1)
        indices = (norm * 255).astype(np.uint8)
        rgb = INFERNO[indices][::-1, :, :]
        rgb_t = np.ascontiguousarray(rgb.transpose(1, 0, 2))
        raw_surf = pygame.surfarray.make_surface(rgb_t)
        self.spec_surf = pygame.transform.smoothscale(raw_surf, (DATA_W - 40, 400))

    def _get_dim(self, name):
        if name not in self.dims_cache:
            self.dims_cache[name] = extract_dimension(
                name, self.strain, self.info["sample_rate"])
        return self.dims_cache[name]

    def _build_widgets(self):
        cx = DATA_W + 16  # control panel x offset
        cw = CTRL_W - 32  # usable width
        dd_w = 170
        dd_h = 22
        f = self.small

        y = 40
        # ── Sonification axis mapping ────────────────────────────────────
        self.mapping_dds = {}
        defaults = {"pitch": "strain", "volume": "(none)", "cutoff": "(none)",
                    "release": "(none)", "panning": "(none)", "note_length": "(none)"}
        opts = ["(none)"] + DATA_DIMS
        for param in SONIC_PARAMS[1:]:
            sel = opts.index(defaults.get(param, "(none)"))
            dd = PgDropdown(cx + 120, y, dd_w, dd_h, opts, sel, label="", font=f)
            self.mapping_dds[param] = dd
            y += 36

        y += 14
        # ── 3D scatter axis mapping ──────────────────────────────────────
        self.scatter_dds = {}
        scatter_defaults = {"X": "time", "Y": "strain", "Z": "rms_energy", "Color": "inst_frequency"}
        for axis, default in scatter_defaults.items():
            sel = DATA_DIMS.index(default) if default in DATA_DIMS else 0
            dd = PgDropdown(cx + 120, y, dd_w, dd_h, DATA_DIMS, sel, label="", font=f)
            self.scatter_dds[axis] = dd
            y += 36

        y += 14
        # ── Sliders ──────────────────────────────────────────────────────
        sl_w = cw - 10
        self.sliders = {}
        slider_defs = [("Spread", 0.61), ("Speed", 0.39), ("Reverb", 0.30)]
        for label, val in slider_defs:
            sl = PgSlider(cx + 5, y, sl_w, 20, 0, 1, val, label=label, font=f)
            self.sliders[label] = sl
            y += 42

        y += 10
        # ── Buttons ──────────────────────────────────────────────────────
        bw = (cw - 20) // 2
        self.btn_play = PgButton(cx + 5, y, bw, 32, "▶ Play", self.small)
        self.btn_record = PgButton(cx + 15 + bw, y, bw, 32, "⏺ Record",
                                    self.small, BTN_REC, BTN_REC_HI)
        y += 46
        self.btn_load = PgButton(cx + 5, y, cw - 10, 28, "Load HDF5...", self.small)

    # ── Drawing ──────────────────────────────────────────────────────────

    def _visible_range(self):
        """Return (start_idx, end_idx) into self.strain for the current zoom window."""
        n = len(self.strain)
        half = self.zoom / 2.0
        lo = max(0.0, self.zoom_center - half)
        hi = min(1.0, self.zoom_center + half)
        # clamp to keep window within data
        if lo == 0.0:
            hi = min(1.0, self.zoom)
        if hi == 1.0:
            lo = max(0.0, 1.0 - self.zoom)
        return int(lo * n), int(hi * n)

    def _draw_data_panels(self):
        """Draw waveform, spectrogram, and 3D scatter in the left area."""
        margin = 12
        ax_margin = 60   # left margin for axis labels
        pad = 8
        top = 44  # below title

        i0, i1 = self._visible_range()
        sr = self.info["sample_rate"]
        t_start = i0 / sr
        t_end = i1 / sr

        # ── Waveform ─────────────────────────────────────────────────────
        wh = 80
        wave_rect = pygame.Rect(margin + ax_margin, top, DATA_W - margin * 2 - ax_margin, wh)
        pygame.draw.rect(self.screen, PANEL_BG, wave_rect, border_radius=4)
        pygame.draw.rect(self.screen, DIM, wave_rect, width=1, border_radius=4)
        self.screen.blit(self.small.render("RAW STRAIN WAVEFORM", True, TXT_DIM),
                         (wave_rect.x + pad, top + 6))
        zoom_pct = int(100 / self.zoom) if self.zoom > 0 else 100
        self.screen.blit(self.small.render(
            f"zoom {zoom_pct}%  |  {t_start:.1f}s – {t_end:.1f}s  (scroll to zoom, shift+scroll to pan)",
            True, TXT_DIM), (wave_rect.x + 300, top + 6))
        pygame.draw.line(self.screen, (30, 30, 45),
                         (wave_rect.x, wave_rect.centery),
                         (wave_rect.right, wave_rect.centery))

        # Draw visible strain segment
        vis_strain = self.strain[i0:i1]
        n_pts = min(1600, len(vis_strain))
        step = max(1, len(vis_strain) // n_pts)
        samples = vis_strain[::step][:n_pts]
        peak = np.max(np.abs(samples))
        if peak > 0:
            samples = samples / peak
        for i in range(1, len(samples)):
            x0 = wave_rect.x + int((i - 1) / len(samples) * wave_rect.width)
            x1 = wave_rect.x + int(i / len(samples) * wave_rect.width)
            y0 = int(wave_rect.centery - samples[i - 1] * wave_rect.height * 0.42)
            y1 = int(wave_rect.centery - samples[i] * wave_rect.height * 0.42)
            pygame.draw.line(self.screen, ACCENT, (x0, y0), (x1, y1), 2)

        # Y-axis ticks (strain amplitude)
        for frac, label in [(-1.0, "+max"), (0.0, "0"), (1.0, "-max")]:
            y_pos = int(wave_rect.centery + frac * wave_rect.height * 0.42)
            pygame.draw.line(self.screen, DIM,
                             (wave_rect.x - 6, y_pos), (wave_rect.x, y_pos), 1)
            self.screen.blit(self.tiny.render(label, True, TXT_DIM),
                             (margin + 4, y_pos - 8))

        # X-axis ticks (time)
        n_ticks = 6
        for i in range(n_ticks + 1):
            frac = i / n_ticks
            x_pos = wave_rect.x + int(frac * wave_rect.width)
            t_val = t_start + frac * (t_end - t_start)
            pygame.draw.line(self.screen, DIM,
                             (x_pos, wave_rect.bottom), (x_pos, wave_rect.bottom + 5), 1)
            self.screen.blit(self.tiny.render(f"{t_val:.1f}s", True, TXT_DIM),
                             (x_pos - 14, wave_rect.bottom + 6))

        # Playback cursor
        if self.playing:
            play_pct = self._play_phase % 1.0
            vis_lo = i0 / len(self.strain)
            vis_hi = i1 / len(self.strain)
            if vis_lo <= play_pct <= vis_hi:
                local_pct = (play_pct - vis_lo) / (vis_hi - vis_lo)
                cx = wave_rect.x + int(local_pct * wave_rect.width)
                pygame.draw.line(self.screen, ACCENT2,
                                 (cx, wave_rect.y), (cx, wave_rect.bottom), 2)

        self.wave_rect = wave_rect  # store for scroll hit testing

        # ── Spectrogram ──────────────────────────────────────────────────
        spec_top = top + wh + 6
        sh = 70
        spec_rect = pygame.Rect(margin + ax_margin, spec_top, DATA_W - margin * 2 - ax_margin, sh)
        pygame.draw.rect(self.screen, PANEL_BG, spec_rect, border_radius=4)
        pygame.draw.rect(self.screen, DIM, spec_rect, width=1, border_radius=4)
        self.screen.blit(self.small.render("SPECTROGRAM", True, TXT_DIM),
                         (spec_rect.x + pad, spec_top + 6))
        if self.spec_surf:
            # Crop spectrogram to visible range
            full_w = self.spec_surf.get_width()
            crop_x = int((i0 / len(self.strain)) * full_w)
            crop_w = int(((i1 - i0) / len(self.strain)) * full_w)
            crop_w = max(1, min(crop_w, full_w - crop_x))
            cropped = self.spec_surf.subsurface((crop_x, 0, crop_w, self.spec_surf.get_height()))
            scaled = pygame.transform.smoothscale(cropped, (spec_rect.width - 8, sh - 36))
            self.screen.blit(scaled, (spec_rect.x + 4, spec_rect.y + 30))

            # Freq axis labels
            for frac, label in [(0.0, "2 kHz"), (0.5, "1 kHz"), (1.0, "0 Hz")]:
                y_pos = int(spec_rect.y + 30 + frac * (sh - 36))
                pygame.draw.line(self.screen, DIM,
                                 (spec_rect.x - 6, y_pos), (spec_rect.x, y_pos), 1)
                self.screen.blit(self.tiny.render(label, True, TXT_DIM),
                                 (margin, y_pos - 8))

            # Time axis ticks
            for i in range(n_ticks + 1):
                frac = i / n_ticks
                x_pos = spec_rect.x + 4 + int(frac * (spec_rect.width - 8))
                t_val = t_start + frac * (t_end - t_start)
                pygame.draw.line(self.screen, DIM,
                                 (x_pos, spec_rect.bottom), (x_pos, spec_rect.bottom + 5), 1)
                self.screen.blit(self.tiny.render(f"{t_val:.1f}s", True, TXT_DIM),
                                 (x_pos - 14, spec_rect.bottom + 6))

        if self.playing:
            play_pct = self._play_phase % 1.0
            vis_lo = i0 / len(self.strain)
            vis_hi = i1 / len(self.strain)
            if vis_lo <= play_pct <= vis_hi:
                local_pct = (play_pct - vis_lo) / (vis_hi - vis_lo)
                cx = spec_rect.x + 4 + int(local_pct * (spec_rect.width - 8))
                pygame.draw.line(self.screen, (255, 255, 255),
                                 (cx, spec_rect.y + 30), (cx, spec_rect.bottom - 4), 2)

        self.spec_rect = spec_rect  # store for scroll hit testing

        # ── 3D scatter (fills remaining vertical space) ──────────────────
        scat_top = spec_top + sh + 10
        scat_h = WIN_H - scat_top - 26
        self.scatter_rect = pygame.Rect(margin, scat_top, DATA_W - margin * 2, scat_h)
        pygame.draw.rect(self.screen, PANEL_BG, self.scatter_rect, border_radius=4)
        pygame.draw.rect(self.screen, DIM, self.scatter_rect, width=1, border_radius=4)

        sx = self.scatter_dds["X"].value
        sy = self.scatter_dds["Y"].value
        sz = self.scatter_dds["Z"].value
        sc = self.scatter_dds["Color"].value
        label_3d = f"3D SCATTER — X: {sx}  Y: {sy}  Z: {sz}  Color: {sc}"
        self.screen.blit(self.tiny.render(label_3d, True, TXT_DIM),
                         (margin + pad, scat_top + 4))
        self.screen.blit(self.tiny.render("drag to rotate · scroll to zoom", True, (60, 60, 80)),
                         (DATA_W - 170, scat_top + 4))

        inner = pygame.Rect(self.scatter_rect.x + 4, self.scatter_rect.y + 18,
                             self.scatter_rect.width - 8, self.scatter_rect.height - 22)
        self._draw_3d_scatter(inner, sx, sy, sz, sc)

    def _draw_3d_scatter(self, rect, dim_x, dim_y, dim_z, dim_c):
        surf = pygame.Surface((rect.width, rect.height))
        surf.fill(PANEL_BG)

        dx = self._get_dim(dim_x)
        dy = self._get_dim(dim_y)
        dz = self._get_dim(dim_z)
        dc = self._get_dim(dim_c)

        # Apply Spread slider as visual expansion (percentile stretch)
        spread = self.sliders["Spread"].val
        spread_factor = 1.0 + spread * 4.0  # 1× to 5× expansion

        def _spread_dim(arr):
            """Percentile-stretch then scale by spread factor."""
            p_lo = np.percentile(arr, 1)
            p_hi = np.percentile(arr, 99)
            if p_hi - p_lo > 1e-12:
                normed = (arr - p_lo) / (p_hi - p_lo)
            else:
                normed = arr
            return (normed - 0.5) * spread_factor

        n = len(dx)
        step = max(1, n // 1200)
        sx = _spread_dim(dx[::step])
        sy = _spread_dim(dy[::step])
        sz = _spread_dim(dz[::step])
        sc = dc[::step]

        ax = math.radians(self.rot_x)
        ay = math.radians(self.rot_y)
        cos_x, sin_x = math.cos(ax), math.sin(ax)
        cos_y, sin_y = math.cos(ay), math.sin(ay)

        x1 = sx * cos_y + sz * sin_y
        z1 = -sx * sin_y + sz * cos_y
        y1 = sy * cos_x - z1 * sin_x
        z2 = sy * sin_x + z1 * cos_x

        cx, cy = rect.width // 2, rect.height // 2
        scale = min(rect.width, rect.height) * 0.82 * self.scatter_zoom
        px = (cx + x1 * scale).astype(int)
        py = (cy - y1 * scale).astype(int)

        order = np.argsort(z2)
        for i in order:
            ci = min(255, max(0, int(sc[i] * 255)))
            col = tuple(int(c) for c in INFERNO[ci])
            x, y = int(px[i]), int(py[i])
            if 0 <= x < rect.width and 0 <= y < rect.height:
                pygame.draw.circle(surf, col, (x, y), 4)

        # Axes
        axes_data = [
            (np.array([0.5, 0, 0]), ACCENT, dim_x),
            (np.array([0, 0.5, 0]), ACCENT3, dim_y),
            (np.array([0, 0, 0.5]), ACCENT2, dim_z),
        ]
        for end, color, label in axes_data:
            ex = end[0] * cos_y + end[2] * sin_y
            ez = -end[0] * sin_y + end[2] * cos_y
            ey = end[1] * cos_x - ez * sin_x
            epx = int(cx + ex * scale)
            epy = int(cy - ey * scale)
            pygame.draw.line(surf, color, (cx, cy), (epx, epy), 2)
            lbl = self.tiny.render(label, True, color)
            surf.blit(lbl, (epx + 4, epy - 6))

        # ── Playback ruler (vertical plane sweeping along time axis) ──
        if self.playing:
            phase = self._play_phase % 1.0
            # The ruler is a plane at the current time position.
            # Project two 3D points on the time axis at the current phase,
            # spanning the full Y range, to form a vertical line in 3D.
            t_pos = (phase - 0.5) * spread_factor  # same scaling as data
            # Build 4 corners of a thin vertical plane at this time position
            plane_pts = []
            for py_off in [-0.5 * spread_factor, 0.5 * spread_factor]:
                for pz_off in [-0.5 * spread_factor, 0.5 * spread_factor]:
                    rx = t_pos * cos_y + pz_off * sin_y
                    rz = -t_pos * sin_y + pz_off * cos_y
                    ry = py_off * cos_x - rz * sin_x
                    ppx = int(cx + rx * scale)
                    ppy = int(cy - ry * scale)
                    plane_pts.append((ppx, ppy))
            # Draw as two triangles forming a translucent quad
            ruler_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            ruler_color = (255, 255, 80, 50)
            # Quad: corners are [0]=bot-near, [1]=bot-far, [2]=top-near, [3]=top-far
            # Draw as two triangles
            try:
                pygame.draw.polygon(ruler_surf, ruler_color,
                                    [plane_pts[0], plane_pts[1],
                                     plane_pts[3], plane_pts[2]])
                # Bright edge lines
                edge_col = (255, 255, 80, 180)
                pygame.draw.line(ruler_surf, edge_col,
                                 plane_pts[0], plane_pts[2], 2)
                pygame.draw.line(ruler_surf, edge_col,
                                 plane_pts[1], plane_pts[3], 2)
                pygame.draw.line(ruler_surf, edge_col,
                                 plane_pts[0], plane_pts[1], 1)
                pygame.draw.line(ruler_surf, edge_col,
                                 plane_pts[2], plane_pts[3], 1)
            except (ValueError, TypeError):
                pass
            surf.blit(ruler_surf, (0, 0))

            # Highlight the data points near the ruler
            # (points whose time value is close to current phase)
            time_dim = self._get_dim("time")
            time_ds = time_dim[::step]
            near_mask = np.abs(time_ds - phase) < 0.005
            for i in np.where(near_mask)[0]:
                hx, hy = int(px[i]), int(py[i])
                if 0 <= hx < rect.width and 0 <= hy < rect.height:
                    pygame.draw.circle(surf, (255, 255, 255), (hx, hy), 6)
                    pygame.draw.circle(surf, (255, 255, 80), (hx, hy), 4)

        # ── Color legend (inferno gradient bar + label) ───────────────
        bar_w, bar_h = 120, 10
        bar_x = rect.width - bar_w - 10
        bar_y = rect.height - bar_h - 20
        for bx in range(bar_w):
            ci = int(bx / bar_w * 255)
            col = tuple(int(c) for c in INFERNO[ci])
            pygame.draw.line(surf, col, (bar_x + bx, bar_y),
                             (bar_x + bx, bar_y + bar_h))
        pygame.draw.rect(surf, DIM, (bar_x - 1, bar_y - 1, bar_w + 2, bar_h + 2), 1)
        lo_lbl = self.tiny.render("0", True, TXT_DIM)
        hi_lbl = self.tiny.render("1", True, TXT_DIM)
        surf.blit(lo_lbl, (bar_x, bar_y + bar_h + 2))
        surf.blit(hi_lbl, (bar_x + bar_w - 8, bar_y + bar_h + 2))
        clr_lbl = self.tiny.render(f"color: {dim_c}", True, TXT_DIM)
        surf.blit(clr_lbl, (bar_x - 2, bar_y - 16))

        self.screen.blit(surf, rect.topleft)

    def _draw_controls(self):
        """Draw the right-side control panel."""
        ctrl_rect = pygame.Rect(DATA_W, 0, CTRL_W, WIN_H)
        pygame.draw.rect(self.screen, CTRL_BG, ctrl_rect)
        pygame.draw.line(self.screen, DIM, (DATA_W, 0), (DATA_W, WIN_H), 2)

        cx = DATA_W + 16
        f = self.small

        # Title
        self.screen.blit(self.font.render("Controls", True, ACCENT), (cx, 10))

        # ── Sonification mapping labels + dropdowns ──────────────────────
        y = 40
        self.screen.blit(f.render("AXIS → PARAMETER", True, TXT_DIM), (cx, y - 18))
        for param, dd in self.mapping_dds.items():
            self.screen.blit(f.render(f"{param}:", True, TXT), (cx, dd.rect.y + 3))
            dd.draw(self.screen)
            y = dd.rect.bottom + 14

        y = dd.rect.bottom + 20
        # ── 3D scatter axes labels + dropdowns ───────────────────────────
        self.screen.blit(f.render("3D SCATTER AXES", True, TXT_DIM), (cx, y - 4))
        y += 16
        for axis, dd in self.scatter_dds.items():
            self.screen.blit(f.render(f"{axis}:", True, TXT), (cx, dd.rect.y + 3))
            dd.draw(self.screen)

        # ── Sliders ──────────────────────────────────────────────────────
        for sl in self.sliders.values():
            sl.draw(self.screen)

        # ── Buttons ──────────────────────────────────────────────────────
        self.btn_play.draw(self.screen)
        self.btn_record.draw(self.screen)
        self.btn_load.draw(self.screen)

        # ── Dataset overview ─────────────────────────────────────────────
        stats_y = self.btn_load.rect.bottom + 20
        self.screen.blit(f.render("DATASET OVERVIEW", True, TXT_DIM), (cx, stats_y))
        stats_y += 18
        stats = compute_stats(self.strain, self.info)
        for label, value in stats:
            line = f"{label}: {value}"
            self.screen.blit(self.tiny.render(line, True, TXT_DIM), (cx, stats_y))
            stats_y += 14

        # ── Status ───────────────────────────────────────────────────────
        self.screen.blit(self.tiny.render(self.status, True, ACCENT2),
                         (cx, WIN_H - 22))

        # Draw open dropdowns on top
        for dd in self.mapping_dds.values():
            dd.draw_dropdown(self.screen)
        for dd in self.scatter_dds.values():
            dd.draw_dropdown(self.screen)

    # ── Event handling ───────────────────────────────────────────────────

    def handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
                return

            # Check dropdowns first (they overlay)
            any_open = any(dd.open for dd in
                           list(self.mapping_dds.values()) + list(self.scatter_dds.values()))

            if any_open:
                for dd in list(self.mapping_dds.values()) + list(self.scatter_dds.values()):
                    if dd.open:
                        changed = dd.handle_event(ev)
                        if changed and dd in self.scatter_dds.values():
                            self.dims_cache.clear()
                        break
                continue

            # Dropdowns (closed — check for open click)
            for dd in list(self.mapping_dds.values()) + list(self.scatter_dds.values()):
                dd.handle_event(ev)

            # Sliders
            for sl in self.sliders.values():
                sl.handle_event(ev)

            # Buttons
            if self.btn_play.handle_event(ev):
                self._on_play()
            if self.btn_record.handle_event(ev):
                self._on_record()
            if self.btn_load.handle_event(ev):
                self._on_load()

            # Scroll to zoom / pan on waveform or spectrogram
            if ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                on_data = ((hasattr(self, 'wave_rect') and self.wave_rect.collidepoint(mx, my)) or
                           (hasattr(self, 'spec_rect') and self.spec_rect.collidepoint(mx, my)))
                on_scatter = (hasattr(self, 'scatter_rect') and
                              self.scatter_rect.collidepoint(mx, my))
                if on_scatter:
                    # Scroll on 3D scatter = zoom in/out
                    factor = 1.15 if ev.y > 0 else 0.87
                    self.scatter_zoom = max(0.2, min(10.0, self.scatter_zoom * factor))
                    continue
                if on_data:
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        # Shift+scroll = pan
                        pan_step = self.zoom * 0.15
                        self.zoom_center = max(self.zoom / 2,
                                               min(1.0 - self.zoom / 2,
                                                   self.zoom_center - ev.y * pan_step))
                    else:
                        # Scroll = zoom
                        factor = 0.85 if ev.y > 0 else 1.18
                        self.zoom = max(self.min_zoom, min(1.0, self.zoom * factor))
                        # clamp center so window stays in range
                        half = self.zoom / 2.0
                        self.zoom_center = max(half, min(1.0 - half, self.zoom_center))
                    continue

            # 3D scatter drag
            if hasattr(self, 'scatter_rect'):
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    if self.scatter_rect.collidepoint(ev.pos):
                        self.dragging_3d = True
                        self.drag_start = ev.pos
                elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                    self.dragging_3d = False
                elif ev.type == pygame.MOUSEMOTION and self.dragging_3d:
                    dx = ev.pos[0] - self.drag_start[0]
                    dy = ev.pos[1] - self.drag_start[1]
                    self.rot_y += dx * 0.5
                    self.rot_x += dy * 0.5
                    self.drag_start = ev.pos

    # ── Actions ──────────────────────────────────────────────────────────

    def _get_mapping(self):
        mapping = {}
        for param, dd in self.mapping_dds.items():
            val = dd.value
            if val and val != "(none)":
                mapping[param] = val
        return mapping

    def _get_settings(self):
        return {
            "spread": self.sliders["Spread"].val,
            "speed_slider": self.sliders["Speed"].val,
            "reverb_mix": self.sliders["Reverb"].val,
            "bp_low": 10,
            "bp_high": 4000,
        }

    def _on_play(self):
        if self.playing:
            self._stop_playback()
            return
        self._start_realtime()

    def _stop_playback(self):
        """Stop playback with a quick fade-out to avoid clicks."""
        if self._master_vol is not None:
            try:
                self._master_vol.time = 0.05
                self._master_vol.value = 0
            except Exception:
                pass
            self._stop_pending = time.time() + 0.08
        else:
            self._cleanup_playback()

    def _cleanup_playback(self):
        """Tear down all pyo objects."""
        self.playing = False
        self._stop_pending = None
        for attr in ('_pan_out', '_reverb_node', '_filt_node', '_osc_node',
                      '_freq_ptr', '_vol_ptr', '_phasor', '_master_vol',
                      '_speed_sig', '_reverb_sig'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.stop()
                except Exception:
                    pass
                setattr(self, attr, None)
        self._phasor = None
        self._freq_table = None
        self._vol_table = None
        self.status = "Stopped"
        self.btn_play.text = "▶ Play"

    def _start_realtime(self):
        """Build a live pyo synthesis chain so sliders have immediate effect."""
        self.status = "Starting live playback..."
        sr = self.info["sample_rate"]
        data_dur = self.info["segment_duration"]
        self._data_dur = data_dur

        # ── frequency table from pitch mapping ───────────────────────
        mapping = self._get_mapping()
        pitch_dim = mapping.get("pitch", "strain")
        if pitch_dim == "(none)":
            pitch_dim = "strain"
        pitch_data = self._get_dim(pitch_dim)

        # Use percentile-based normalization so the full 0-1 range is used
        p_lo = np.percentile(pitch_data, 1)
        p_hi = np.percentile(pitch_data, 99)
        if p_hi - p_lo > 1e-12:
            pitch_norm = np.clip((pitch_data - p_lo) / (p_hi - p_lo), 0, 1)
        else:
            pitch_norm = pitch_data

        spread = self.sliders["Spread"].val
        self._last_spread = spread
        semi_range = 2 + spread * 46
        center_hz = 440.0
        pitch_mod = pitch_norm * 2 - 1
        freq_array = center_hz * np.power(2.0, pitch_mod * semi_range / 12.0)

        # downsample to ~1000 control points per second of data
        CTRL_SR = 1000
        table_len = int(data_dur * CTRL_SR)
        table_len = max(256, min(table_len, 120000))
        t_orig = np.linspace(0, 1, len(freq_array))
        t_new = np.linspace(0, 1, table_len)
        freq_ds = np.interp(t_new, t_orig, freq_array)

        # ── volume table ─────────────────────────────────────────────
        vol_dim = mapping.get("volume")
        if vol_dim and vol_dim != "(none)":
            vol_data = self._get_dim(vol_dim)
            vol_arr = 0.2 + vol_data * 0.8
            vol_ds = np.interp(t_new, np.linspace(0, 1, len(vol_arr)), vol_arr)
        else:
            vol_ds = np.ones(table_len) * 0.7

        # ── pyo objects ──────────────────────────────────────────────
        self._freq_table = DataTable(size=table_len, init=freq_ds.tolist())
        self._vol_table  = DataTable(size=table_len, init=vol_ds.tolist())

        speed_val = self.sliders["Speed"].val
        speed = 0.25 * (16 ** speed_val)
        read_freq = speed / data_dur
        self._last_speed_val = speed_val
        self._last_reverb_val = self.sliders["Reverb"].val

        self._speed_sig = SigTo(value=read_freq, time=0.15)
        self._phasor = Phasor(freq=self._speed_sig)
        self._freq_ptr = Pointer(self._freq_table, index=self._phasor)
        self._vol_ptr  = Pointer(self._vol_table, index=self._phasor)

        self._osc_node  = Sine(freq=self._freq_ptr, mul=self._vol_ptr)
        self._filt_node = Biquad(self._osc_node, freq=4000, q=0.7, type=0)

        rev_val = self.sliders["Reverb"].val
        self._reverb_sig  = SigTo(value=rev_val, time=0.1)
        self._reverb_node = Freeverb(self._filt_node, size=0.85,
                                      damp=0.5, bal=self._reverb_sig)

        self._master_vol = SigTo(value=0, time=0.05)
        self._pan_out = Pan(self._reverb_node, outs=2, pan=0.5,
                             mul=self._master_vol)
        self._pan_out.out()

        # fade in
        self._master_vol.time = 0.15
        self._master_vol.value = 0.6

        self.playing = True
        self.play_start = time.time()
        self._last_tick_time = time.time()
        self._play_phase = 0.0
        self.play_duration = data_dur / speed
        self.status = f"Playing live — {data_dur}s data"
        self.btn_play.text = "■ Stop"

    def _update_live_params(self):
        """Sync slider values to the running pyo chain every frame."""
        if not self.playing:
            return

        # handle pending stop after fade-out
        if self._stop_pending is not None:
            if time.time() >= self._stop_pending:
                self._cleanup_playback()
            return

        now = time.time()
        dt = now - self._last_tick_time
        self._last_tick_time = now

        # ── speed (only update when slider actually moved) ────────
        speed_val = self.sliders["Speed"].val
        if abs(speed_val - self._last_speed_val) > 0.002:
            self._last_speed_val = speed_val
            speed = 0.25 * (16 ** speed_val)
            new_freq = speed / self._data_dur
            if self._speed_sig is not None:
                self._speed_sig.value = new_freq
        else:
            speed = 0.25 * (16 ** speed_val)

        # advance cursor phase
        self._play_phase += dt * speed / self._data_dur
        self.play_duration = self._data_dur / speed

        # ── reverb (only update when slider actually moved) ──────────
        rev_val = self.sliders["Reverb"].val
        if abs(rev_val - self._last_reverb_val) > 0.002:
            self._last_reverb_val = rev_val
            if self._reverb_sig is not None:
                self._reverb_sig.value = rev_val

        # ── spread (recompute freq table if changed) ─────────────────
        spread = self.sliders["Spread"].val
        if abs(spread - self._last_spread) > 0.005:
            self._last_spread = spread
            self._recompute_freq_table(spread)

    def _recompute_freq_table(self, spread):
        """Recompute frequency table when Spread slider changes."""
        if self._freq_table is None:
            return
        mapping = self._get_mapping()
        pitch_dim = mapping.get("pitch", "strain")
        if pitch_dim == "(none)":
            pitch_dim = "strain"
        pitch_data = self._get_dim(pitch_dim)
        # percentile normalization
        p_lo = np.percentile(pitch_data, 1)
        p_hi = np.percentile(pitch_data, 99)
        if p_hi - p_lo > 1e-12:
            pitch_norm = np.clip((pitch_data - p_lo) / (p_hi - p_lo), 0, 1)
        else:
            pitch_norm = pitch_data
        semi_range = 2 + spread * 46
        center_hz = 440.0
        pitch_mod = pitch_norm * 2 - 1
        freq_array = center_hz * np.power(2.0, pitch_mod * semi_range / 12.0)
        table_len = self._freq_table.getSize()
        t_orig = np.linspace(0, 1, len(freq_array))
        t_new = np.linspace(0, 1, table_len)
        freq_ds = np.interp(t_new, t_orig, freq_array)
        self._freq_table.replace(freq_ds.tolist())

    def _on_record(self):
        self.status = "Recording..."

        def _do():
            mapping = self._get_mapping()
            settings = self._get_settings()
            stereo, sr = render_audio(
                self.strain, self.info["sample_rate"], mapping, **settings)
            os.makedirs(EXPORT_DIR, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(EXPORT_DIR, f"gw_sonified_{ts}.wav")
            save_stereo_wav(out_path, stereo, sr)
            duration = len(stereo) / sr
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            self.status = f"Saved: {os.path.basename(out_path)} — {duration:.1f}s, {size_mb:.1f} MB"

        threading.Thread(target=_do, daemon=True).start()

    def _on_load(self):
        import tkinter as tk
        from tkinter import filedialog
        # Stop playback before loading
        if self.playing:
            self._stop_playback()
        # Minimize pygame so tkinter dialog is clickable
        pygame.display.iconify()
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        path = filedialog.askopenfilename(
            parent=root,
            title="Select GWOSC HDF5 file",
            filetypes=[("HDF5 files", "*.hdf5"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~/Downloads"))
        root.destroy()
        # Restore pygame window
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        if not path:
            return
        self.status = f"Loading {os.path.basename(path)}..."
        try:
            strain, info = load_gw_strain(path)
            self.strain = strain
            self.info = info
            self.dims_cache.clear()
            self._build_spectrogram()
            self.status = f"Loaded: {info['detector']} | {info['samples']:,} samples"
        except Exception as e:
            self.status = f"Error: {e}"

    # ── Main loop ────────────────────────────────────────────────────────

    def draw(self):
        self.screen.fill(BG)
        # Title bar
        det = self.info["detector"]
        dur = self.info["segment_duration"]
        sr = self.info["sample_rate"]
        title = f"GW Explorer — {det} | {dur}s @ {sr} Hz | {len(self.strain):,} samples"
        self.screen.blit(self.font.render(title, True, ACCENT), (12, 10))

        self._draw_data_panels()
        self._draw_controls()
        pygame.display.flip()

    def tick(self):
        self.handle_events()
        self._update_live_params()
        self.draw()
        self.clock.tick(FPS)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GW Explorer — LIGO Data Sonifier")
    parser.add_argument("hdf5_file", nargs="?", default=DEFAULT_HDF5,
                        help="Path to GWOSC HDF5 file")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    if not os.path.isfile(args.hdf5_file):
        sys.exit(f"File not found: {args.hdf5_file}")

    print(f"Loading {os.path.basename(args.hdf5_file)}...")
    strain, info = load_gw_strain(args.hdf5_file, args.start, args.duration)
    print(f"  {info['detector']} | {info['samples']:,} samples "
          f"@ {info['sample_rate']} Hz | {info['segment_duration']}s")

    print("  Booting audio server...")
    dev = 13
    server = Server(sr=PLAYBACK_SR, nchnls=2, duplex=0, buffersize=1024,
                    audio="portaudio")
    server.setOutputDevice(dev)
    server.boot()
    server.start()

    app = GWExplorer(strain, info)
    print("  Ready.")

    try:
        while app.running:
            app.tick()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        server.shutdown()
        pygame.quit()
        print("Done.")


if __name__ == "__main__":
    main()
