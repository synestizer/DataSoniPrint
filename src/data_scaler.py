"""
data_scaler.py — Map arbitrary numeric data to audio parameters.

Inspired by listentocolors.net: translate any dataset into audible sequences.

Usage:
    from data_scaler import DataScaler

    scaler = DataScaler()
    scaler.load_csv("my_data.csv")
    # or: scaler.load_values([0.1, 0.5, 0.9, 0.3, ...])
    events = scaler.to_events(
        freq_col="temperature",   # column name or index
        amp_col="pressure",       # optional: column for amplitude
        dur_col=None,             # optional: column for note duration
    )
    # events = [{"freq": 440.0, "amp": 0.3, "dur": 0.25}, ...]

Scaling approach:
    - Frequency: linear or log map from data range → audible range (80–4000 Hz)
    - Amplitude: linear map from data range → 0.01–0.5
    - Duration: linear map from data range → 0.05–2.0 seconds
    - All ranges are configurable
"""

import csv
import math


def _read_csv(path):
    """Read CSV, return (headers, rows) where rows are list of float lists."""
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        rows = []
        for row in reader:
            parsed = []
            for cell in row:
                cell = cell.strip()
                try:
                    parsed.append(float(cell))
                except ValueError:
                    parsed.append(None)
            rows.append(parsed)
    return headers, rows


def _col_index(headers, col):
    """Resolve a column name or index to an integer index."""
    if col is None:
        return None
    if isinstance(col, int):
        return col
    if headers and col in headers:
        return headers.index(col)
    raise ValueError(f"Column '{col}' not found. Available: {headers}")


def _col_values(rows, idx):
    """Extract numeric values from a column, skipping None."""
    if idx is None:
        return []
    return [row[idx] for row in rows if idx < len(row) and row[idx] is not None]


def lin_scale(value, src_min, src_max, dst_min, dst_max):
    """Linear interpolation from source range to destination range."""
    if src_max == src_min:
        return (dst_min + dst_max) / 2.0
    t = (value - src_min) / (src_max - src_min)
    t = max(0.0, min(1.0, t))
    return dst_min + t * (dst_max - dst_min)


def log_scale(value, src_min, src_max, dst_min, dst_max):
    """Logarithmic scaling — good for frequency mapping (perceptual pitch)."""
    if src_max == src_min:
        return math.sqrt(dst_min * dst_max)
    t = (value - src_min) / (src_max - src_min)
    t = max(0.0, min(1.0, t))
    log_min = math.log(max(dst_min, 1.0))
    log_max = math.log(max(dst_max, 1.0))
    return math.exp(log_min + t * (log_max - log_min))


class DataScaler:
    """Load data, map columns to audio parameters, emit event dicts."""

    def __init__(
        self,
        freq_range=(80.0, 4000.0),
        amp_range=(0.02, 0.45),
        dur_range=(0.05, 1.5),
        freq_scale="log",  # "log" or "lin"
    ):
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.dur_range = dur_range
        self.freq_scale = freq_scale

        self.headers = None
        self.rows = []

    # -- Loading --

    def load_csv(self, path):
        """Load a CSV file. First row = headers."""
        self.headers, self.rows = _read_csv(path)
        print(f"Loaded {len(self.rows)} rows, columns: {self.headers}")
        return self

    def load_values(self, values, column_name="value"):
        """Load a flat list of numbers as a single-column dataset."""
        self.headers = [column_name]
        self.rows = [[v] for v in values]
        return self

    def load_rows(self, headers, rows):
        """Load pre-parsed data directly."""
        self.headers = headers
        self.rows = rows
        return self

    def load_hdf5(self, path, group=None, datasets=None):
        """
        Load an HDF5 file.

        Args:
            path: path to .h5 / .hdf5 file
            group: HDF5 group to read from (default: root "/")
            datasets: list of dataset names to load (default: all 1-D numeric datasets)

        If datasets have different lengths, rows are padded with None.
        """
        import h5py
        with h5py.File(path, "r") as f:
            root = f[group] if group else f

            # discover datasets
            available = []
            def _collect(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 1:
                    try:
                        if obj.dtype.kind in ("f", "i", "u"):  # float, int, unsigned
                            available.append(name)
                    except Exception:
                        pass
            root.visititems(_collect)

            if not available:
                raise ValueError(f"No 1-D numeric datasets found in '{group or '/'}'."
                                 f" Contents: {list(root.keys())}")

            if datasets:
                # validate requested datasets exist
                for d in datasets:
                    if d not in available:
                        raise ValueError(f"Dataset '{d}' not found. Available: {available}")
                use = datasets
            else:
                use = available

            print(f"HDF5 datasets: {use}")

            # read columns
            columns = {}
            max_len = 0
            for name in use:
                data = root[name][:]
                columns[name] = [float(v) for v in data]
                max_len = max(max_len, len(columns[name]))

            # build rows (pad shorter columns with None)
            self.headers = list(use)
            self.rows = []
            for i in range(max_len):
                row = []
                for name in use:
                    col = columns[name]
                    row.append(col[i] if i < len(col) else None)
                self.rows.append(row)

        print(f"Loaded {len(self.rows)} rows from HDF5, columns: {self.headers}")
        return self

    # -- Info --

    def column_stats(self, col):
        """Return (min, max, mean, count) for a column."""
        idx = _col_index(self.headers, col)
        vals = _col_values(self.rows, idx)
        if not vals:
            return None
        return {
            "min": min(vals),
            "max": max(vals),
            "mean": sum(vals) / len(vals),
            "count": len(vals),
        }

    def summary(self):
        """Print a summary of all numeric columns."""
        if not self.headers:
            print("No data loaded.")
            return
        for h in self.headers:
            stats = self.column_stats(h)
            if stats:
                print(f"  {h}: min={stats['min']:.4f}  max={stats['max']:.4f}  "
                      f"mean={stats['mean']:.4f}  n={stats['count']}")

    # -- Mapping --

    def to_events(self, freq_col=0, amp_col=None, dur_col=None,
                  default_amp=0.2, default_dur=0.25):
        """
        Map data rows to a list of audio event dicts.

        Each event: {"freq": Hz, "amp": 0-1, "dur": seconds, "row_index": int}

        Parameters:
            freq_col: column for frequency mapping (required)
            amp_col: column for amplitude mapping (or None → default_amp)
            dur_col: column for duration mapping (or None → default_dur)
            default_amp: amplitude when amp_col is None
            default_dur: duration when dur_col is None
        """
        freq_idx = _col_index(self.headers, freq_col)
        amp_idx = _col_index(self.headers, amp_col)
        dur_idx = _col_index(self.headers, dur_col)

        freq_vals = _col_values(self.rows, freq_idx)
        amp_vals = _col_values(self.rows, amp_idx)
        dur_vals = _col_values(self.rows, dur_idx)

        freq_min, freq_max = (min(freq_vals), max(freq_vals)) if freq_vals else (0, 1)
        amp_min, amp_max = (min(amp_vals), max(amp_vals)) if amp_vals else (0, 1)
        dur_min, dur_max = (min(dur_vals), max(dur_vals)) if dur_vals else (0, 1)

        scale_freq = log_scale if self.freq_scale == "log" else lin_scale

        events = []
        for i, row in enumerate(self.rows):
            # frequency (required)
            if freq_idx is None or freq_idx >= len(row) or row[freq_idx] is None:
                continue
            freq = scale_freq(row[freq_idx], freq_min, freq_max,
                              self.freq_range[0], self.freq_range[1])

            # amplitude
            if amp_idx is not None and amp_idx < len(row) and row[amp_idx] is not None:
                amp = lin_scale(row[amp_idx], amp_min, amp_max,
                                self.amp_range[0], self.amp_range[1])
            else:
                amp = default_amp

            # duration
            if dur_idx is not None and dur_idx < len(row) and row[dur_idx] is not None:
                dur = lin_scale(row[dur_idx], dur_min, dur_max,
                                self.dur_range[0], self.dur_range[1])
            else:
                dur = default_dur

            events.append({
                "freq": round(freq, 2),
                "amp": round(amp, 4),
                "dur": round(dur, 4),
                "row_index": i,
            })

        return events

    def to_chord_events(self, columns, default_dur=0.5):
        """
        Map multiple columns simultaneously — each column becomes a
        frequency in a chord played at each time step.

        Returns: [{"freqs": [f1, f2, ...], "dur": seconds, "row_index": int}, ...]
        """
        col_indices = [_col_index(self.headers, c) for c in columns]

        # compute per-column ranges
        col_ranges = []
        for idx in col_indices:
            vals = _col_values(self.rows, idx)
            col_ranges.append((min(vals), max(vals)) if vals else (0, 1))

        scale_freq = log_scale if self.freq_scale == "log" else lin_scale

        events = []
        for i, row in enumerate(self.rows):
            freqs = []
            for idx, (cmin, cmax) in zip(col_indices, col_ranges):
                if idx is not None and idx < len(row) and row[idx] is not None:
                    f = scale_freq(row[idx], cmin, cmax,
                                   self.freq_range[0], self.freq_range[1])
                    freqs.append(round(f, 2))
            if freqs:
                events.append({
                    "freqs": freqs,
                    "amp": 0.15,
                    "dur": default_dur,
                    "row_index": i,
                })
        return events
