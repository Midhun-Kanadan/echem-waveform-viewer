"""
Shared helpers for the electrochemical viewer and the export scripts:
noise-relative detection guards, optional denoising, multi-file averaging,
and Excel export with native (editable) Excel charts.
"""

import io

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

FS = 10_000  # DL850E sample rate (Hz)

DENOISE_METHODS = ["Off", "Low-pass (zero-phase)", "Savitzky-Golay"]


# ── Noise-relative detection guards ───────────────────────────────────────────
def noise_sigma(x) -> float:
    """Robust white-noise std from first differences (insensitive to the waveform)."""
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(np.diff(x))) / 0.6745 / np.sqrt(2))


def is_flat(x, min_range_sigmas: float = 20.0) -> bool:
    """True for files without real pulsing (e.g. instrument initialisation).

    Signal range (p1→p99) is compared with the noise level, so the test works for
    both large-amplitude samples (PPGa7865, ~500 σ) and small ones (7917, ~50 σ);
    pure-noise files are ~5–7 σ.
    """
    rng = float(np.percentile(x, 99) - np.percentile(x, 1))
    return rng < min_range_sigmas * noise_sigma(x)


def min_spike_amp(x) -> float:
    """Minimum height above the 30th-percentile baseline for a real anodic spike.

    Scaled to the T_on/T_off swing (p5→p90). Real spikes are ~3–4× the swing in
    every sample seen so far; 0.5× the swing reproduces the old fixed 0.2 V for
    PPGa7865 while also working for the ~10× smaller 7917 signal.
    """
    swing = float(np.percentile(x, 90) - np.percentile(x, 5))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    return max(0.5 * swing, 0.5 * abs(iqr), 15.0 * noise_sigma(x))


# ── Denoising ─────────────────────────────────────────────────────────────────
def denoise(x, method: str = "Off", cutoff_hz: float = 500.0, sg_window_ms: float = 5.0,
            sg_order: int = 3):
    """Return a denoised copy of x. Never shifts features in time.

    - "Low-pass (zero-phase)": 4th-order Butterworth applied forward+backward (filtfilt).
    - "Savitzky-Golay": local polynomial fit, window in ms.
    """
    x = np.asarray(x, dtype=float)
    if method.startswith("Low-pass"):
        b, a = butter(4, min(cutoff_hz, 0.45 * FS) / (FS / 2))
        return filtfilt(b, a, x)
    if method.startswith("Savitzky"):
        win = max(sg_order + 2, int(round(sg_window_ms * FS / 1000)) | 1)  # odd
        return savgol_filter(x, win, sg_order)
    return x.copy()


def denoise_label(method: str, cutoff_hz: float, sg_window_ms: float, n_avg: int = 1) -> str:
    parts = []
    if method.startswith("Low-pass"):
        parts.append(f"zero-phase low-pass {cutoff_hz:.0f} Hz")
    elif method.startswith("Savitzky"):
        parts.append(f"Savitzky-Golay {sg_window_ms:g} ms")
    if n_avg > 1:
        parts.append(f"average of {n_avg} files")
    return ", ".join(parts) if parts else "raw"


# ── Averaging spike-aligned windows from several files ────────────────────────
def average_windows(wins: list) -> dict:
    """Average windows (dicts with t_ms/ch1/ch2[/ch1_raw/ch2_raw]) aligned at τ = 0.

    All windows sit on the same 0.1 ms grid relative to their spike, so they are
    aligned by integer sample index; only the τ range common to all is kept.
    """
    idx = [np.round(np.asarray(w["t_ms"]) * FS / 1000).astype(int) for w in wins]
    lo, hi = max(i[0] for i in idx), min(i[-1] for i in idx)
    out = {"t_ms": np.arange(lo, hi + 1) * 1000 / FS}
    for key in ("ch1", "ch2", "ch1_raw", "ch2_raw"):
        if all(key in w for w in wins):
            out[key] = np.mean([np.asarray(w[key])[(i >= lo) & (i <= hi)]
                                for w, i in zip(wins, idx)], axis=0)
    return out


# ── Per-file transient features (Transient Evolution) ─────────────────────────
FEATURE_INFO = {
    # key: (label, unit, explanation)
    "pause_V":        ("Φ at end of pause", "V",
                       "Resting level the cell relaxes to when no current flows (last 2 ms of the pause)."),
    "pulse_end_V":    ("Φ at end of cathodic pulse", "V",
                       "Potential needed to drive the deposition current (last 0.5 ms of the pulse)."),
    "pulse_depth_mV": ("Pulse depth (pause − pulse)", "mV",
                       "How far the deposition pulse pulls the potential below the resting level."),
    "spike_height_mV": ("Anodic spike height above pause", "mV",
                        "Peak of the anodic (reverse) pulse relative to the resting level."),
    "spike_peak_V":   ("Anodic spike peak", "V", "Absolute peak potential of the anodic pulse."),
    "tau63_ms":       ("Recovery time (63 %)", "ms",
                       "Time after the pulse ends until 63 % of the relaxation is done (≈ RC time constant)."),
    "i_cath_V":       ("Cathodic current (shunt)", "V", "Imposed deposition current — should be constant."),
    "i_anod_V":       ("Anodic current (shunt)", "V", "Imposed reverse current — should be constant."),
}


def cycle_features(ch1, ch2, fs: int = FS):
    """Features of one capture, from the average of all regular pulse cycles in it.

    Phases are found from the *current* (ch2), which is imposed and identical for all
    samples, so this works independently of the potential's scale:
      cathodic pulse = current below half its 10th-percentile level,
      anodic pulse   = current above half its 99.5th-percentile level.
    Cycles touching an anodic pulse are excluded. Returns None for flat/unusable files.
    """
    from scipy.signal import medfilt
    x = np.asarray(ch1, dtype=float); i = np.asarray(ch2, dtype=float)
    if len(x) < fs // 2 or is_flat(x):
        return None
    ism = medfilt(i, 11)
    lo_lvl, hi_lvl = np.percentile(ism, 10), np.percentile(ism, 99.5)
    if lo_lvl >= 0 or hi_lvl <= 0:
        return None
    on = ism < 0.5 * lo_lvl
    anod = ism > 0.5 * hi_lvl
    starts = np.where(np.diff(on.astype(int)) == 1)[0] + 1
    ends = np.where(np.diff(on.astype(int)) == -1)[0] + 1
    if len(starts) < 3 or len(ends) < 3:
        return None
    pulse_len = int(np.median([e - s for s in starts for e in ends[ends > s][:1]]))
    period = int(np.median(np.diff(starts)))
    pause_len = period - pulse_len
    if pulse_len < 5 or pause_len < 20:
        return None
    guard = int(0.004 * fs)
    anod_idx = np.where(anod)[0]
    cyc = []
    for e in ends:
        a, b = e - pulse_len, e + pause_len - guard
        if a < 0 or b > len(x):
            continue
        if len(anod_idx) and np.any((anod_idx > a - period // 2) & (anod_idx < b + guard)):
            continue
        cyc.append(x[a:b])
    if len(cyc) < 2:
        return None
    m = np.mean(cyc, axis=0)
    pe = int(0.0005 * fs); pz = int(0.002 * fs)
    pulse_end_V = float(m[pulse_len - pe: pulse_len].mean())
    pause_V = float(m[-pz:].mean())
    rec = m[pulse_len:] - pulse_end_V
    tgt = 0.632 * (pause_V - pulse_end_V)
    tau = float(np.argmax(rec >= tgt) / fs * 1000) if tgt > 0 and np.any(rec >= tgt) else np.nan

    # anodic spike: peak of lightly low-passed potential inside each anodic pulse (+2 ms)
    xl = denoise(x, "Low-pass", 1000)
    runs = np.split(anod_idx, np.where(np.diff(anod_idx) > 1)[0] + 1) if len(anod_idx) else []
    peaks = [xl[r[0]: min(len(x), r[-1] + int(0.002 * fs))].max() for r in runs if len(r) >= 5]
    spike_peak = float(np.median(peaks)) if peaks else np.nan
    return {
        "pause_V": pause_V,
        "pulse_end_V": pulse_end_V,
        "pulse_depth_mV": (pause_V - pulse_end_V) * 1000,
        "spike_peak_V": spike_peak,
        "spike_height_mV": (spike_peak - pause_V) * 1000 if peaks else np.nan,
        "tau63_ms": tau,
        "i_cath_V": float(np.median(i[on])),
        "i_anod_V": float(np.median(i[anod])) if anod.any() else np.nan,
        "n_cycles": len(cyc),
        "n_anodic": len(peaks),
        "pulse_ms": pulse_len / fs * 1000,
        "pause_ms": pause_len / fs * 1000,
    }


# ── Excel export with native charts ───────────────────────────────────────────
def _style_series(s, color_hex: str, width_pt: float = 0.75):
    s.marker.symbol = "none"
    s.smooth = False
    s.graphicalProperties.line.solidFill = color_hex.lstrip("#").upper()
    s.graphicalProperties.line.width = int(width_pt * 12700)


def _axis(ax, title, lo=None, hi=None, major=None, num_fmt=None):
    from openpyxl.chart.title import title_maker
    ax.title = title_maker(title)
    ax.title.overlay = False           # otherwise Excel draws it over the tick labels
    ax.delete = False
    if lo is not None:
        ax.scaling.min = float(lo)
    if hi is not None:
        ax.scaling.max = float(hi)
    if major:
        ax.majorUnit = float(major)
    if num_fmt:
        ax.number_format = num_fmt
    ax.majorGridlines = None


def _nice_x(lo, hi, step=50.0):
    """Round the τ axis to multiples of `step` ms so Excel's ticks land on round values."""
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


def _scatter(title, x_title, x_lo=None, x_hi=None):
    from openpyxl.chart import ScatterChart
    from openpyxl.chart.title import title_maker
    ch = ScatterChart()
    ch.title = title_maker(title)
    ch.title.overlay = False           # keep the title above the plot area
    ch.style = 13
    ch.scatterStyle = "line"
    ch.height, ch.width = 11, 26
    ch.legend.position = "b"
    ch.legend.overlay = False
    _axis(ch.x_axis, x_title, x_lo, x_hi)
    ch.x_axis.crosses = "min"          # x axis at the bottom even when all Φ < 0
    ch.x_axis.tickLblPos = "low"
    return ch


def _write_table(ws, headers, cols, start_row=1):
    ws.append(headers)
    for row in zip(*cols):
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in row])
    for i, h in enumerate(headers, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = max(12, len(h) + 2)
    ws.freeze_panes = "A2"


def signal_excel(t_ms, phi, cur, title: str, phi_raw=None, cur_raw=None, note: str = "",
                 phi_color="#1F3864", cur_color="#FF0000",
                 phi_range=(None, None), cur_range=(None, None)) -> bytes:
    """Workbook with the data table and a dual-axis Potential/Current chart.

    If raw arrays are given (denoising active), they are written as extra columns
    and the chart shows the denoised curves.
    """
    from openpyxl import Workbook
    from openpyxl.chart import Reference, ScatterChart, Series

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    headers = ["Time (ms)", "Potential (V)", "Current (V)"]
    cols = [t_ms, phi, cur]
    if phi_raw is not None:
        headers += ["Potential raw (V)", "Current raw (V)"]
        cols += [phi_raw, cur_raw]
    _write_table(ws, headers, cols)
    n = len(t_ms) + 1

    x = Reference(ws, min_col=1, min_row=2, max_row=n)
    c1 = _scatter(title, "τ in ms", *_nice_x(float(np.min(t_ms)), float(np.max(t_ms))))
    s = Series(Reference(ws, min_col=2, min_row=1, max_row=n), x, title_from_data=True)
    _style_series(s, phi_color)
    c1.series.append(s)
    _axis(c1.y_axis, "Φ in V", *phi_range)
    c1.y_axis.crosses = "min"

    c2 = ScatterChart()
    s2 = Series(Reference(ws, min_col=3, min_row=1, max_row=n), x, title_from_data=True)
    _style_series(s2, cur_color)
    c2.series.append(s2)
    c2.y_axis.axId = 200
    _axis(c2.y_axis, "Current (V across shunt)", *cur_range)
    c2.y_axis.crosses = "max"
    c1 += c2

    cs = wb.create_sheet("Chart", 0)
    cs["A1"] = title
    if note:
        cs["A2"] = note
    cs.add_chart(c1, "A4")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def evolution_excel(df, feature_keys: list, title: str, note: str = "") -> bytes:
    """Workbook for the Transient Evolution tab: Data sheet + one native chart per feature
    (feature vs. deposition time in hours)."""
    from openpyxl import Workbook
    from openpyxl.chart import Reference, Series

    wb = Workbook()
    ws = wb.active
    ws.title = "Data"
    cols = ["time_h", "file"] + [k for k in feature_keys if k in df.columns] + \
           [c for c in ("n_cycles", "n_anodic", "pulse_ms", "pause_ms") if c in df.columns]
    heads = []
    for c in cols:
        if c in FEATURE_INFO:
            lbl, unit, _ = FEATURE_INFO[c]
            heads.append(f"{lbl} ({unit})")
        else:
            heads.append({"time_h": "Deposition time (h)", "file": "File"}.get(c, c))
    ws.append(heads)
    for _, r in df[cols].iterrows():
        ws.append([None if (isinstance(v, float) and np.isnan(v)) else
                   (v if isinstance(v, str) else float(v)) for v in r.values])
    for k, h in enumerate(heads, start=1):
        ws.column_dimensions[ws.cell(row=1, column=k).column_letter].width = max(12, min(40, len(h) + 2))
    ws.freeze_panes = "A2"
    n = len(df) + 1

    cs = wb.create_sheet("Charts", 0)
    cs["A1"] = title
    if note:
        cs["A2"] = note
    x = Reference(ws, min_col=1, min_row=2, max_row=n)
    row = 4
    for k in feature_keys:
        if k not in cols:
            continue
        j = cols.index(k) + 1
        lbl, unit, _ = FEATURE_INFO[k]
        th = df["time_h"].astype(float)
        ch = _scatter(lbl, "Deposition time (h)", 0.0, float(np.ceil(th.max())) if len(th) else None)
        ch.scatterStyle = "lineMarker"
        ch.varyColors = False
        ch.height, ch.width = 7.5, 22
        ch.legend = None
        s = Series(Reference(ws, min_col=j, min_row=1, max_row=n), x, title_from_data=True)
        s.marker.symbol = "circle"; s.marker.size = 3
        s.marker.graphicalProperties.solidFill = "1F3864"
        s.marker.graphicalProperties.line.solidFill = "1F3864"
        s.graphicalProperties.line.solidFill = "1F3864"
        s.graphicalProperties.line.width = 9525
        ch.series.append(s)
        yv = df[k].astype(float).dropna()
        ylo = yhi = step = None
        if len(yv):   # zoom to the data (Excel would start at 0), on round tick values
            lo_, hi_ = float(yv.min()), float(yv.max())
            sp = (hi_ - lo_) or abs(lo_) * 0.05 or 1.0
            step = 10 ** np.floor(np.log10(sp / 5))
            step *= next(m for m in (1, 2, 2.5, 5, 10) if sp / (step * m) <= 6)
            ylo, yhi = float(np.floor(lo_ / step) * step), float(np.ceil(hi_ / step) * step)
        _axis(ch.y_axis, f"{lbl} ({unit})", ylo, yhi, step)
        ch.y_axis.crosses = "min"
        cs.add_chart(ch, f"A{row}")
        row += 16
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def stacked_excel(curves: list, info_rows: list, title: str, note: str = "",
                  current_curve: dict = None, phi_range=(None, None), cur_range=(None, None)) -> bytes:
    """Workbook for the stacked-evolution plot.

    curves: list of {"label", "t_ms", "phi", "color"} (already aligned/offset).
    All curves are placed on one common 0.1 ms τ grid (NaN → empty cell), so every
    value sits at its true τ — unlike pasting columns side by side.
    Sheets: Chart, Stacked (common grid), Files (source table).
    """
    from openpyxl import Workbook
    from openpyxl.chart import Reference, ScatterChart, Series

    allc = curves + ([current_curve] if current_curve else [])
    lo = min(int(round(c["t_ms"][0] * 10)) for c in allc)
    hi = max(int(round(c["t_ms"][-1] * 10)) for c in allc)
    grid = np.arange(lo, hi + 1)

    def on_grid(c, key):
        col = np.full(len(grid), np.nan)
        k = np.round(np.asarray(c["t_ms"]) * 10).astype(int) - lo
        col[k] = c[key]
        return col

    wb = Workbook()
    ws = wb.active
    ws.title = "Stacked"
    headers = ["τ (ms)"] + [f"Potential {c['label']} (V)" for c in curves]
    cols = [grid / 10.0] + [on_grid(c, "phi") for c in curves]
    if current_curve:
        headers.append(f"Current {current_curve['label']} (V)")
        cols.append(on_grid(current_curve, "cur"))
    _write_table(ws, headers, cols)
    n = len(grid) + 1

    x = Reference(ws, min_col=1, min_row=2, max_row=n)
    c1 = _scatter(title, "τ in ms", *_nice_x(lo / 10.0, hi / 10.0))
    c1.dispBlanksAs = "gap"
    for j, c in enumerate(curves, start=2):
        s = Series(Reference(ws, min_col=j, min_row=1, max_row=n), x, title_from_data=True)
        _style_series(s, c["color"])
        c1.series.append(s)
    _axis(c1.y_axis, "Φ in V", *phi_range)
    c1.y_axis.crosses = "min"
    if current_curve:
        c2 = ScatterChart()
        s2 = Series(Reference(ws, min_col=len(curves) + 2, min_row=1, max_row=n), x, title_from_data=True)
        _style_series(s2, current_curve["color"])
        c2.series.append(s2)
        c2.y_axis.axId = 200
        _axis(c2.y_axis, "Current (V across shunt)", *cur_range)
        c2.y_axis.crosses = "max"
        c1 += c2

    cs = wb.create_sheet("Chart", 0)
    cs["A1"] = title
    if note:
        cs["A2"] = note
    cs.add_chart(c1, "A4")

    fs = wb.create_sheet("Files")
    if info_rows:
        keys = list(info_rows[0].keys())
        fs.append(keys)
        for r in info_rows:
            fs.append([str(r[k]) if not isinstance(r[k], (int, float)) else r[k] for k in keys])
        for i, k in enumerate(keys, start=1):
            fs.column_dimensions[fs.cell(row=1, column=i).column_letter].width = max(14, len(k) + 4)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
