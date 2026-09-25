"""
Electrochemical Signal Viewer
Interactive Streamlit app for Yokogawa DL850E transient recorder data
(pulse plating: signal viewer, stacked evolution, transient evolution).
"""

import io
from datetime import datetime as _dt, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import to_hex

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from scipy.signal import find_peaks

from plotly.subplots import make_subplots

# Streamlit (incl. Community Cloud after a git push) re-runs app.py without restarting Python,
# so an already-imported echem_utils would stay at its old version and new names would fail to
# import. Reload it whenever the file on disk has changed.
import importlib
import os
import echem_utils
_eu_mtime = os.path.getmtime(echem_utils.__file__)
if getattr(echem_utils, "_loaded_mtime", None) != _eu_mtime:
    echem_utils = importlib.reload(echem_utils)
    echem_utils._loaded_mtime = _eu_mtime

from echem_utils import (DENOISE_METHODS, FEATURE_INFO, average_windows, cycle_features,
                         cycle_minima, denoise, denoise_label, evolution_excel, is_flat,
                         min_spike_amp, minima_meansd_excel, signal_excel, stacked_excel)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Electrochemical Viewer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    h1 { font-size: 1.5rem !important; }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 8px;
        padding: 10px 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Electrochemical Signal Viewer")
st.caption("Yokogawa DL850E · Filtered TXT files · Dual-axis interactive plot")

# ── Module-level helpers ───────────────────────────────────────────────────────
def _parse_ts(name: str):
    """Parse timestamp from filename like f20260323_154935_939_filter.txt."""
    try:
        d, t = name[1:9], name[10:16]
        return _dt(int(d[:4]), int(d[4:6]), int(d[6:8]),
                   int(t[:2]), int(t[2:4]), int(t[4:6]))
    except Exception:
        return None


def _read_df(src) -> pd.DataFrame:
    """src is either a file path (str, local-folder mode) or the file's bytes (upload mode)."""
    buf = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    df = pd.read_csv(buf, sep="\t", header=None, names=["time", "ch1", "ch2"])
    df = df.iloc[10:].reset_index(drop=True)            # drop filter warm-up
    df = df[df["time"] <= 2.002].reset_index(drop=True)  # drop filter tail
    return df


@st.cache_data(show_spinner="Loading file…")
def load_txt(name: str, src) -> pd.DataFrame:
    return _read_df(src)


@st.cache_data(show_spinner=False)
def file_features(name: str, src):
    """Transient features of one capture (see echem_utils.cycle_features). Raw signal."""
    try:
        df = _read_df(src)
        return cycle_features(df["ch1"].values, df["ch2"].values)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def file_minima(name: str, src, cutoff: float = 500.0):
    """Per-cycle minima of all complete 10-pulse sequences in one capture
    (see echem_utils.cycle_minima). Returns the list of sequences or None."""
    try:
        df = _read_df(src)
        seqs, _ = cycle_minima(df["ch1"].values, df["ch2"].values, cutoff_hz=cutoff)
        return seqs
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def find_qualifying_window(name: str, src, min_cycles: int = 10,
                            spike_pct: int = 98, spike_dist: int = 4000,
                            after_ms: float = 0.0, dn_method: str = "Off",
                            dn_cutoff: float = 500.0, dn_sg_ms: float = 5.0,
                            dn_current: bool = False):
    """Return window around first anodic spike with >= min_cycles T_on before it.

    Detection always runs on the raw signal; denoising (if any) is applied to the
    whole file before slicing, so the window has no filter edge effects.
    """
    try:
        df = _read_df(src)

        # Guard: signal must have real oscillations (not a flat initialisation file).
        # Noise-relative, so it works for large (PPGa7865) and small (7917) signals.
        if is_flat(df["ch1"].values):
            return None

        # Detect anodic spikes
        for pct in [spike_pct, 95, 90]:
            thresh = np.percentile(df["ch1"], pct)
            peaks, _ = find_peaks(df["ch1"], height=thresh, distance=spike_dist)
            if len(peaks) >= 1:
                break
        if len(peaks) < 1:
            return None

        # Guard: spike must be a real anodic excursion above the on-time baseline.
        # Threshold scales with the T_on/T_off swing and the noise level
        # (≈ 0.2 V for PPGa7865, ≈ 0.02 V for 7917) — see echem_utils.min_spike_amp.
        baseline  = float(np.percentile(df["ch1"], 30))
        min_amp   = min_spike_amp(df["ch1"].values)
        real_peaks = [p for p in peaks if float(df["ch1"].iloc[p]) - baseline >= min_amp]
        if not real_peaks:
            return None
        peaks = np.array(real_peaks)

        # Detect cathodic dips (T_off minima) — loose distance to not miss real dips
        neg_ch1    = -df["ch1"].values
        dip_thresh = np.percentile(neg_ch1, 80)
        dips, _    = find_peaks(neg_ch1, height=dip_thresh, distance=350)

        # Estimate T_cycle using only spacings > 30 ms (excludes false-positive pairs)
        if len(dips) >= 2:
            spacings  = np.diff(df["time"].values[dips])
            valid_sp  = spacings[spacings > 0.030]
            t_cycle_s = float(np.median(valid_sp)) if len(valid_sp) > 0 else 0.050
        else:
            t_cycle_s = 0.050  # 50 ms default

        # First peak with >= min_cycles T_off dips in its left inter-spike zone.
        for ci, peak in enumerate(peaks):
            prev_peak   = int(peaks[ci - 1]) if ci > 0 else 0
            dips_before = dips[(dips > prev_peak) & (dips < peak)]

            _n      = len(dips_before)
            spike_t = df["time"].iloc[peak]

            # If the last dip is essentially AT the spike (< 5 ms gap), remove it.
            # The anodic spike IS the T_off→T_on event for that cycle; keeping it
            # would produce a T_on of near-zero width as the final visible cycle.
            if ci > 0 and _n > 0:
                if (spike_t - df["time"].values[dips_before[-1]]) < 0.005:
                    dips_before = dips_before[:-1]
                    _n = len(dips_before)

            _last_dip_t  = df["time"].values[dips_before[-1]] if _n > 0 else -999.0
            _prev_t      = df["time"].iloc[prev_peak] if ci > 0 else df["time"].iloc[0]
            # One missed dip is acceptable if the inter-spike zone spans ≥ min_cycles
            _zone_covers = (spike_t - _prev_t) >= min_cycles * t_cycle_s

            if _n >= min_cycles or (_n == min_cycles - 1 and _zone_covers):

                _bl40 = float(np.percentile(df["ch1"], 40))

                if _n == min_cycles - 1 and _zone_covers:
                    _d0    = int(dips_before[0])
                    _lo    = prev_peak + 1
                    _t_gap = df["time"].values[_d0] - (df["time"].iloc[prev_peak] if ci > 0 else df["time"].iloc[0])

                    if _t_gap < 1.5 * t_cycle_s:
                        # dips_before[0] is the T_off dip right after prev_peak's spike
                        # (last dip was removed because it was AT the spike).
                        # The 10th T_on (T_on_0) is BEFORE dips_before[0] — scan backward
                        # from dips_before[0] to find where T_on_0 starts.
                        _rev      = df["ch1"].values[_lo : _d0][::-1]
                        _in_plat  = np.where(_rev >= _bl40)[0]
                        if len(_in_plat) > 0:
                            _plat_exit = np.where(_rev[_in_plat[0]:] < _bl40)[0]
                            start_idx  = (_d0 - int(_in_plat[0]) - int(_plat_exit[0])
                                          if len(_plat_exit) > 0 else _lo)
                        else:
                            start_idx = _d0
                    else:
                        # First T_off dip was missed/undetected.
                        # Step back 1 T_cycle from dips_before[0] and scan forward
                        # to the T_on baseline.
                        t_d0  = df["time"].values[_d0]
                        cand  = int(np.searchsorted(df["time"].values, t_d0 - t_cycle_s))
                        cand  = max(_lo, cand)
                        scan  = df["ch1"].values[cand : _d0]
                        risen = np.where(scan >= _bl40)[0]
                        start_idx = (cand + int(risen[0])) if len(risen) > 0 else _d0

                else:
                    start_idx = int(dips_before[0])

                # End: spike peak → stop at first post-spike dip minimum (within after_ms window)
                _post_lo = int(peak) + 1
                _post_hi = min(len(df), int(peak) + max(1, int(round(after_ms * 10))) + 1)
                if after_ms > 0 and _post_lo < _post_hi:
                    _post_seg  = df["ch1"].values[_post_lo : _post_hi]
                    _post_mins, _ = find_peaks(-_post_seg, distance=10)
                    end_idx = (_post_lo + int(_post_mins[0])) if len(_post_mins) > 0 else (_post_hi - 1)
                else:
                    end_idx = int(peak)

                spike_time = df["time"].iloc[peak]
                df_win     = df.iloc[start_idx : end_idx + 1].copy()
                t_ms       = (df_win["time"] - spike_time) * 1000   # 0 at spike
                # τ of the last T_off dip before the spike (used for dip-alignment mode)
                _last_dip_t_ms = float(
                    (df["time"].values[int(dips_before[-1])] - spike_time) * 1000
                )
                _sl = slice(start_idx, end_idx + 1)
                return {
                    "t_ms":           t_ms.values,
                    "ch1":            denoise(df["ch1"].values, dn_method, dn_cutoff, dn_sg_ms)[_sl],
                    "ch2":            denoise(df["ch2"].values, dn_method if dn_current else "Off",
                                              dn_cutoff, dn_sg_ms)[_sl],
                    "ch1_raw":        df_win["ch1"].values,
                    "ch2_raw":        df_win["ch2"].values,
                    "n_cycles":       _n + (1 if (_n == min_cycles - 1 and _zone_covers) else 0),
                    "spike_t_ms":     0.0,
                    "last_dip_t_ms":  _last_dip_t_ms,
                    "file":           name,
                }
    except Exception:
        pass
    return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Data")

    data_source = st.radio("Source", ["Local folder", "Upload files"], horizontal=True)

    if data_source == "Local folder":
        folder_path = st.text_input(
            "Dataset folder path",
            value=r"D:\Personal\Sreya\dataset-22-09-2026",
            help="Absolute path to the folder containing `*_filter.txt` files.",
        )
        if not folder_path:
            st.info("Enter the path to your dataset folder.")
            st.stop()
        include_sub = st.checkbox(
            "Include subfolders", value=False,
            help="Also load `*_filter.txt` from subfolders (e.g. `7917 DATA`, `7918 DATA`). "
                 "Needed for multi-file averaging. Files with the same name are loaded once.",
        )
        _folder = Path(folder_path)
        if not _folder.is_dir():
            st.error(f"Folder not found: `{folder_path}`")
            st.stop()
        _paths = sorted((_folder.rglob if include_sub else _folder.glob)("*_filter.txt"))
        if not _paths:
            st.error("No `*_filter.txt` files found in that folder.")
            st.stop()
        _local_key = (folder_path, include_sub)
        if st.session_state.get("_upload_key") != _local_key:
            # Local mode stores paths only; files are read on demand (a full run is several GB).
            _srcs = {}
            for p in _paths:
                _srcs.setdefault(p.name, str(p))
            st.session_state["_file_bytes"] = _srcs
            st.session_state["_upload_key"] = _local_key

    else:
        uploaded_txts = st.file_uploader(
            "Upload filter files",
            type=["txt"],
            accept_multiple_files=True,
            help="Select all `*_filter.txt` files at once (Ctrl+A in your dataset folder).",
        )
        if not uploaded_txts:
            st.info(
                "**How to use**\n\n"
                "1. Click **Browse files**.\n"
                "2. Navigate to your dataset folder.\n"
                "3. Select all `*_filter.txt` files (Ctrl+A) and click Open.\n"
                "4. The viewer will load automatically."
            )
            st.stop()
        _upload_key = tuple(sorted((f.name, f.size) for f in uploaded_txts))
        if st.session_state.get("_upload_key") != _upload_key:
            st.session_state["_file_bytes"] = {f.name: f.read() for f in uploaded_txts}
            st.session_state["_upload_key"] = _upload_key

    _file_bytes = st.session_state["_file_bytes"]
    file_names = sorted(n for n in _file_bytes if n.endswith("_filter.txt"))
    if not file_names:
        st.error("No `*_filter.txt` files found. Check your selection.")
        st.stop()

    timestamps  = [_parse_ts(n) for n in file_names]

    # Two-step date → time picker
    unique_dates = sorted({ts.date() for ts in timestamps if ts})
    date_labels  = [str(d) for d in unique_dates]

    sel_date_str = st.selectbox("Date", date_labels, index=len(date_labels) - 1)
    sel_date     = unique_dates[date_labels.index(sel_date_str)]

    day_pairs   = [(n, ts) for n, ts in zip(file_names, timestamps)
                   if ts and ts.date() == sel_date]
    time_labels = [ts.strftime("%H:%M:%S") for _, ts in day_pairs]
    day_fnames  = [n for n, _ in day_pairs]

    sel_time      = st.selectbox(
        f"Time  ({len(day_fnames)} captures on this date)",
        time_labels, index=len(time_labels) - 1,
    )
    selected_file = day_fnames[time_labels.index(sel_time)]
    txt_path      = Path(selected_file)
    st.caption(f"📄 `{selected_file}`")

    sample = st.text_input("Sample name", value="7917+7918",
                           help="Used in plot titles and download file names.").strip() or "Sample"
    _fs = sample.replace(" ", "_")   # file-name-safe prefix

    st.divider()

    # Denoising (applies to both tabs)
    st.header("🧹 Denoise")
    dn_method = st.radio(
        "Method", DENOISE_METHODS, index=0,
        help="Removes broadband measurement noise for display/export. "
             "Spike detection always uses the raw signal. Both filters are zero-phase "
             "(no time shift). Raw data is always kept in the exports.",
    )
    dn_cutoff, dn_sg_ms = 500.0, 5.0
    if dn_method.startswith("Low-pass"):
        dn_cutoff = float(st.slider("Cut-off (Hz)", 100, 2000, 500, step=50,
                                    help="500 Hz: ~3× less noise, spike tip ~6 % lower. "
                                         "Lower = smoother but edges round off."))
    elif dn_method.startswith("Savitzky"):
        dn_sg_ms = float(st.slider("Window (ms)", 1.0, 20.0, 5.0, step=0.5,
                                   help="Polynomial order 3. Larger windows round the spike peak."))
    dn_current = dn_method != "Off" and st.checkbox(
        "Also denoise current", value=False,
        help="Off by default: the current is a sharp square pulse and its noise is no worse than in "
             "earlier samples. A 500 Hz low-pass rounds its edges (~1 ms) and adds slight ringing; "
             "if you enable this, prefer ≥ 1500 Hz.")
    show_raw = dn_method != "Off" and st.checkbox("Show raw signal underneath", value=True)

    st.divider()
    st.caption("_Controls below apply to the **Signal Viewer** tab._")

    # Spike detection
    st.header("🔍 Spike Detection")
    spike_pct  = st.slider("Percentile threshold", 85, 99, 98)
    spike_dist = st.slider("Min spike distance (samples)", 500, 9000, 4000, step=250,
                            help="10 kHz → 4000 samples = 400 ms")
    show_spike_markers = st.checkbox("Mark spike positions", value=True)

    st.divider()

    # Time window
    st.header("🪟 Time Window")
    window_mode = st.radio(
        "Mode",
        ["Full file (2 s)", "Between spikes", "Centered on spike", "Manual range"],
        index=2,
    )

    spike_start_n = 2; spike_end_n = 4
    center_spike_n = 1; n_cycles = 10
    x_start_ms = 0.0; x_end_ms = 2000.0

    if window_mode == "Between spikes":
        c1, c2 = st.columns(2)
        spike_start_n = c1.number_input("From spike #", min_value=1, max_value=10, value=2)
        spike_end_n   = c2.number_input("To spike #",   min_value=2, max_value=10, value=4)
    elif window_mode == "Centered on spike":
        center_spike_n = st.number_input("Center spike #", min_value=1, max_value=10, value=1)
        n_cycles       = st.number_input("T_on cycles each side", min_value=1, max_value=30, value=10)
    elif window_mode == "Manual range":
        x_start_ms = st.number_input("Start (ms)", value=0.0, step=50.0, format="%.1f")
        x_end_ms   = st.number_input("End (ms)",   value=2000.0, step=50.0, format="%.1f")

    st.divider()

    # Axis ranges
    st.header("📐 Axis Ranges")
    auto_axes = st.checkbox("Auto-scale Φ and I axes to the data", value=True,
                            help="Untick to use the fixed ranges below (defaults suit PPGa7865).")
    st.markdown("**Potential — left axis**")
    phi_min  = st.number_input("Φ min (V)",          value=-3.5, step=0.1, format="%.2f")
    phi_max  = st.number_input("Φ max (V)",          value=0.5,  step=0.1, format="%.2f")
    phi_tick = st.number_input("Φ tick spacing (V)", value=0.5,  step=0.1, min_value=0.05, format="%.2f")
    st.markdown("**Current — right axis**")
    curr_min  = st.number_input("I min (V)",          value=-0.003, step=0.001, format="%.4f")
    curr_max  = st.number_input("I max (V)",          value=0.010,  step=0.001, format="%.4f")
    curr_tick = st.number_input("I tick spacing (V)", value=0.002,  step=0.001, min_value=0.0001, format="%.4f")
    st.markdown("**Time — x axis**")
    x_tick_ms = st.number_input("Tick spacing (ms)", value=50, step=10, min_value=5)

    st.divider()

    # Appearance
    st.header("🎨 Appearance")
    plot_theme = st.radio("Plot theme", ["Dark", "Light"], index=0, horizontal=True)
    phi_color  = st.color_picker("Potential colour", "#4da6ff" if plot_theme == "Dark" else "#0000FF")
    curr_color = st.color_picker("Current colour",   "#ff4d4d" if plot_theme == "Dark" else "#FF0000")
    line_width = st.slider("Line width", 0.3, 3.0, 0.7, step=0.1)
    show_grid  = st.checkbox("Show grid", value=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_viewer, tab_stacked, tab_evol, tab_min = st.tabs(
    ["📈 Signal Viewer", "🔬 Stacked Evolution", "📉 Transient Evolution", "📊 Minimum vs Cycle"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal Viewer
# ══════════════════════════════════════════════════════════════════════════════
with tab_viewer:

    df = load_txt(selected_file, _file_bytes[selected_file]).copy()
    # Denoise the whole file before windowing (no filter edge effects in the window)
    df["ch1_d"] = denoise(df["ch1"].values, dn_method, dn_cutoff, dn_sg_ms)
    df["ch2_d"] = denoise(df["ch2"].values, dn_method if dn_current else "Off", dn_cutoff, dn_sg_ms)
    dn_text = denoise_label(dn_method, dn_cutoff, dn_sg_ms) + ("" if dn_current or dn_method == "Off"
                                                               else " (potential only)")

    # Spike detection (always on the raw signal)
    thresh = np.percentile(df["ch1"], spike_pct)
    peaks, _ = find_peaks(df["ch1"], height=thresh, distance=spike_dist)
    if len(peaks) < 2:
        for fp in [95, 90, 85]:
            thresh = np.percentile(df["ch1"], fp)
            peaks, _ = find_peaks(df["ch1"], height=thresh, distance=spike_dist)
            if len(peaks) >= 2:
                break

    # Apply window
    t_offset = 0.0

    if window_mode == "Between spikes" and len(peaks) >= 2:
        i0 = min(int(spike_start_n) - 1, len(peaks) - 1)
        i1 = min(int(spike_end_n)   - 1, len(peaks) - 1)
        i0, i1 = min(i0, i1 - 1), max(i0 + 1, i1)
        t_offset = df["time"].iloc[peaks[i0]]

        # Start: skip the spike tail — find first sample after peaks[i0] where
        # ch1 drops back to T_on baseline level (40th pct), not the spike high.
        _baseline_bs = float(np.percentile(df["ch1"], 40))
        _rec = df["ch1"].values[peaks[i0] + 1 : peaks[i1]]
        _hit = np.where(_rec <= _baseline_bs)[0]
        _start = (peaks[i0] + 1 + int(_hit[0])) if len(_hit) > 0 else peaks[i0]

        # End: find last T_off dip before peaks[i1], then scan forward to where
        # ch1 rises sharply above the T_on baseline (spike ascent). Stop just
        # before that point — keeps the full 10th T_on plateau, no spike shown.
        _neg        = -df["ch1"].values
        _dip_thr    = np.percentile(_neg, 80)
        _dips_bs, _ = find_peaks(_neg, height=_dip_thr, distance=350)
        _dips_before_i1 = _dips_bs[_dips_bs < peaks[i1]]
        if len(_dips_before_i1) > 0:
            _last_dip = int(_dips_before_i1[-1])
            # Clearly above the T_on plateau: 25 % of the spike height (≈ 0.3 V for PPGa7865)
            _spk_h = float(np.median(df["ch1"].values[peaks])) - _baseline_bs
            _rise_thresh = _baseline_bs + 0.25 * _spk_h
            _scan  = df["ch1"].values[_last_dip : peaks[i1]]
            _above = np.where(_scan > _rise_thresh)[0]
            _end   = (_last_dip + int(_above[0]) - 1) if len(_above) > 0 else int(peaks[i1]) - 1
        else:
            _end = int(peaks[i1]) - 1

        df_win = df.iloc[_start : _end + 1].copy()

    elif window_mode == "Centered on spike" and len(peaks) >= 1:
        neg_ch1    = -df["ch1"].values
        dip_thresh = np.percentile(neg_ch1, 80)
        dips, _    = find_peaks(neg_ch1, height=dip_thresh, distance=350)

        ci          = min(int(center_spike_n) - 1, len(peaks) - 1)
        center_peak = peaks[ci]
        prev_peak   = int(peaks[ci - 1]) if ci > 0 else 0
        next_peak   = int(peaks[ci + 1]) if ci < len(peaks) - 1 else len(df) - 1

        dips_left  = dips[(dips > prev_peak) & (dips < center_peak)]
        dips_right = dips[(dips > center_peak) & (dips < next_peak)]
        nc         = int(n_cycles)

        start_idx = int(dips_left[-nc])   if len(dips_left)  >= nc else (int(dips_left[0])   if len(dips_left)  > 0 else prev_peak + 1)
        end_idx   = int(dips_right[nc-1]) if len(dips_right) >= nc else (int(dips_right[-1]) if len(dips_right) > 0 else next_peak - 1)

        t_offset = df["time"].iloc[start_idx]
        df_win   = df.iloc[start_idx : end_idx + 1].copy()

        al, ar = len(dips_left), len(dips_right)
        if al < nc or ar < nc:
            st.warning(
                f"Spike #{int(center_spike_n)}: only **{al}** T_on cycles before and **{ar}** after "
                f"(requested {nc}). Try spike #2 or later for full coverage."
            )
    else:
        df_win = df.copy()

    t_ms = (df_win["time"] - t_offset) * 1000
    ch1  = df_win["ch1_d"].values          # == raw when denoising is off
    ch2  = df_win["ch2_d"].values
    ch1_raw = df_win["ch1"].values
    ch2_raw = df_win["ch2"].values

    if window_mode == "Manual range":
        mask = (t_ms >= x_start_ms) & (t_ms <= x_end_ms)
        t_ms = t_ms[mask]; ch1 = ch1[mask.values]; ch2 = ch2[mask.values]
        ch1_raw = ch1_raw[mask.values]; ch2_raw = ch2_raw[mask.values]

    if auto_axes and len(ch1):
        def _auto(lo, hi):
            span = (hi - lo) or abs(hi) or 1.0
            step = 10 ** np.floor(np.log10(span / 6))
            step *= next(m for m in (1, 2, 2.5, 5, 10) if span / (step * m) <= 8)
            return float(np.floor((lo - 0.05 * span) / step) * step), \
                   float(np.ceil((hi + 0.08 * span) / step) * step), float(step)
        _r1, _r2 = (ch1_raw, ch2_raw) if show_raw else (ch1, ch2)
        phi_min, phi_max, phi_tick = _auto(float(np.min(_r1)), float(np.max(_r1)))
        curr_min, curr_max, curr_tick = _auto(float(np.min(_r2)), float(np.max(_r2)))

    spike_t_ms = (df["time"].iloc[peaks] - t_offset) * 1000
    if window_mode == "Manual range":
        spike_t_ms = spike_t_ms[(spike_t_ms >= x_start_ms) & (spike_t_ms <= x_end_ms)]
    elif window_mode == "Between spikes":
        spike_t_ms = spike_t_ms[(spike_t_ms >= 0) & (spike_t_ms <= t_ms.max())]

    # Build figure
    fig = go.Figure()
    if show_raw:
        fig.add_trace(go.Scatter(x=t_ms, y=ch1_raw, name="Potential raw",
                                 line=dict(color=phi_color, width=line_width), opacity=0.25, yaxis="y1"))
        fig.add_trace(go.Scatter(x=t_ms, y=ch2_raw, name="Current raw",
                                 line=dict(color=curr_color, width=line_width), opacity=0.25, yaxis="y2"))
    _sfx = f" — {dn_text}" if dn_method != "Off" else ""
    fig.add_trace(go.Scatter(x=t_ms, y=ch1, name=f"Potential (Φ){_sfx}",
                             line=dict(color=phi_color, width=line_width), yaxis="y1"))
    fig.add_trace(go.Scatter(x=t_ms, y=ch2, name="Current (I)" + (_sfx if dn_current else ""),
                             line=dict(color=curr_color, width=line_width), yaxis="y2"))
    if show_spike_markers and len(spike_t_ms) > 0:
        fig.add_trace(go.Scatter(
            x=spike_t_ms, y=[phi_min + 0.96 * (phi_max - phi_min)] * len(spike_t_ms),
            mode="markers", marker=dict(symbol="triangle-down", size=9),
            name=f"Spikes (n={len(peaks)})", yaxis="y1",
        ))

    # Theme
    if plot_theme == "Dark":
        bg_plot, bg_paper = "#0f1117", "rgba(0,0,0,0)"
        ax_col, gc, lc    = "#d0d0d0", "rgba(255,255,255,0.1)", "#555555"
        leg_bg, leg_bc    = "rgba(15,17,23,0.85)", "rgba(255,255,255,0.15)"
        title_col, mkr_col = "#ffffff", "#e0e0e0"
    else:
        bg_plot, bg_paper = "white", "white"
        ax_col, gc, lc    = "#222222", "rgba(180,180,180,0.5)", "black"
        leg_bg, leg_bc    = "rgba(255,255,255,0.9)", "gray"
        title_col, mkr_col = "#000000", "black"

    if show_spike_markers and len(spike_t_ms) > 0:
        fig.data[-1].marker.color = mkr_col

    grid_cfg   = dict(showgrid=show_grid, gridwidth=0.5, gridcolor=gc)
    phi_ticks  = list(np.round(np.arange(phi_min,  phi_max  + phi_tick  * 0.5, phi_tick),  6))
    curr_ticks = list(np.round(np.arange(curr_min, curr_max + curr_tick * 0.5, curr_tick), 6))

    fig.update_layout(
        height=520, margin=dict(l=70, r=90, t=90, b=50),
        plot_bgcolor=bg_plot, paper_bgcolor=bg_paper, hovermode="x unified",
        legend=dict(orientation="v", x=1.08, y=1, bgcolor=leg_bg,
                    bordercolor=leg_bc, borderwidth=1, font=dict(size=12, color=ax_col)),
        xaxis=dict(title=dict(text="τ in ms", font=dict(size=14, color=ax_col)),
                   side="top", dtick=x_tick_ms, tickangle=90,
                   tickfont=dict(size=10, color=ax_col),
                   showline=True, linecolor=lc, mirror=True, **grid_cfg),
        yaxis=dict(title=dict(text="Φ in V", font=dict(size=14, color=ax_col), standoff=10),
                   range=[phi_min, phi_max], tickvals=phi_ticks,
                   tickfont=dict(size=11, color=ax_col),
                   showline=True, linecolor=lc, mirror=False, zeroline=False, **grid_cfg),
        yaxis2=dict(title=dict(text="Current (V across shunt)", font=dict(size=14, color=ax_col), standoff=10),
                    range=[curr_min, curr_max], tickvals=curr_ticks,
                    tickfont=dict(size=11, color=ax_col),
                    showline=True, linecolor=lc, overlaying="y", side="right",
                    showgrid=False, zeroline=False),
        title=dict(text=f"<b>{sample}</b>  ·  {txt_path.name}"
                        + (f"  ·  <i>{dn_text}</i>" if dn_method != "Off" else ""),
                   font=dict(size=13, color=title_col), x=0.0, xanchor="left"),
    )

    st.plotly_chart(fig, width='stretch')

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Spikes detected",  len(peaks))
    c2.metric("Window (ms)",      f"{float(t_ms.iloc[-1] - t_ms.iloc[0]):.0f}")
    c3.metric("Φ min / max (V)",  f"{ch1.min():.3f} / {ch1.max():.3f}")
    c4.metric("I min / max (V)",  f"{ch2.min():.5f} / {ch2.max():.5f}")
    c5.metric("Samples shown",    f"{len(t_ms):,}")

    st.divider()

    # Data range panel
    with st.expander("📊 Data range & values for current plot", expanded=False):
        df_display = pd.DataFrame({
            "Time (ms)":     t_ms.values if hasattr(t_ms, "values") else t_ms,
            "Potential (V)": ch1,
            "Current (V)":   ch2,
        }).reset_index(drop=True)
        if dn_method != "Off":
            st.caption(f"Potential/Current columns are denoised ({dn_text}); raw columns are added to the CSV.")

        st.markdown("#### Summary statistics")
        stats = df_display.agg(["min", "max", "mean", "std"]).T
        stats.columns = ["Min", "Max", "Mean", "Std dev"]
        st.dataframe(stats.round(6), width='stretch')

        st.markdown("#### Filter & inspect values")
        cf1, cf2, cf3 = st.columns(3)
        t_all = df_display["Time (ms)"]
        p_all = df_display["Potential (V)"]
        i_all = df_display["Current (V)"]
        with cf1:
            t_range = st.slider("Time range (ms)", float(t_all.min()), float(t_all.max()),
                                (float(t_all.min()), float(t_all.max())), format="%.1f")
        with cf2:
            p_range = st.slider("Potential range (V)", float(p_all.min()), float(p_all.max()),
                                (float(p_all.min()), float(p_all.max())), format="%.4f")
        with cf3:
            i_range = st.slider("Current range (V)", float(i_all.min()), float(i_all.max()),
                                (float(i_all.min()), float(i_all.max())), format="%.6f")

        mask_f = (
            (df_display["Time (ms)"]     .between(*t_range)) &
            (df_display["Potential (V)"] .between(*p_range)) &
            (df_display["Current (V)"]   .between(*i_range))
        )
        df_filt = df_display[mask_f].reset_index(drop=True)
        st.caption(f"Showing **{len(df_filt):,}** of **{len(df_display):,}** samples")
        st.dataframe(
            df_filt.style.format({"Time (ms)": "{:.3f}", "Potential (V)": "{:.5f}", "Current (V)": "{:.6f}"}),
            height=280, width='stretch',
        )
        _csv = df_display.assign(**({"Potential raw (V)": ch1_raw, "Current raw (V)": ch2_raw}
                                    if dn_method != "Off" else {}))[mask_f.values]
        st.download_button("⬇️ Download filtered data as CSV",
                           _csv.to_csv(index=False).encode(),
                           f"{_fs}_{txt_path.stem}_filtered.csv", "text/csv")

    st.divider()

    # Download PNG
    def make_mpl_png() -> bytes:
        fig_dl, ax1 = plt.subplots(figsize=(16, 5))
        if show_raw:
            ax1.plot(t_ms, ch1_raw, color=phi_color, linewidth=0.4, alpha=0.25)
        ax1.plot(t_ms, ch1, color=phi_color, linewidth=0.5, label="Potential")
        ax1.set_ylabel("Φ in V", fontsize=11)
        ax1.set_ylim(phi_min, phi_max)
        ax1.tick_params(axis="x", which="both", rotation=90, labelsize=7,
                        bottom=False, labelbottom=False, top=True, labeltop=True)
        ax1.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_ms))
        ax1.xaxis.set_minor_locator(mticker.MultipleLocator(x_tick_ms / 2))
        ax1.yaxis.set_major_locator(mticker.MultipleLocator(phi_tick))
        if show_grid:
            ax1.grid(True, linewidth=0.4, alpha=0.5)
            ax1.grid(True, which="minor", linewidth=0.2, alpha=0.3)
        ax2 = ax1.twinx()
        if show_raw:
            ax2.plot(t_ms, ch2_raw, color=curr_color, linewidth=0.4, alpha=0.25)
        ax2.plot(t_ms, ch2, color=curr_color, linewidth=0.5, label="Current")
        ax2.set_ylabel("Current (V across shunt)", fontsize=11)
        ax2.set_ylim(curr_min, curr_max)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(curr_tick))
        ax1.set_xlabel("τ in ms", fontsize=11)
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=9)
        fig_dl.suptitle(f"{sample} · {txt_path.name}"
                        + (f" · {dn_text}" if dn_method != "Off" else ""), fontsize=11)
        plt.tight_layout()
        buf = io.BytesIO()
        fig_dl.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_dl)
        buf.seek(0)
        return buf.read()

    _dn_sfx = "" if dn_method == "Off" else "_denoised"
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns([1, 1, 1, 3])
    with col_dl1:
        st.download_button("⬇️ PNG", make_mpl_png(),
                           f"{_fs}_{txt_path.stem}{_dn_sfx}.png", "image/png",
                           width='stretch')
    with col_dl2:
        _sv_html = fig.to_html(include_plotlyjs="cdn").encode("utf-8")
        st.download_button("⬇️ HTML", _sv_html,
                           f"{_fs}_{txt_path.stem}{_dn_sfx}.html", "text/html",
                           width='stretch')
    with col_dl3:
        _t = np.asarray(t_ms, dtype=float)
        _xlsx = signal_excel(
            _t, ch1, ch2, title=f"{sample} · {txt_path.name}",
            phi_raw=ch1_raw if dn_method != "Off" else None,
            cur_raw=ch2_raw if dn_method != "Off" else None,
            note=f"Signal: {dn_text}. Window: {window_mode}. Source file: {txt_path.name}",
            phi_color=phi_color if plot_theme == "Light" else "#1F3864",
            cur_color=curr_color if plot_theme == "Light" else "#FF0000",
            phi_range=(phi_min, phi_max), cur_range=(curr_min, curr_max),
        )
        st.download_button("⬇️ Excel", _xlsx,
                           f"{_fs}_{txt_path.stem}{_dn_sfx}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           width='stretch')
    with col_dl4:
        st.caption("PNG: static matplotlib export · HTML: interactive Plotly chart · "
                   "Excel: data + editable Excel chart (raw columns included when denoised)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Transient Evolution  (placed before tab 2 in the code because tab 2
# may call st.stop(), which would otherwise leave this tab empty)
# ══════════════════════════════════════════════════════════════════════════════
with tab_evol:
    st.subheader("📉 Transient Evolution — how the pulse response changes during deposition")
    st.caption(
        "For every selected capture, all regular pulse cycles are averaged and a few numbers are "
        "measured (resting level in the pause, potential at the end of the deposition pulse, "
        "pulse depth, anodic-spike height, recovery time). Plotted against deposition time, they "
        "show *when* the electrode surface stops changing — the working hypothesis links this to "
        "coating coverage. Phases are found from the imposed current, so any potential scale works. "
        "Raw signal (sidebar Denoise is not applied; averaging cycles already removes most noise)."
    )
    _n_all = len(file_names)
    if _n_all < 30:
        st.info(f"Only **{_n_all}** files loaded. For a continuous curve, tick **Include subfolders** "
                "in the sidebar to load the full run (e.g. `7917 DATA` + `7918 DATA`).")

    ev1, ev2, ev3 = st.columns([1, 1, 2])
    with ev1:
        ev_stride = st.number_input("Use every Nth file", min_value=1, max_value=max(1, _n_all),
                                    value=max(1, int(np.ceil(_n_all / 300))),
                                    help="~10 s between captures. Every 20th file ≈ one point per ~3 min.")
    with ev2:
        ev_smooth = st.number_input("Rolling median (points)", min_value=1, max_value=51, value=1, step=2,
                                    help="Smooths the trend line; the individual points stay visible.")
    with ev3:
        _feat_default = ["pause_V", "pulse_end_V", "pulse_depth_mV", "spike_height_mV", "tau63_ms"]
        ev_feats = st.multiselect(
            "Quantities", list(FEATURE_INFO), default=_feat_default,
            format_func=lambda k: f"{FEATURE_INFO[k][0]} ({FEATURE_INFO[k][1]})")

    _ev_key = (st.session_state.get("_upload_key"), int(ev_stride))
    if st.button("📉 Compute transient evolution", type="primary"):
        _sel = file_names[::int(ev_stride)]
        if file_names[-1] not in _sel:
            _sel.append(file_names[-1])
        _t0 = timestamps[0]
        _rows, _prog = [], st.progress(0.0, text="Measuring captures…")
        for k, fn in enumerate(_sel):
            r = file_features(fn, _file_bytes[fn])
            ts_ = _parse_ts(fn)
            if r is not None and ts_ is not None:
                _rows.append({"file": fn, "time_h": (ts_ - _t0).total_seconds() / 3600, **r})
            if k % 5 == 0 or k == len(_sel) - 1:
                _prog.progress((k + 1) / len(_sel), text=f"Measuring captures… {k + 1}/{len(_sel)}")
        _prog.empty()
        st.session_state["_evol"] = (_ev_key, pd.DataFrame(_rows))

    _ev = st.session_state.get("_evol")
    if _ev is None or _ev[0] != _ev_key:
        st.info("Choose the settings above, then click **📉 Compute transient evolution**.")
    elif _ev[1].empty:
        st.error("No usable captures (all flat, or no pulse cycles found).")
    elif not ev_feats:
        st.warning("Select at least one quantity.")
    else:
        evdf = _ev[1].sort_values("time_h").reset_index(drop=True)
        n_f = len(ev_feats)
        figE = make_subplots(rows=n_f, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                             subplot_titles=[f"{FEATURE_INFO[k][0]} ({FEATURE_INFO[k][1]})" for k in ev_feats])
        summary = []
        for r_, k in enumerate(ev_feats, start=1):
            y = evdf[k]
            ys = y.rolling(int(ev_smooth), center=True, min_periods=1).median() if ev_smooth > 1 else y
            figE.add_trace(go.Scatter(x=evdf["time_h"], y=y, mode="markers",
                                      marker=dict(size=4, color="#1f3864", opacity=0.45),
                                      name=FEATURE_INFO[k][0], showlegend=False,
                                      customdata=evdf["file"],
                                      hovertemplate="%{x:.2f} h<br>%{y:.4f}<br>%{customdata}<extra></extra>"),
                           row=r_, col=1)
            if ev_smooth > 1:
                figE.add_trace(go.Scatter(x=evdf["time_h"], y=ys, mode="lines",
                                          line=dict(color="#c0392b", width=1.5), showlegend=False,
                                          hoverinfo="skip"), row=r_, col=1)
            figE.update_yaxes(title_text=FEATURE_INFO[k][1], row=r_, col=1)
            # Settling time: from when on the (smoothed) value stays inside a band around its
            # final level (median of the last 10 % of points). Band = max(10 % of the total
            # change, 3 × the point-to-point scatter), so noise alone does not count as drift.
            yv = ys.values; ok = ~np.isnan(yv)
            if ok.sum() >= 5:
                tt_, yy_ = evdf["time_h"].values[ok], yv[ok]
                fin = float(np.median(yy_[-max(3, len(yy_) // 10):]))
                ini = float(y.values[ok][0]); tot = ini - fin
                scatter = float(np.median(np.abs(np.diff(y.values[ok]))) / 0.954)
                band = max(0.1 * abs(tot), 3 * scatter)
                bad = np.where(np.abs(yy_ - fin) > band)[0]
                t_set = float(tt_[0]) if not len(bad) else (
                    float(tt_[bad[-1] + 1]) if bad[-1] + 1 < len(tt_) else np.nan)
                summary.append({
                    "Quantity": f"{FEATURE_INFO[k][0]} ({FEATURE_INFO[k][1]})",
                    "Start (first capture)": round(ini, 4), "End (last 10 %)": round(fin, 4),
                    "Total change": round(fin - ini, 4),
                    "Settled after (h)": ("n/a (< 20 points)" if ok.sum() < 20 else
                                          "not settled" if np.isnan(t_set) or t_set > 0.9 * tt_[-1]
                                          else f"{t_set:.2f}"),
                    "Band used (±)": round(band, 4),
                })
        figE.update_xaxes(title_text="Deposition time (h, from first loaded file)", row=n_f, col=1)
        figE.update_layout(height=230 * n_f + 80, margin=dict(l=70, r=30, t=60, b=50),
                           title=dict(text=f"<b>{sample}</b> — Transient Evolution "
                                           f"({len(evdf)} captures, every {int(ev_stride)}. file)",
                                      x=0, xanchor="left", font=dict(size=13)),
                           hovermode="closest", plot_bgcolor="white")
        figE.update_xaxes(showgrid=True, gridcolor="rgba(150,150,150,0.3)", showline=True, linecolor="black")
        figE.update_yaxes(showgrid=True, gridcolor="rgba(150,150,150,0.3)", showline=True, linecolor="black")
        st.plotly_chart(figE, width='stretch')

        if summary:
            st.markdown("#### Summary")
            st.dataframe(pd.DataFrame(summary), width='stretch', hide_index=True)
            st.caption("'Settled after' = time from which the (smoothed) value stays within ± band of its "
                       "final level; band = max(10 % of the total change, 3 × point-to-point scatter). "
                       "'not settled' = still drifting in the last 10 % of the run. A rough indicator — "
                       "always look at the plot. Use a rolling median ≥ 5 on a dense run.")

        _pp = evdf[["pulse_ms", "pause_ms"]].median()
        st.caption(f"Detected protocol: cathodic pulse ≈ **{_pp['pulse_ms']:.1f} ms**, pause ≈ "
                   f"**{_pp['pause_ms']:.1f} ms**; imposed current (shunt) cathodic "
                   f"{evdf['i_cath_V'].median()*1000:.3f} mV, anodic {evdf['i_anod_V'].median()*1000:.3f} mV "
                   f"(should be constant over the run).")

        with st.expander("ℹ️ What each quantity means"):
            for k in ev_feats:
                lbl, unit, expl = FEATURE_INFO[k]
                st.markdown(f"- **{lbl}** ({unit}): {expl}")

        _base = f"{_fs}_transient_evolution"
        d1, d2, d3, _ = st.columns([1, 1, 1, 3])
        d1.download_button("⬇ CSV", evdf.to_csv(index=False).encode(), f"{_base}.csv", "text/csv")
        d2.download_button("⬇ HTML", figE.to_html(include_plotlyjs="cdn").encode("utf-8"),
                           f"{_base}.html", "text/html")
        d3.download_button(
            "⬇ Excel",
            evolution_excel(evdf, ev_feats, f"{sample} — Transient Evolution",
                            note=f"{len(evdf)} captures (every {int(ev_stride)}. file). "
                                 "Each point = average of all regular pulse cycles in one capture. Raw signal."),
            f"{_base}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Minimum vs Cycle  (also placed before tab 2 because of its st.stop())
# ══════════════════════════════════════════════════════════════════════════════
with tab_min:
    st.subheader("📊 Minimum vs Cycle — lowest potential of each deposition pulse")
    st.caption(
        "For each time point, the 10 cathodic (deposition) pulses before an anodic spike are found from "
        "the imposed current (cycle 1 = first of the ten, cycle 10 = the one right before the spike). "
        "Each file usually holds 3 complete 10-pulse sequences; they can be averaged (mean ± SD) to "
        "separate real trends from noise. The minimum is located on a zero-phase low-passed potential; "
        "the raw minimum is biased low by the noise (≈ −2 to −3 mV). **Cycle 1 follows the previous "
        "anodic pulse directly (no pause)**, so it is systematically less negative — hidden by default."
    )
    mc1, mc2, mc3 = st.columns([2.2, 1.4, 1.4])
    with mc1:
        mc_tp = st.text_input("⏱ Time points — minutes after the first usable file (comma-separated)",
                              value="1, 10, 60, 300, 600", key="mc_tp",
                              help="'Earliest' and 'Last' are added automatically, as in the Stacked tab.")
    with mc2:
        mc_value = st.selectbox("Value per cycle",
                                ["Minimum (low-pass filtered)", "Minimum (raw)", "Φ at end of pulse"],
                                key="mc_value")
    with mc3:
        mc_seq = st.selectbox("Pulse sequence", ["Mean of all complete sequences",
                                                 "Sequence 1", "Sequence 2", "Sequence 3"], key="mc_seq",
                              help="Sequence 1 = the one before spike #2 (same as the stacked plot for 7917).")
    mc_orient = st.radio("Chart orientation", ["Cycle on x", "Potential on x (as in 'Sheet1')"],
                         index=0, horizontal=True, key="mc_orient")
    mc4, mc5, mc6 = st.columns([1, 1, 2])
    with mc4:
        mc_c1 = st.checkbox("Include cycle 1", value=False, key="mc_c1")
    with mc5:
        mc_err = st.checkbox("Error bars (± SD)", value=True, key="mc_err",
                             disabled=not mc_seq.startswith("Mean"))
    with mc6:
        mc_cut = float(st.slider("Low-pass cut-off for locating the minimum (Hz)", 200, 2000, 500, step=50,
                                 key="mc_cut"))
    _mc_key = (st.session_state.get("_upload_key"), mc_tp, mc_cut)

    if st.button("📊 Compute minimum vs cycle", type="primary", key="mc_btn"):
        try:
            _tps = [float(v.strip()) for v in mc_tp.split(",") if v.strip()]
        except ValueError:
            st.error("Enter comma-separated numbers (minutes).")
            _tps = None
        if _tps is not None:
            _valid = [(f, ts) for f, ts in zip(file_names, timestamps) if ts is not None]
            _res = {}
            with st.spinner("Finding pulse sequences…"):
                first = next(((f, ts) for f, ts in _valid if file_minima(f, _file_bytes[f], mc_cut)), None)
                last = next(((f, ts) for f, ts in reversed(_valid) if file_minima(f, _file_bytes[f], mc_cut)), None)
                picks = []
                if first:
                    picks.append(("Earliest", first[0]))
                    for tp in _tps:
                        tgt = first[1] + timedelta(minutes=tp)
                        if last and tgt > last[1] + timedelta(minutes=1):
                            continue
                        k0 = min(range(len(_valid)), key=lambda k: abs((_valid[k][1] - tgt).total_seconds()))
                        cands = [(abs((_valid[k][1] - tgt).total_seconds()), _valid[k][0])
                                 for k in range(max(0, k0 - 10), min(len(_valid), k0 + 11))
                                 if file_minima(_valid[k][0], _file_bytes[_valid[k][0]], mc_cut)]
                        if cands:
                            lbl = (f"{tp:.0f} min" if tp < 60 else f"{tp/60:.0f} h" if tp % 60 == 0
                                   else f"{tp/60:.1f} h")
                            picks.append((lbl, min(cands)[1]))
                    if last:
                        picks.append(("Last", last[0]))
                seen = set()
                for lbl, f in picks:
                    if f in seen:
                        continue
                    seen.add(f)
                    ts_ = _parse_ts(f)
                    _res[lbl] = {"file": f, "seqs": file_minima(f, _file_bytes[f], mc_cut),
                                 "elapsed_h": (ts_ - first[1]).total_seconds() / 3600 if ts_ else np.nan}
            st.session_state["_mc"] = (_mc_key, _res)

    _mc = st.session_state.get("_mc")
    if _mc is None or _mc[0] != _mc_key:
        st.info("Set the time points, then click **📊 Compute minimum vs cycle**. "
                "Works on the 7 hand-picked files; tick *Include subfolders* for the full run.")
    elif not _mc[1]:
        st.error("No file with a complete 10-pulse sequence was found.")
    else:
        _key = {"Minimum (low-pass filtered)": "min_V", "Minimum (raw)": "min_raw_V",
                "Φ at end of pulse": "end_V"}[mc_value]
        use_mean = mc_seq.startswith("Mean")
        stats, files_used, rows_tab = {}, {}, []
        for lbl, r in _mc[1].items():
            arr = np.array([[c[_key] for c in s] for s in r["seqs"]], dtype=float)
            if use_mean:
                m = arr.mean(0)
                sd = arr.std(0, ddof=1) if len(arr) > 1 else np.full(arr.shape[1], np.nan)
            else:
                k = min(int(mc_seq.split()[-1]) - 1, len(arr) - 1)
                m, sd = arr[k], np.full(arr.shape[1], np.nan)
            stats[lbl] = (m, sd, len(arr))
            files_used[lbl] = r["file"]
            rows_tab.append({"Label": lbl, "File": r["file"], "Time (h)": round(r["elapsed_h"], 2),
                             "Sequences in file": len(arr),
                             "Mean cycles 2–10 (V)": round(float(np.mean(m[1:])), 5),
                             "Spread between sequences, SD (mV)": (round(float(np.nanmean(arr[:, 1:].std(0, ddof=1))) * 1000, 2)
                                                                   if len(arr) > 1 else np.nan)})
        c0 = 0 if mc_c1 else 1
        cyc = np.arange(1, 11)
        shades = np.linspace(0.95, 0.35, len(stats))
        cols = [to_hex(cm.Blues(v)) for v in shades]
        pot_x = mc_orient.startswith("Potential")
        figM = go.Figure()
        for (lbl, (m, sd, n)), col in zip(stats.items(), cols):
            err = (dict(type="data", array=sd[c0:], visible=True, thickness=1.2, width=4, color=col)
                   if (use_mean and mc_err) else None)
            figM.add_trace(go.Scatter(
                x=m[c0:] if pot_x else cyc[c0:], y=cyc[c0:] if pot_x else m[c0:],
                mode="lines+markers", name=lbl,
                line=dict(color=col, width=2), marker=dict(size=8, color=col, line=dict(color="white", width=1)),
                error_x=err if pot_x else None, error_y=None if pot_x else err,
                hovertemplate=(f"<b>{lbl}</b><br>cycle %{{y}}<br>%{{x:.4f}} V<extra></extra>" if pot_x else
                               f"<b>{lbl}</b><br>cycle %{{x}}<br>%{{y:.4f}} V<extra></extra>")))
        _vl = {"min_V": "Minimum Φ in V", "min_raw_V": "Minimum Φ (raw) in V", "end_V": "Φ at end of pulse in V"}[_key]
        _ax_cyc = dict(title="Cycle (1 = first of the ten, 10 = last before the anodic spike)",
                       tickmode="linear", dtick=1, range=[c0 + 0.5, 10.5], showline=True, linecolor="black",
                       gridcolor="rgba(150,150,150,0.3)")
        _ax_pot = dict(title=_vl, tickformat=".3f", showline=True, linecolor="black",
                       gridcolor="rgba(150,150,150,0.3)")
        figM.update_layout(
            height=520, margin=dict(l=70, r=30, t=70, b=50), plot_bgcolor="white", hovermode="closest",
            title=dict(text=f"<b>{sample}</b> — {mc_value.lower()} per cycle · "
                            f"{'mean ± SD of all complete sequences' if use_mean else mc_seq.lower()}",
                       x=0, xanchor="left", font=dict(size=13)),
            xaxis=_ax_pot if pot_x else _ax_cyc, yaxis=_ax_cyc if pot_x else _ax_pot,
            legend=dict(title="time point"))
        st.plotly_chart(figM, width='stretch')
        _span = np.ptp(np.concatenate([s[0][c0:] for s in stats.values()])) * 1000
        st.caption(f"The y-axis spans about **{_span:.0f} mV**. For small-amplitude samples (e.g. 7917) the "
                   "cycle-to-cycle zig-zag is at the noise level — compare it with the error bars.")

        st.markdown("#### Files and noise")
        st.dataframe(pd.DataFrame(rows_tab), width='stretch', hide_index=True)
        tab_df = pd.DataFrame({"Cycle": cyc, **{l: s[0] for l, s in stats.items()}})
        with st.expander("Values per cycle (V)"):
            st.dataframe(tab_df.round(5), width='stretch', hide_index=True)

        _mbase = f"{_fs}_minimum_vs_cycle" + ("" if use_mean else f"_seq{mc_seq.split()[-1]}")
        m1, m2, m3, _ = st.columns([1, 1, 1, 3])
        m1.download_button("⬇ CSV", tab_df.assign(**{f"{l} SD": s[1] for l, s in stats.items()})
                           .to_csv(index=False).encode(), f"{_mbase}.csv", "text/csv", key="mc_csv")
        m2.download_button("⬇ HTML", figM.to_html(include_plotlyjs="cdn").encode("utf-8"),
                           f"{_mbase}.html", "text/html", key="mc_html")
        m3.download_button(
            "⬇ Excel",
            minima_meansd_excel(stats, f"{sample} — {mc_value}", _vl.replace(" in V", " (V)"),
                                note=(f"{mc_value}; {'mean ± SD over all complete 10-pulse sequences' if use_mean else mc_seq}. "
                                      f"Low-pass {mc_cut:.0f} Hz used to locate the minimum. Cycle 1 follows the "
                                      "previous anodic pulse directly. Exported from the Minimum vs Cycle tab."),
                                files=files_used,
                                spread_label="mean ± SD" if use_mean else mc_seq.lower()),
            f"{_mbase}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="mc_xlsx")
        st.caption("Excel: 'Chart' = cycle on x (cycles 2–10 with ± SD, and all cycles); 'Sheet1' = the original "
                   "paired Potential | Cycle layout with two charts in its orientation (potential on x, straight "
                   "lines, ± SD error bars); 'Sheet1 (2)' = Cycle | one column per time point; 'Mean and SD' = values.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Stacked Evolution
# ══════════════════════════════════════════════════════════════════════════════
with tab_stacked:
    st.subheader("🔬 Stacked Evolution — Signal Progression Over Time")

    exp_start = next((ts for ts in timestamps if ts is not None), None)
    if exp_start is None:
        st.error("Cannot determine experiment start time from filenames.")
        st.stop()

    exp_end = next((ts for ts in reversed(timestamps) if ts is not None), None)
    total_h  = (exp_end - exp_start).total_seconds() / 3600 if exp_end else 0

    st.caption(
        f"Experiment: **{exp_start.strftime('%Y-%m-%d %H:%M')}** → "
        f"**{exp_end.strftime('%Y-%m-%d %H:%M')}** "
        f"({total_h:.1f} h total, {len(file_names)} files)"
    )

    st.markdown("---")

    # ── Controls ──────────────────────────────────────────────────────────────
    col_a, col_b, col_b2 = st.columns([3, 1, 1])
    with col_a:
        tp_input = st.text_input(
            "⏱ Time points — minutes from first qualifying file (comma-separated)",
            value="1, 10, 60, 300, 600",
            help=f"Max useful: {total_h*60:.0f} min ({total_h:.1f} h). "
                 f"Experiment start: {exp_start.strftime('%H:%M:%S')}. "
                 f"Reference (t=0) is the first qualifying file's timestamp.",
        )
    with col_b:
        min_cyc = st.number_input("Min T_on cycles", min_value=3, max_value=20, value=10,
                                   help="Minimum T_on cycles required before the spike")
    with col_b2:
        post_spike_ms = st.number_input("Post-spike tail (ms)", min_value=0.0, max_value=200.0,
                                         value=30.0, step=5.0,
                                         help="Milliseconds of post-spike descent to include on the right")

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        stack_ch = st.radio("Channel to stack", ["Potential (Φ)", "Current (I)", "Both"],
                             index=0, horizontal=True)
    with col_d:
        stack_theme = st.radio("Theme", ["Dark", "Light"], index=1, horizontal=True, key="st_theme")
    with col_e:
        stack_lw = st.slider("Line width", 0.3, 2.5, 0.9, step=0.1, key="st_lw")

    align_mode = st.radio(
        "Align on",
        ["Spike (τ = 0 at spike)", "Last T_off dip (τ = 0 at last dip)"],
        index=0, horizontal=True, key="st_align",
        help="'Spike' centres all curves on the anodic spike. "
             "'Last T_off dip' shifts each curve so its last T_off dip before the spike lands at τ = 0, "
             "perfectly aligning the T_on/T_off features across curves (the spike then appears at a small positive τ).",
    )

    if stack_ch in ("Current (I)", "Both"):
        _tp_labels = ["Earliest", "1 min", "10 min", "1 h", "5 h", "10 h", "Last"]
        _default   = "5 h" if "5 h" in _tp_labels else _tp_labels[-1]
        curr_rep_label = st.selectbox(
            "Representative current curve",
            _tp_labels,
            index=_tp_labels.index(_default),
            help="Current is nearly identical across time points — one representative curve is plotted.",
        )

    n_avg = st.number_input(
        "Average N consecutive files per time point", min_value=1, max_value=20, value=1,
        help="Coherent averaging: the chosen file's window is averaged with the next N−1 "
             "qualifying files (previous files for 'Last'), aligned on the spike. "
             "~10 s between files → N=5 spans ~50 s. Reduces noise by ≈√N without "
             "distorting edges. Needs the full run: tick 'Include subfolders' in the sidebar. "
             "Combines with the sidebar Denoise filter.",
    )
    st.caption(f"Denoise: **{denoise_label(dn_method, dn_cutoff, dn_sg_ms, int(n_avg))}** "
               "(filter set in the sidebar)")

    with st.expander("📐 Y-axis limits for stacked plot"):
        s_auto = st.checkbox("Auto-scale to the data", value=True, key="s_auto",
                             help="Untick to use the fixed limits below (defaults suit PPGa7865).")
        lc1, lc2, lc3, lc4 = st.columns(4)
        s_phi_min  = lc1.number_input("Φ min (V)",  value=-4.5,  step=0.1,   format="%.2f",  key="s_phi_min")
        s_phi_max  = lc2.number_input("Φ max (V)",  value=-0.2,  step=0.1,   format="%.2f",  key="s_phi_max")
        s_curr_min = lc3.number_input("I min (V)",  value=-0.003,step=0.001, format="%.4f",  key="s_curr_min")
        s_curr_max = lc4.number_input("I max (V)",  value=0.010, step=0.001, format="%.4f",  key="s_curr_max")

    run_stack = st.button("🚀 Generate Stacked Plot", type="primary", width='content')

    if run_stack:

        # Parse time points
        try:
            raw_tps = [float(x.strip()) for x in tp_input.split(",") if x.strip()]
        except ValueError:
            st.error("Enter comma-separated numbers (minutes).")
            st.stop()

        def _nearest_file(target_ts):
            valid = [(f, ts) for f, ts in zip(file_names, timestamps) if ts is not None]
            return min(valid, key=lambda x: abs((x[1] - target_ts).total_seconds()))

        def _qual(fname):
            return find_qualifying_window(fname, _file_bytes[fname], min_cyc, after_ms=post_spike_ms,
                                          dn_method=dn_method, dn_cutoff=dn_cutoff, dn_sg_ms=dn_sg_ms,
                                          dn_current=dn_current)

        def _averaged(res, fname, step=+1):
            """Average res with the next (step=+1) or previous (step=-1) qualifying files."""
            res = {**res, "n_files": 1, "span_s": 0.0}
            if n_avg <= 1:
                return res
            wins, used = [res], [fname]
            k = file_names.index(fname) + step
            while len(wins) < n_avg and 0 <= k < len(file_names) and abs(k - file_names.index(fname)) <= 3 * n_avg:
                r = _qual(file_names[k])
                if r is not None:
                    wins.append(r); used.append(file_names[k])
                k += step
            avg = average_windows(wins)
            ts_used = [_parse_ts(u) for u in used]
            return {**res, **avg,
                    "last_dip_t_ms": float(np.mean([w["last_dip_t_ms"] for w in wins])),
                    "n_cycles": min(w["n_cycles"] for w in wins),
                    "n_files": len(wins),
                    "span_s": abs((max(ts_used) - min(ts_used)).total_seconds())}

        entries    = []
        info_rows  = []
        missing    = []

        with st.spinner("Scanning files for qualifying windows…"):

            # ── "Earliest" ── scan from first file forward
            earliest_ts = None
            for fname in file_names:
                res = _qual(fname)
                if res is not None:
                    fts = _parse_ts(fname)
                    earliest_ts = fts  # reference timestamp for subsequent time points
                    elapsed_min = (fts - exp_start).total_seconds() / 60 if fts else 0
                    entries.append({**_averaged(res, fname, +1), "label": "Earliest", "elapsed_min": elapsed_min})
                    break
            if not entries:
                missing.append("Earliest")

            # ── "Last" ── scan from last file backward
            for fname in reversed(file_names):
                res = _qual(fname)
                if res is not None:
                    fts = _parse_ts(fname)
                    elapsed_min = (fts - exp_start).total_seconds() / 60 if fts else total_h * 60
                    entries.append({**_averaged(res, fname, -1), "label": "Last", "elapsed_min": elapsed_min})
                    break
            else:
                missing.append("Last")

            # ── Each requested time point ──
            ref_ts = earliest_ts if earliest_ts is not None else exp_start
            for tp_min in raw_tps:
                if tp_min > total_h * 60:
                    missing.append(
                        f"{tp_min:.0f} min ({tp_min/60:.1f} h) — beyond experiment duration"
                    )
                    continue

                target_ts = ref_ts + timedelta(minutes=tp_min)
                best_fname, _ = _nearest_file(target_ts)
                best_idx = file_names.index(best_fname)

                # Scan ±10 files around the nearest; pick the qualifying file
                # closest in time to the target
                scan_lo = max(0, best_idx - 10)
                scan_hi = min(best_idx + 11, len(file_names))
                candidates = []
                for fi in range(scan_lo, scan_hi):
                    r = _qual(file_names[fi])
                    if r is not None:
                        fts_c = _parse_ts(file_names[fi])
                        dist  = abs((fts_c - target_ts).total_seconds()) if fts_c else float("inf")
                        candidates.append((dist, file_names[fi], r))

                if not candidates:
                    missing.append(
                        f"{tp_min:.0f} min — no qualifying spike found near "
                        f"`{best_fname}`"
                    )
                    continue

                candidates.sort(key=lambda x: x[0])
                _, used_fname, res = candidates[0]

                fts = _parse_ts(used_fname)
                elapsed_min = (fts - exp_start).total_seconds() / 60 if fts else tp_min
                label = (f"{tp_min:.0f} min" if tp_min < 60
                         else f"{tp_min/60:.0f} h" if tp_min % 60 == 0
                         else f"{tp_min/60:.1f} h")
                entries.append({**_averaged(res, used_fname, +1), "label": label, "elapsed_min": elapsed_min})

        if earliest_ts is not None:
            st.caption(
                f"t = 0 reference: **{earliest_ts.strftime('%H:%M:%S')}** "
                f"(first qualifying file, {(earliest_ts - exp_start).total_seconds()/60:.1f} min after experiment start)"
            )

        if missing:
            st.warning("⚠️ Could not find data for:\n\n" + "\n".join(f"- {m}" for m in missing))

        if not entries:
            st.error("No valid entries found. Reduce 'Min T_on cycles' or adjust time points.")
            st.stop()

        # Sort by time so "Last" lands after all mid-points
        entries.sort(key=lambda e: e["elapsed_min"])

        # Deduplicate: if two labels landed on the exact same file, keep the earlier label
        seen_files = {}
        deduped    = []
        merged_log = []   # human-readable list of what was merged
        for e in entries:
            if e["file"] not in seen_files:
                seen_files[e["file"]] = e["label"]
                deduped.append(e)
            else:
                merged_log.append(
                    f"**{e['label']}** → same file as **{seen_files[e['file']]}** "
                    f"(`{e['file']}`)"
                )
        entries = deduped
        if merged_log:
            st.info(
                "ℹ️ The following time points resolved to the same file as an earlier entry "
                "and were removed to avoid duplicate traces:\n\n"
                + "\n".join(f"- {m}" for m in merged_log)
            )

        # ── Color palette: dark → light ───────────────────────────────────────
        n = len(entries)
        shade_vals = np.linspace(0.90, 0.30, n)   # dark to light

        if stack_theme == "Dark":
            phi_cols  = [to_hex(cm.Blues(v)) for v in shade_vals]
            curr_cols = [to_hex(cm.Reds(v))  for v in shade_vals]
            bg_s, paper_s = "#0f1117", "rgba(0,0,0,0)"
            ax_s, gc_s, lc_s = "#d0d0d0", "rgba(255,255,255,0.10)", "#555555"
        else:
            phi_cols  = [to_hex(cm.Blues(v)) for v in shade_vals]
            curr_cols = [to_hex(cm.Reds(v))  for v in shade_vals]
            bg_s, paper_s = "white", "white"
            ax_s, gc_s, lc_s = "#222222", "rgba(150,150,150,0.4)", "black"

        use_dual = (stack_ch == "Both")
        _st_dn = denoise_label(dn_method, dn_cutoff, dn_sg_ms, int(n_avg))
        if dn_method != "Off" and not dn_current:
            _st_dn += " (filter on potential only)"
        if n_avg > 1 and any(e.get("n_files", 1) < n_avg for e in entries):
            st.warning("Some time points had fewer qualifying neighbour files than requested "
                       "(see 'Files averaged' below). Tick 'Include subfolders' to load the full run.")

        # ── Build figure ──────────────────────────────────────────────────────
        fig_s = go.Figure()

        # ── Alignment: compute per-entry τ offset ────────────────────────────
        _use_dip_align = align_mode.startswith("Last T_off dip")
        if _use_dip_align and all("last_dip_t_ms" in e for e in entries):
            # Shift each curve by the difference between its last T_off dip τ and
            # the median last-dip τ across all curves.  This keeps the shifts small
            # (typically ±5 ms) and aligns all T_on/T_off features perfectly.
            _ref = float(np.median([e["last_dip_t_ms"] for e in entries]))
            _offsets = {e["label"]: _ref - e["last_dip_t_ms"] for e in entries}
        else:
            _offsets = {e["label"]: 0.0 for e in entries}

        # Pick the representative current entry (closest label match, fallback to last).
        _rep_entry = next((e for e in entries if e["label"] == curr_rep_label), entries[-1]) \
                     if stack_ch in ("Current (I)", "Both") else None
        _curr_color = "#c0392b" if stack_theme == "Light" else "#ff6b6b"

        _xl_curves = []   # exactly what is plotted, for the Excel export
        for i, e in enumerate(entries):
            _off = _offsets[e["label"]]
            # Trim spike-recovery tail from left edge: drop samples above T_on level
            _t_on_lvl = float(np.median(e["ch1"]))
            _valid     = np.where(e["ch1"] <= _t_on_lvl)[0]
            _s         = int(_valid[0]) if len(_valid) > 0 else 0
            t          = e["t_ms"][_s:] + _off
            lbl = e["label"]
            if stack_ch in ("Potential (Φ)", "Both"):
                _xl_curves.append({"label": lbl, "t_ms": t, "phi": e["ch1"][_s:], "color": phi_cols[i]})
                fig_s.add_trace(go.Scatter(
                    x=t, y=e["ch1"][_s:], name=f"Φ — {lbl}",
                    line=dict(color=phi_cols[i], width=stack_lw),
                    yaxis="y", legendgroup=lbl,
                    hovertemplate=f"<b>{lbl}</b><br>τ=%{{x:.1f}} ms<br>Φ=%{{y:.4f}} V<extra></extra>",
                ))

        # Current: single representative trace only (no left-edge trimming — ch2 has no spike artifact)
        if stack_ch in ("Current (I)", "Both") and _rep_entry is not None:
            _roff = _offsets[_rep_entry["label"]]
            fig_s.add_trace(go.Scatter(
                x=_rep_entry["t_ms"] + _roff, y=_rep_entry["ch2"],
                name=f"I — {_rep_entry['label']} (representative)",
                line=dict(color=_curr_color, width=stack_lw),
                yaxis="y2" if use_dual else "y",
                hovertemplate=f"<b>I ({_rep_entry['label']})</b><br>τ=%{{x:.1f}} ms<br>I=%{{y:.5f}} V<extra></extra>",
            ))

        if s_auto:
            def _pad(lo, hi, top=0.08):
                span = (hi - lo) or 1e-6
                return [lo - 0.05 * span, hi + top * span]
            _phi_all = np.concatenate([e["ch1"] for e in entries])
            s_phi_min, s_phi_max = _pad(float(_phi_all.min()), float(_phi_all.max()))
            if _rep_entry is not None:
                s_curr_min, s_curr_max = _pad(float(_rep_entry["ch2"].min()), float(_rep_entry["ch2"].max()))
        yax1_range = ([s_phi_min, s_phi_max] if stack_ch != "Current (I)"
                      else [s_curr_min, s_curr_max])
        yax1_title = "Φ in V" if stack_ch != "Current (I)" else "Current (V)"

        layout_kw = dict(
            height=560, margin=dict(l=70, r=90, t=90, b=50),
            plot_bgcolor=bg_s, paper_bgcolor=paper_s, hovermode="x unified",
            legend=dict(x=1.08, y=1, font=dict(size=11, color=ax_s),
                        bgcolor="rgba(0,0,0,0.4)" if stack_theme == "Dark" else "rgba(255,255,255,0.85)",
                        bordercolor=ax_s, borderwidth=1),
            title=dict(text=f"<b>{sample}</b> — Stacked Evolution"
                            + (f"  ·  <i>{_st_dn}</i>" if _st_dn != "raw" else ""),
                       font=dict(size=13, color=ax_s), x=0, xanchor="left"),
            xaxis=dict(
                title=dict(text="τ in ms  (0 = last T_off dip)" if _use_dip_align else "τ in ms  (0 = spike)",
                           font=dict(size=14, color=ax_s)),
                side="top", tickangle=90, tickfont=dict(size=9, color=ax_s),
                showgrid=True, gridwidth=0.5, gridcolor=gc_s,
                showline=True, linecolor=lc_s, mirror=True,
            ),
            yaxis=dict(
                title=dict(text=yax1_title, font=dict(size=14, color=ax_s)),
                range=yax1_range, tickfont=dict(size=10, color=ax_s),
                showgrid=True, gridwidth=0.5, gridcolor=gc_s,
                showline=True, linecolor=lc_s, zeroline=False,
            ),
        )
        if use_dual:
            layout_kw["yaxis2"] = dict(
                title=dict(text="Current (V)", font=dict(size=14, color=ax_s)),
                range=[s_curr_min, s_curr_max],
                tickfont=dict(size=10, color=ax_s),
                overlaying="y", side="right",
                showgrid=False, zeroline=False, showline=True, linecolor=lc_s,
            )

        fig_s.update_layout(**layout_kw)
        st.plotly_chart(fig_s, width='stretch')

        # ── Download buttons ──────────────────────────────────────────────────
        _st_base = f"{_fs}_stacked_evolution" + ("" if _st_dn == "raw" else "_denoised")
        dl_cols = st.columns([1, 1, 1, 3])
        # PNG via kaleido
        try:
            _png_bytes = fig_s.to_image(format="png", width=1600, height=900, scale=2)
            dl_cols[0].download_button(
                "⬇ PNG",
                data=_png_bytes,
                file_name=f"{_st_base}.png",
                mime="image/png",
            )
        except Exception:
            dl_cols[0].caption("PNG: install `kaleido`")
        # Interactive HTML (always works)
        _html_bytes = fig_s.to_html(include_plotlyjs="cdn").encode("utf-8")
        dl_cols[1].download_button(
            "⬇ HTML",
            data=_html_bytes,
            file_name=f"{_st_base}.html",
            mime="text/html",
        )
        _xl_slot = dl_cols[2]   # filled after the source table is built

        # ── Color legend chips ────────────────────────────────────────────────
        st.markdown("**Color legend** (dark = earliest → light = latest)")
        chips = "".join(
            f'<span style="display:inline-block;margin:2px 6px;padding:3px 10px;'
            f'border-radius:4px;background:{phi_cols[i]};color:#000;font-size:0.8rem;">'
            f'{e["label"]}</span>'
            for i, e in enumerate(entries)
        )
        st.markdown(chips, unsafe_allow_html=True)

        st.markdown("---")

        # ── Source files table ────────────────────────────────────────────────
        st.markdown("#### Files used for each time point")
        for e in entries:
            fts = _parse_ts(e["file"])
            elapsed = e["elapsed_min"]
            elapsed_str = (f"{elapsed:.1f} min" if elapsed < 60
                           else f"{elapsed/60:.2f} h")
            info_rows.append({
                "Label":          e["label"],
                "File":           e["file"],
                "Timestamp":      fts.strftime("%Y-%m-%d %H:%M:%S") if fts else "—",
                "Time (exp. start)": elapsed_str,
                "T_on cycles":    e.get("n_cycles", "—"),
                "Spike at (ms)":  f"{e.get('spike_t_ms', 0):.1f}",
                "Files averaged": e.get("n_files", 1),
                "Averaging span (s)": f"{e.get('span_s', 0.0):.0f}",
                "τ shift (ms)":   f"{_offsets[e['label']]:+.2f}",
            })

        st.dataframe(pd.DataFrame(info_rows), width='stretch', hide_index=True)

        _cur_xl = None
        if _rep_entry is not None:
            _cur_xl = {"label": _rep_entry["label"], "t_ms": _rep_entry["t_ms"] + _offsets[_rep_entry["label"]],
                       "cur": _rep_entry["ch2"], "color": _curr_color}
        _note = (f"Signal: {_st_dn}. Alignment: {align_mode}. Min T_on cycles: {int(min_cyc)}. "
                 f"Post-spike tail: {post_spike_ms:g} ms. All curves on one common 0.1 ms τ grid.")
        if _xl_curves or _cur_xl:
            _xl_slot.download_button(
                "⬇ Excel",
                data=stacked_excel(_xl_curves, info_rows, f"{sample} — Stacked Evolution", note=_note,
                                   current_curve=_cur_xl,
                                   phi_range=(s_phi_min, s_phi_max), cur_range=(s_curr_min, s_curr_max)),
                file_name=f"{_st_base}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    else:
        st.info(
            "Configure the time points above, then click **🚀 Generate Stacked Plot**.\n\n"
            "Default time points: **1 min · 10 min · 1 h · 5 h · 10 h**\n\n"
            "Each trace shows 10 T_on cycles before the first eligible anodic spike "
            "in the file nearest to that time point. Colors go **dark (earliest) → light (latest)**."
        )
