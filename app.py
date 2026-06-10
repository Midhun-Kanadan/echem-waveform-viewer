"""
PPGa7865 Electrochemical Signal Viewer
Interactive Streamlit app for Yokogawa DL850E transient recorder data.
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPGa7865 · Electrochemical Viewer",
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

st.title("⚡ PPGa7865 — Electrochemical Signal Viewer")
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


@st.cache_data(show_spinner="Loading file…")
def load_txt(name: str, data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data), sep="\t", header=None, names=["time", "ch1", "ch2"])
    df = df.iloc[10:].reset_index(drop=True)           # drop filter warm-up
    df = df[df["time"] <= 2.002].reset_index(drop=True) # drop filter tail
    return df


@st.cache_data(show_spinner=False)
def find_qualifying_window(name: str, data: bytes, min_cycles: int = 10,
                            spike_pct: int = 98, spike_dist: int = 4000):
    """Return window around first anodic spike with >= min_cycles T_on before it."""
    try:
        df = pd.read_csv(io.BytesIO(data), sep="\t", header=None, names=["time", "ch1", "ch2"])
        df = df.iloc[10:].reset_index(drop=True)
        df = df[df["time"] <= 2.002].reset_index(drop=True)

        # Guard: signal must have real oscillations (not a flat initialisation file)
        # Std of ch1 in the middle 80 % of the file must exceed 0.05 V
        mid = df["ch1"].iloc[int(len(df)*0.1) : int(len(df)*0.9)]
        if mid.std() < 0.05:
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
        # Threshold is dynamic: max(0.2 V, half the interquartile range of the signal)
        # so it adapts if spike amplitude changes over the course of the experiment.
        baseline  = float(np.percentile(df["ch1"], 30))
        iqr       = float(np.percentile(df["ch1"], 75)) - float(np.percentile(df["ch1"], 25))
        min_amp   = max(0.2, 0.5 * abs(iqr))
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
                    # CASE B: one dip missing (removed at-spike dip, or undetected
                    # first dip). Step back 1 T_cycle from the first detected dip
                    # then scan forward to the T_on baseline.
                    t_d0      = df["time"].values[int(dips_before[0])]
                    cand      = int(np.searchsorted(df["time"].values, t_d0 - t_cycle_s))
                    cand      = max(prev_peak + 1, cand)
                    scan      = df["ch1"].values[cand : int(dips_before[0])]
                    risen     = np.where(scan >= _bl40)[0]
                    start_idx = (cand + int(risen[0])) if len(risen) > 0 else int(dips_before[0])

                else:
                    start_idx = int(dips_before[0])

                # End: spike peak exactly — no post-spike data shown.
                end_idx = int(peak)

                spike_time = df["time"].iloc[peak]
                df_win     = df.iloc[start_idx : end_idx + 1].copy()
                t_ms       = (df_win["time"] - spike_time) * 1000   # 0 at spike
                return {
                    "t_ms":      t_ms.values,
                    "ch1":       df_win["ch1"].values,
                    "ch2":       df_win["ch2"].values,
                    "n_cycles":  _n + (1 if (_n == min_cycles - 1 and _zone_covers) else 0),
                    "spike_t_ms": 0.0,   # spike is always at x = 0
                    "file":      name,
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
            value=r"D:\Family\Sreya\dataset",
            help="Absolute path to the folder containing `*_filter.txt` files.",
        )
        if not folder_path:
            st.info("Enter the path to your dataset folder.")
            st.stop()
        _folder = Path(folder_path)
        if not _folder.is_dir():
            st.error(f"Folder not found: `{folder_path}`")
            st.stop()
        _paths = sorted(_folder.glob("*_filter.txt"))
        if not _paths:
            st.error("No `*_filter.txt` files found in that folder.")
            st.stop()
        _local_key = folder_path
        if st.session_state.get("_upload_key") != _local_key:
            st.session_state["_file_bytes"] = {p.name: p.read_bytes() for p in _paths}
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
tab_viewer, tab_stacked = st.tabs(["📈 Signal Viewer", "🔬 Stacked Evolution"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal Viewer
# ══════════════════════════════════════════════════════════════════════════════
with tab_viewer:

    df = load_txt(selected_file, _file_bytes[selected_file])

    # Spike detection
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
            _rise_thresh = _baseline_bs + 0.3   # clearly above T_on plateau
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
    ch1  = df_win["ch1"].values
    ch2  = df_win["ch2"].values

    if window_mode == "Manual range":
        mask = (t_ms >= x_start_ms) & (t_ms <= x_end_ms)
        t_ms = t_ms[mask]; ch1 = ch1[mask.values]; ch2 = ch2[mask.values]

    spike_t_ms = (df["time"].iloc[peaks] - t_offset) * 1000
    if window_mode == "Manual range":
        spike_t_ms = spike_t_ms[(spike_t_ms >= x_start_ms) & (spike_t_ms <= x_end_ms)]
    elif window_mode == "Between spikes":
        spike_t_ms = spike_t_ms[(spike_t_ms >= 0) & (spike_t_ms <= t_ms.max())]

    # Build figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_ms, y=ch1, name="Potential (Φ)",
                             line=dict(color=phi_color, width=line_width), yaxis="y1"))
    fig.add_trace(go.Scatter(x=t_ms, y=ch2, name="Current (I)",
                             line=dict(color=curr_color, width=line_width), yaxis="y2"))
    if show_spike_markers and len(spike_t_ms) > 0:
        fig.add_trace(go.Scatter(
            x=spike_t_ms, y=[phi_max * 0.92] * len(spike_t_ms),
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
        title=dict(text=f"<b>PPGa7865</b>  ·  {txt_path.name}",
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
        st.download_button("⬇️ Download filtered data as CSV",
                           df_filt.to_csv(index=False).encode(),
                           f"PPGa7865_{txt_path.stem}_filtered.csv", "text/csv")

    st.divider()

    # Download PNG
    def make_mpl_png() -> bytes:
        fig_dl, ax1 = plt.subplots(figsize=(16, 5))
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
        ax2.plot(t_ms, ch2, color=curr_color, linewidth=0.5, label="Current")
        ax2.set_ylabel("Current (V across shunt)", fontsize=11)
        ax2.set_ylim(curr_min, curr_max)
        ax2.yaxis.set_major_locator(mticker.MultipleLocator(curr_tick))
        ax1.set_xlabel("τ in ms", fontsize=11)
        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=9)
        fig_dl.suptitle(f"PPGa7865 · {txt_path.name}", fontsize=11)
        plt.tight_layout()
        buf = io.BytesIO()
        fig_dl.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig_dl)
        buf.seek(0)
        return buf.read()

    col_dl1, col_dl2 = st.columns([1, 5])
    with col_dl1:
        st.download_button("⬇️ Download PNG", make_mpl_png(),
                           f"PPGa7865_{txt_path.stem}.png", "image/png",
                           width='stretch')
    with col_dl2:
        st.caption("Exports a high-res matplotlib figure with current axis settings.")


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
    col_a, col_b = st.columns([3, 1])
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

    col_c, col_d, col_e = st.columns(3)
    with col_c:
        stack_ch = st.radio("Channel to stack", ["Potential (Φ)", "Current (I)", "Both"],
                             index=0, horizontal=True)
    with col_d:
        stack_theme = st.radio("Theme", ["Dark", "Light"], index=1, horizontal=True, key="st_theme")
    with col_e:
        stack_lw = st.slider("Line width", 0.3, 2.5, 0.9, step=0.1, key="st_lw")

    with st.expander("📐 Y-axis limits for stacked plot"):
        lc1, lc2, lc3, lc4 = st.columns(4)
        s_phi_min  = lc1.number_input("Φ min (V)",  value=-3.5,  step=0.1,   format="%.2f",  key="s_phi_min")
        s_phi_max  = lc2.number_input("Φ max (V)",  value=0.5,   step=0.1,   format="%.2f",  key="s_phi_max")
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

        entries    = []
        info_rows  = []
        missing    = []

        with st.spinner("Scanning files for qualifying windows…"):

            # ── "Earliest" ── scan from first file forward
            earliest_ts = None
            for fname in file_names:
                res = find_qualifying_window(fname, _file_bytes[fname], min_cyc)
                if res is not None:
                    fts = _parse_ts(fname)
                    earliest_ts = fts  # reference timestamp for subsequent time points
                    elapsed_min = (fts - exp_start).total_seconds() / 60 if fts else 0
                    entries.append({**res, "label": "Earliest", "elapsed_min": elapsed_min})
                    break
            if not entries:
                missing.append("Earliest")

            # ── "Last" ── scan from last file backward
            for fname in reversed(file_names):
                res = find_qualifying_window(fname, _file_bytes[fname], min_cyc)
                if res is not None:
                    fts = _parse_ts(fname)
                    elapsed_min = (fts - exp_start).total_seconds() / 60 if fts else total_h * 60
                    entries.append({**res, "label": "Last", "elapsed_min": elapsed_min})
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
                    r = find_qualifying_window(file_names[fi], _file_bytes[file_names[fi]], min_cyc)
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
                entries.append({**res, "label": label, "elapsed_min": elapsed_min})

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

        # ── Build figure ──────────────────────────────────────────────────────
        fig_s = go.Figure()

        for i, e in enumerate(entries):
            t   = e["t_ms"]
            lbl = e["label"]
            if stack_ch in ("Potential (Φ)", "Both"):
                fig_s.add_trace(go.Scatter(
                    x=t, y=e["ch1"], name=f"Φ — {lbl}",
                    line=dict(color=phi_cols[i], width=stack_lw),
                    yaxis="y", legendgroup=lbl,
                    hovertemplate=f"<b>{lbl}</b><br>τ=%{{x:.1f}} ms<br>Φ=%{{y:.4f}} V<extra></extra>",
                ))
            if stack_ch in ("Current (I)", "Both"):
                fig_s.add_trace(go.Scatter(
                    x=t, y=e["ch2"], name=f"I — {lbl}",
                    line=dict(color=curr_cols[i], width=stack_lw),
                    yaxis="y2" if use_dual else "y",
                    legendgroup=lbl,
                    hovertemplate=f"<b>{lbl}</b><br>τ=%{{x:.1f}} ms<br>I=%{{y:.5f}} V<extra></extra>",
                ))

        yax1_range = ([s_phi_min, s_phi_max] if stack_ch != "Current (I)"
                      else [s_curr_min, s_curr_max])
        yax1_title = "Φ in V" if stack_ch != "Current (I)" else "Current (V)"

        layout_kw = dict(
            height=560, margin=dict(l=70, r=90, t=90, b=50),
            plot_bgcolor=bg_s, paper_bgcolor=paper_s, hovermode="x unified",
            legend=dict(x=1.08, y=1, font=dict(size=11, color=ax_s),
                        bgcolor="rgba(0,0,0,0.4)" if stack_theme == "Dark" else "rgba(255,255,255,0.85)",
                        bordercolor=ax_s, borderwidth=1),
            title=dict(text="<b>PPGa7865</b> — Stacked Evolution",
                       font=dict(size=13, color=ax_s), x=0, xanchor="left"),
            xaxis=dict(
                title=dict(text="τ in ms", font=dict(size=14, color=ax_s)),
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
        dl_cols = st.columns([1, 1, 4])
        # PNG via kaleido
        try:
            _png_bytes = fig_s.to_image(format="png", width=1600, height=900, scale=2)
            dl_cols[0].download_button(
                "⬇ PNG",
                data=_png_bytes,
                file_name="stacked_evolution.png",
                mime="image/png",
            )
        except Exception:
            dl_cols[0].caption("PNG: install `kaleido`")
        # Interactive HTML (always works)
        _html_bytes = fig_s.to_html(include_plotlyjs="cdn").encode("utf-8")
        dl_cols[1].download_button(
            "⬇ HTML",
            data=_html_bytes,
            file_name="stacked_evolution.html",
            mime="text/html",
        )

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
            })

        st.dataframe(pd.DataFrame(info_rows), width='stretch', hide_index=True)

    else:
        st.info(
            "Configure the time points above, then click **🚀 Generate Stacked Plot**.\n\n"
            "Default time points: **1 min · 10 min · 1 h · 5 h · 10 h**\n\n"
            "Each trace shows 10 T_on cycles before the first eligible anodic spike "
            "in the file nearest to that time point. Colors go **dark (earliest) → light (latest)**."
        )
