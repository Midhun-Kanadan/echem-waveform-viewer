# Electrochemical Signal Viewer

Interactive Streamlit app for visualising **Yokogawa DL850E** transient recorder data from pulsed galvanic experiments (T_on / T_off cycling with anodic spike detection).

## Live app

**[echem-waveform-viewer-lfrqn69wyxciso4nojnba2.streamlit.app](https://echem-waveform-viewer-lfrqn69wyxciso4nojnba2.streamlit.app/)**

Hosted on [Streamlit Community Cloud](https://streamlit.io/cloud); it redeploys automatically on every push to `main`. If the app has been idle it may take a moment to wake up.

## Features

| Tab | What it shows |
|-----|--------------|
| **Signal Viewer** | Full waveform browser with spike markers. Window modes: full file, between spikes, centred on spike, manual range. Download as CSV, PNG, HTML or **Excel with a native (editable) chart**. |
| **Stacked Evolution** | Overlays 10 T_on cycles before the first qualifying anodic spike for 7 time points (Earliest · 1 min · 10 min · 1 h · 5 h · 10 h · Last). Optional averaging of N consecutive files per time point. Excel export puts all curves on one common τ grid. |
| **Minimum vs Cycle** | Lowest potential of each of the 10 deposition pulses before the anodic spike, per time point. Pulses are found from the imposed current; values can be averaged over all complete pulse sequences in a file (mean ± SD error bars). Files are picked automatically from time points or listed manually (`label = file name`). Chart with potential or cycle on the x-axis, distinct or graded colours. Download as CSV, HTML, PNG or Excel (charts with error bars), plus a cycle-minima CSV/Excel with a per-cycle *Details* sheet. |
| **Transient Evolution** | For every Nth capture, averages all pulse cycles and plots resting potential, pulse-end potential, pulse depth, anodic-spike height and recovery time against deposition time, with a summary of when each quantity settles. Download as CSV, HTML or Excel with charts. |

**Sidebar options that apply to all tabs**

- **Sample name**: used in titles and file names.
- **Denoise**: off, zero-phase low-pass, or Savitzky-Golay.
  - Spike detection always uses the raw signal.
  - The raw signal can be shown underneath the filtered one.
  - Raw columns are kept in the exports.
- **Include subfolders**: load a full run spread over several folders.
- **Auto-scaled axes**: detection thresholds scale with the signal, so small-amplitude samples work too.

## How to use

### 1 — Prepare your files

Your dataset folder should contain files exported from the DL850E:

```
f20260323_155148_452_filter.txt
f20260323_155227_527_filter.txt
...
```

Each file is **tab-separated**, has **no header**, and contains three columns. The first 10 rows are digital-filter warm-up and are skipped automatically:

```
time(s)   ch1(V)   ch2(V)
```

### 2 — Upload and explore

1. Open the app link.
2. In the sidebar, set **Source** to **Upload files** and click **Browse files**.
3. Navigate to your dataset folder, press **Ctrl+A** to select all `*_filter.txt` files, then click **Open**.
4. The viewer loads automatically — no zipping required.

## Local development

```bash
git clone https://github.com/Midhun-Kanadan/echem-waveform-viewer.git
cd echem-waveform-viewer
pip install -r requirements.txt
streamlit run app.py
```

When running locally you can also choose **Local folder** in the sidebar and paste the path to your dataset folder. Files are then read on demand, which also works for full runs of several GB.

## Data format

| Column | Channel | Typical range |
|--------|---------|---------------|
| `ch1` | Potential Φ (V), measured between cathode and anode | sample-dependent, e.g. −3 … −0.5 V or −0.23 … −0.07 V |
| `ch2` | Current via shunt (V) | −0.003 V … 0.010 V |

- Sampling rate: **10 kHz** (0.1 ms / sample)
- File duration: **≤ 2.002 s**
- Filename timestamp format: `f YYYYMMDD_HHMMSS_mmm_filter.txt`

## Requirements

```
streamlit >= 1.35
pandas, numpy, scipy, plotly, matplotlib
openpyxl # Excel export
kaleido  # optional — PNG export in the Stacked Evolution tab (kaleido ≥ 1.0 also needs Chrome)
```

## License

For research use. Please cite the originating experiment if you publish results.
