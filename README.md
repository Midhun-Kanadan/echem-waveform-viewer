# Electrochemical Signal Viewer

Interactive Streamlit app for visualising **Yokogawa DL850E** transient recorder data from pulsed galvanic experiments (T_on / T_off cycling with anodic spike detection).

## Live app

> Deploy via [Streamlit Community Cloud](https://streamlit.io/cloud) and share the link.

## Features

| Tab | What it shows |
|-----|--------------|
| **Signal Viewer** | Full waveform browser with spike markers. Window modes: full file, between spikes, centred on spike, manual range. |
| **Stacked Evolution** | Overlays 10 T_on cycles before the first qualifying anodic spike for 7 time points (Earliest · 1 min · 10 min · 1 h · 5 h · 10 h · Last). Download as PNG or interactive HTML. |

## How to use

### 1 — Prepare your files

Your dataset folder should contain files exported from the DL850E:

```
f20260323_155148_452_filter.txt
f20260323_155227_527_filter.txt
...
```

Each file is **tab-separated**, has **10 header rows** to skip, and contains three columns:

```
time(s)   ch1(V)   ch2(V)
```

### 2 — Upload and explore

1. Open the app link.
2. In the sidebar, click **Browse files**.
3. Navigate to your dataset folder, press **Ctrl+A** to select all `*_filter.txt` files, then click **Open**.
4. The viewer loads automatically — no zipping required.

## Local development

```bash
git clone https://github.com/<your-username>/echem-waveform-viewer.git
cd echem-waveform-viewer
pip install -r requirements.txt
streamlit run app.py
```

When running locally you can also upload a ZIP via the sidebar, exactly as in the cloud version.

## Data format

| Column | Channel | Typical range |
|--------|---------|---------------|
| `ch1` | Potential Φ (V) | −3.5 V … 0.5 V |
| `ch2` | Current via shunt (V) | −0.003 V … 0.010 V |

- Sampling rate: **10 kHz** (0.1 ms / sample)
- File duration: **≤ 2.002 s**
- Filename timestamp format: `f YYYYMMDD_HHMMSS_mmm_filter.txt`

## Requirements

```
streamlit >= 1.35
pandas, numpy, scipy, plotly, matplotlib
kaleido  # optional — needed for PNG export in the Stacked Evolution tab
```

## License

For research use. Please cite the originating experiment if you publish results.
