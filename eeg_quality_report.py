"""Per-participant EEG quality report (descriptive; nothing is removed or changed).

Usage:  DATASET_PATH=/path/to/data python eeg_quality_report.py
Uses the same file search and label rule as full-pipeline.py, and the same
loading steps (first 2-D array, transpose if channels are in columns).
For every recording it reports, BEFORE any filtering or NaN replacement:
  n_nan            number of NaN / infinite values
  n_flat_channels  channels with (near) zero variance
  max_abs          largest absolute sample value (units as stored in the file)
and AFTER the pipeline's band-pass filter (0.5-50 Hz, as in the main analysis),
for the non-overlapping 512-sample windows:
  n_windows, pp_median / pp_p95 / pp_max   peak-to-peak amplitude per window and channel
  frac_windows_over_{F}x  fraction of windows in which any channel's peak-to-peak
                          exceeds F times that channel's median window peak-to-peak
                          in the same recording (relative measure)
The units of the stored values are not documented in the dataset files, so no
absolute microvolt threshold is used. The measures are reported, not used to
exclude anything. Writes eeg_quality_by_subject.csv and eeg_quality_summary.txt
into OUT_DIR (default: current directory).
"""
import os, glob
import numpy as np, pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt

DATA_DIR = os.environ.get("DATASET_PATH") or "."
OUT_DIR = os.environ.get("OUT_DIR") or "."
if not os.path.isdir(DATA_DIR) or not glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True):
    raise SystemExit(f"No .mat files found under '{DATA_DIR}'. Set DATASET_PATH to the dataset folder.")
FS, WIN = 128, 512
FACTORS = [3, 5]

def find_files():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True)):
        rel = os.path.relpath(f, DATA_DIR).lower()
        if "control" in rel: out.append((f, 0))
        elif "adhd" in rel: out.append((f, 1))
    return out

b, a = butter(4, [0.5 / (FS / 2), 50.0 / (FS / 2)], btype="band")
rows = []
for sid, (f, label) in enumerate(find_files()):
    mat = sio.loadmat(f)
    arr = next((v for k, v in mat.items() if not k.startswith("__")
                and isinstance(v, np.ndarray) and v.ndim == 2), None)
    if arr is None:
        continue
    arr = arr.astype(float)
    if arr.shape[0] != 19 and arr.shape[1] == 19:
        arr = arr.T
    if arr.shape[0] != 19 or arr.shape[1] < WIN:
        continue
    n_bad = int((~np.isfinite(arr)).sum())
    flat = int((np.nanstd(arr, axis=1) < 1e-8).sum())
    mx = float(np.nanmax(np.abs(arr)))
    x = filtfilt(b, a, np.nan_to_num(arr), axis=-1)
    n_w = x.shape[1] // WIN
    w = x[:, :n_w * WIN].reshape(19, n_w, WIN)
    pp = w.max(axis=2) - w.min(axis=2)                # (19, n_w)
    row = dict(sid=sid, label=label, file=os.path.basename(f), n_samples=arr.shape[1],
               n_windows=n_w, n_nan=n_bad, n_flat_channels=flat, max_abs=round(mx, 1),
               pp_median=round(float(np.median(pp)), 1), pp_p95=round(float(np.percentile(pp, 95)), 1),
               pp_max=round(float(pp.max()), 1))
    med = np.median(pp, axis=1, keepdims=True)       # per-channel median in this recording
    ratio = (pp / (med + 1e-12)).max(axis=0)          # worst channel per window
    for F in FACTORS:
        row[f"frac_windows_over_{F}x"] = round(float((ratio > F).mean()), 4)
    rows.append(row)

d = pd.DataFrame(rows)
d.to_csv(os.path.join(OUT_DIR, "eeg_quality_by_subject.csv"), index=False)
L = [f"recordings: {len(d)} (ADHD {int((d.label==1).sum())}, control {int((d.label==0).sum())}), windows: {int(d.n_windows.sum())}",
     f"recordings with NaN/inf values: {int((d.n_nan>0).sum())} (total values {int(d.n_nan.sum())})",
     f"recordings with a flat channel: {int((d.n_flat_channels>0).sum())}",
     f"max |value| over all recordings: {d.max_abs.max()}  (median of per-recording max: {d.max_abs.median()})",
     f"window peak-to-peak (per channel): median of medians {d.pp_median.median()}, median of p95 {d.pp_p95.median()}, largest {d.pp_max.max()}"]
for F in FACTORS:
    n = int(round((d[f'frac_windows_over_{F}x'] * d.n_windows).sum()))
    L.append(f"windows with a channel peak-to-peak > {F}x its recording median: {n} of {int(d.n_windows.sum())} ({n/d.n_windows.sum()*100:.2f}%); recordings affected: {int((d[f'frac_windows_over_{F}x']>0).sum())}; max per recording {d[f'frac_windows_over_{F}x'].max()*100:.1f}% of windows")
t = "\n".join(L); print(t)
open(os.path.join(OUT_DIR, "eeg_quality_summary.txt"), "w").write(t)
