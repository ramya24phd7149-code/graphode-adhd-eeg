"""
Graph Neural ODE pipeline for EEG-based ADHD classification.

Loads task-related EEG (attention task), computes multi-band wPLI connectivity, builds
per-window graphs, trains a Graph Neural ODE model (plus several baseline
and control models), and runs the full statistical evaluation (repeated
cross-validation, corrected significance tests, equivalence tests,
permutation tests). All extra behaviour below is opt-in via environment
variables and defaults to off, so running the script with no environment
variables set reproduces the primary reported results.

Env overrides (all optional): DATASET_PATH, OUT_DIR, REPEATS, FOLDS, N_PERM,
RUN_DEEP, RUN_MULTIBAND, RUN_DEGREE_MATCHED, RUN_INTERPRETABILITY,
INTERPRET_MODEL, USE_NAMED_CHANNELS, ODE_STEP, EQUIV_MARGIN,
GRAPH_METHOD, WPLI_THRESHOLD, NODE_HANDCRAFTED, AGG_METHOD,
WEIGHT_DECAY, DROPOUT_OVERRIDE, ARTIFACT_UV_THRESH, RUN_ARTIFACT_REJECTION,
EXCLUDE_FRONTAL, RUN_ICA, ICA_Z_THRESH, ICA_MAX_REMOVE, RUN_WINDOW_SPLIT_COMPARISON, ONLY_WINDOW_SPLIT_COMPARISON, N_PERM_WINDOW, N_PERM_DEEP, N_PERM_DEEP_FOLDS, N_PERM_DEEP_START,
REPEATS_CONTROL, RUN_STACKEDGAT.
"""
import os, glob, time, json, warnings, hashlib
import numpy as np, pandas as pd
import scipy.io as sio
from scipy.signal import butter, filtfilt, get_window
from scipy.stats import t as t_dist, norm
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             matthews_corrcoef, confusion_matrix, roc_curve)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- FIGURE STYLE ----------------------------
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "legend.fontsize": 9.5, "legend.frameon": False,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
})
COLOR_ADHD, COLOR_CTRL = "#d62728", "#1f77b4"
COLOR_PRIMARY, COLOR_ACCENT, COLOR_MUTED = "#2ca02c", "#ff7f0e", "#7f7f7f"
MODEL_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                 "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

CHANNEL_NAMES_DEFAULT = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3",
                         "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
USE_NAMED_CHANNELS = os.environ.get("USE_NAMED_CHANNELS", "0") == "1"

try:
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, global_mean_pool
    from torchdiffeq import odeint
    HAS_TORCH = True
except Exception as e:
    print("[info] torch / torch_geometric / torchdiffeq not available:", e)
    HAS_TORCH = False

# ----------------------------- CONFIG ---------------------------------
DATA_DIR = os.environ.get("DATASET_PATH",
                          "/kaggle/input/datasets/swarnaramya24phd7149/adhd-dataset19")
OUT_DIR = os.environ.get("OUT_DIR", "/kaggle/working/results_v8")
REPEATS = int(os.environ.get("REPEATS", 5))
FOLDS = int(os.environ.get("FOLDS", 10))
# Permutation count raised from 100 -> 1000 for the cheap classical
# baselines (fast: subject-level features, ~121 rows). N_PERM_WINDOW and
# N_PERM_DEEP are SEPARATE, much smaller budgets kept low by default to fit a
# time-constrained run: N_PERM_WINDOW=0 skips the window-trained-SVM
# permutation test by default (the formal fold-level comparison still runs;
# only this extra permutation check is skipped).
# N_PERM_DEEP_FOLDS additionally caps the deep permutation test to the first
# few folds of repeat 0 (not all FOLDS), since every permutation there means
# retraining a GNN from scratch.
N_PERM = int(os.environ.get("N_PERM", 1000))
N_PERM_WINDOW = int(os.environ.get("N_PERM_WINDOW", 0))
N_PERM_DEEP = int(os.environ.get("N_PERM_DEEP", 5))
N_PERM_DEEP_FOLDS = int(os.environ.get("N_PERM_DEEP_FOLDS", 3))
RUN_PERM_DEEP = os.environ.get("RUN_PERM_DEEP", "1") == "1"
RUN_DEEP = HAS_TORCH and os.environ.get("RUN_DEEP", "1") == "1"
RUN_MULTIBAND = os.environ.get("RUN_MULTIBAND", "1") == "1"
RUN_DEGREE_MATCHED = os.environ.get("RUN_DEGREE_MATCHED", "1") == "1"
RUN_INTERPRETABILITY = os.environ.get("RUN_INTERPRETABILITY", "1") == "1"
INTERPRET_MODEL = os.environ.get("INTERPRET_MODEL", "GraphODE")
ODE_STEP = float(os.environ.get("ODE_STEP", 0.2))          # unchanged default from v7
EQUIV_MARGIN = float(os.environ.get("EQUIV_MARGIN", 0.03))  # TOST margin (3 pp)

# Simple amplitude-based artifact rejection,
# applied to the raw (pre-z-score) filtered signal inside load_windows(). A
# window is dropped if any channel's absolute amplitude exceeds this
# threshold (microvolts) at any sample. EXCLUDE_FRONTAL is a SEPARATE
# sensitivity run (set via env, different OUT_DIR) that additionally drops
# Fp1/Fp2 (the two most eye-blink-prone channels) before any feature is
# computed, to check whether frontal-artifact contamination is driving the
# result.
ARTIFACT_UV_THRESH = float(os.environ.get("ARTIFACT_UV_THRESH", 150.0))
# v8.2: default OFF in this time-budgeted run. Turning it ON changes the
# window count for EVERY arm, which forces GraphODE/StackedGAT/MultiBand/
# DegreeMatched to retrain from scratch (~most of the original 92-minute run)
# instead of resuming their already-verified v8.1 results. Recommended
# workflow: first do a 2-3 minute CHECK-ONLY pass (RUN_ARTIFACT_REJECTION=1
# RUN_DEEP=0) to see the rejected-window percentage in artifact_rejection_log
# .csv; only pay for a full retrain if that percentage turns out non-trivial.
RUN_ARTIFACT_REJECTION = os.environ.get("RUN_ARTIFACT_REJECTION", "0") == "1"
EXCLUDE_FRONTAL = os.environ.get("EXCLUDE_FRONTAL", "0") == "1"
FRONTAL_IDX = [0, 1]  # Fp1, Fp2 in the 19-channel montage order

# Optional ICA-based ocular-artifact correction (default OFF). Applied per
# subject to the band-pass-filtered recording, before windowing. FastICA
# decomposes the 19 channels into 19 independent components; the components
# whose time courses correlate most strongly with the frontal channels
# (Fp1/Fp2, used as a proxy for the electro-oculogram because no EOG channel
# is available) are removed. A component is removed when its z-scored
# correlation score across the 19 components exceeds ICA_Z_THRESH (the
# convention of MNE's find_bads_eog), at most ICA_MAX_REMOVE per subject.
# The number of removed components per subject is saved in
# artifact_rejection_log.csv (column n_ic_removed).
RUN_ICA = os.environ.get("RUN_ICA", "0") == "1"
ICA_Z_THRESH = float(os.environ.get("ICA_Z_THRESH", 3.0))
ICA_MAX_REMOVE = int(os.environ.get("ICA_MAX_REMOVE", 2))

# Extra deep-model control arms, each
# individually switchable so a run can be scoped down if needed.
RUN_FULLY_CONNECTED = os.environ.get("RUN_FULLY_CONNECTED", "1") == "1"
RUN_RANDOM_GRAPH = os.environ.get("RUN_RANDOM_GRAPH", "1") == "1"
RUN_STACKEDGAT = os.environ.get("RUN_STACKEDGAT", "1") == "1"
RUN_CNN_BASELINES = os.environ.get("RUN_CNN_BASELINES", "1") == "1"
RUN_WINDOW_TRAINED_SVM = os.environ.get("RUN_WINDOW_TRAINED_SVM", "1") == "1"
# Split-design comparison for the window-trained SVM on wPLI: subject-disjoint outer
# splits (the main design) versus random-window splits (windows of the same
# participant can fall in both training and test sets; NOT an estimate of
# performance on unseen participants). Writes window_split_comparison.csv.
RUN_WINDOW_SPLIT_COMPARISON = os.environ.get("RUN_WINDOW_SPLIT_COMPARISON", "1") == "1"
RUN_HYPERPARAM_SWEEP = os.environ.get("RUN_HYPERPARAM_SWEEP", "1") == "1"
# v8.2: the four new CONTROL/ablation arms (FullyConnected, RandomGraph,
# CNN-Only, CNN-LSTM) run with FEWER repeats than the primary models by
# default -- they exist to show "does this ablation change the picture
# qualitatively", not to carry the same statistical precision as the main
# GraphODE-vs-baseline claims, and 5x fewer repeats is a large, deliberate
# time saving for a budget-constrained rerun.
REPEATS_CONTROL = int(os.environ.get("REPEATS_CONTROL", 1))
os.makedirs(OUT_DIR, exist_ok=True)

# v8: new opt-in knobs -- all default to v7-identical behavior
GRAPH_METHOD = os.environ.get("GRAPH_METHOD", "knn")            # "knn" | "threshold"
WPLI_THRESHOLD = float(os.environ.get("WPLI_THRESHOLD", 0.15))  # used only if GRAPH_METHOD=threshold
NODE_HANDCRAFTED = os.environ.get("NODE_HANDCRAFTED", "0") == "1"
AGG_METHOD = os.environ.get("AGG_METHOD", "mean")               # "mean" | "median" | "trimmed"
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", 1e-4))      # unchanged default from v7
NODE_FEAT_DIM = 4 if NODE_HANDCRAFTED else 0

FS, WIN, STRIDE = 128, 512, 512
K_NEIGHBOURS = 7
HIDDEN, LR = 64, 1e-3
DROPOUT = float(os.environ.get("DROPOUT_OVERRIDE", 0.3))       # unchanged default from v7
MAX_EPOCHS, PATIENCE, BATCH = 30, 6, 64
SEED = 42
FREQ_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
N_BANDS = len(FREQ_BANDS)
N_TRAJ = int(round(1.0 / ODE_STEP)) + 1
BIOMARKER_COLS = ["divergence_rate", "switching_rate"]

# config hash: any change in these -> new resume files (no silent reuse)
_CFG_DICT = {
    "impl": "v8-wpli-std", "folds": FOLDS, "seed": SEED, "k": K_NEIGHBOURS, "hidden": HIDDEN,
    "dropout": DROPOUT, "lr": LR, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
    "batch": BATCH, "ode_step": ODE_STEP, "win": WIN, "stride": STRIDE,
    "graph_method": GRAPH_METHOD, "wpli_threshold": WPLI_THRESHOLD,
    "node_handcrafted": NODE_HANDCRAFTED, "weight_decay": WEIGHT_DECAY,
    # Degree-matched k* is now selected per outer
    # training fold, never from test-subject signal. Bumped so old (leaked) CSVs
    # are never silently resumed/reused.
    "degree_match_scope": "per_fold_train_only",
    # v8.2: the new control arms (FullyConnected/RandomGraph/CNN-Only/CNN-LSTM)
    # and the window-trained SVM are registered under their OWN model names, so
    # they get their own fresh stems automatically -- no hash bump needed for
    # them. Only include things here that change what GraphODE/StackedGAT/
    # MultiBand/DegreeMatched themselves compute, so that when artifact
    # rejection is left OFF (the default for a time-constrained run), those
    # four arms' existing result files from the v8.1 run are correctly
    # resumed/skipped instead of being retrained from scratch.
    "artifact_uv_thresh": ARTIFACT_UV_THRESH if RUN_ARTIFACT_REJECTION else None,
    "exclude_frontal": EXCLUDE_FRONTAL,
}
if RUN_ICA:  # only added when ICA is on, so default-run tags are unchanged
    _CFG_DICT["ica"] = [ICA_Z_THRESH, ICA_MAX_REMOVE]
CFG_TAG = hashlib.md5(json.dumps(_CFG_DICT, sort_keys=True).encode()).hexdigest()[:8]


def deep_stem(name, spec, rep):
    return f"deep_{name}_{CFG_TAG}{spec.get('extra', '')}_rep{rep}"


# ----------------------------- DATA -----------------------------------
def find_files():
    out = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.mat"), recursive=True)):
        rel = os.path.relpath(f, DATA_DIR).lower()
        if "control" in rel:
            out.append((f, 0))
        elif "adhd" in rel:
            out.append((f, 1))
    return out


def apply_ica_eog(arr):
    """ICA ocular-artifact correction for one subject. `arr` is (channels, time),
    already band-pass filtered. Returns (cleaned array, number of components
    removed). Components are scored by their absolute correlation with the
    frontal proxy channels (Fp1/Fp2); see the RUN_ICA note in the config."""
    from sklearn.decomposition import FastICA
    ica = FastICA(n_components=arr.shape[0], whiten="unit-variance",
                  max_iter=500, tol=1e-3, random_state=SEED)
    S = ica.fit_transform(arr.T)                      # (time, components)
    proxy = arr[FRONTAL_IDX]                          # (2, time)
    scores = np.array([max(abs(np.corrcoef(S[:, c], proxy[p])[0, 1])
                           for p in range(proxy.shape[0])) for c in range(S.shape[1])])
    z = (scores - scores.mean()) / (scores.std() + 1e-12)
    order = np.argsort(-z)[:ICA_MAX_REMOVE]
    bad = [int(c) for c in order if z[c] > ICA_Z_THRESH]
    if not bad:
        return arr, 0
    S_clean = S.copy()
    S_clean[:, bad] = 0.0
    return ica.inverse_transform(S_clean).T, len(bad)


def load_windows(files, artifact_thresh=None):
    """Optional amplitude-based artifact
    rejection. `artifact_thresh` (microvolts) is checked against the raw
    band-pass-filtered signal BEFORE per-window z-scoring (z-scoring would
    erase the amplitude information needed to flag an artifact). Pass None to
    reproduce the original (no-rejection) v7/v8 behavior exactly. Also returns
    a per-subject kept/rejected window count log."""
    b, a = butter(4, [0.5 / (FS / 2), 50.0 / (FS / 2)], btype="band")
    X, s, sub_lab, reject_log = [], [], [], []
    for f, label in files:
        try:
            mat = sio.loadmat(f)
            arr = next((v for k, v in mat.items() if not k.startswith("__")
                        and isinstance(v, np.ndarray) and v.ndim == 2), None)
            if arr is None:
                continue
            arr = np.nan_to_num(arr)
            if arr.shape[0] != 19 and arr.shape[1] == 19:
                arr = arr.T
            if arr.shape[0] != 19 or arr.shape[1] < WIN:
                continue
            arr = filtfilt(b, a, arr, axis=-1)
            n_ic = 0
            if RUN_ICA:
                arr, n_ic = apply_ica_eog(arr)
        except Exception as e:
            print("skip", os.path.basename(f), e)
            continue
        sid = len(sub_lab)
        sub_lab.append(label)
        n_kept = n_rej = 0
        for st in range(0, arr.shape[1] - WIN + 1, STRIDE):
            w_raw = arr[:, st:st + WIN]
            if artifact_thresh is not None and np.abs(w_raw).max() > artifact_thresh:
                n_rej += 1
                continue
            w = (w_raw - w_raw.mean(1, keepdims=True)) / (w_raw.std(1, keepdims=True) + 1e-8)
            X.append(w.astype(np.float32)); s.append(sid); n_kept += 1
        reject_log.append({'sid': sid, 'label': label, 'n_kept': n_kept, 'n_rejected': n_rej,
                           'n_ic_removed': n_ic})
    return np.stack(X), np.array(s), np.array(sub_lab), pd.DataFrame(reject_log)


_HANN128 = get_window("hann", 128)
_HANN512 = get_window("hann", WIN)


def wpli_all(X, bands=FREQ_BANDS, chunk=25):
    """Standard wPLI, computed once for broadband AND every band. Unchanged from v7."""
    N, C, T = X.shape
    starts = list(range(0, T - 128 + 1, 64))
    freqs = np.fft.rfftfreq(128, 1.0 / FS)
    fsel = np.where((freqs >= 0.5) & (freqs < 45))[0]
    fsel_freqs = freqs[fsel]
    masks = {"broad": np.ones(len(fsel), bool)}
    for name, (lo, hi) in bands.items():
        masks[name] = (fsel_freqs >= lo) & (fsel_freqs < hi)
    out = {n: np.zeros((N, C, C), np.float32) for n in masks}
    idx = np.arange(C)
    for i in range(0, N, chunk):
        xb = X[i:i + chunk].astype(np.float64)
        segs = np.stack([xb[:, :, s0:s0 + 128] for s0 in starts], axis=2)   # n,c,s,t
        segs = (segs - segs.mean(-1, keepdims=True)) * _HANN128
        Fz = np.fft.rfft(segs, axis=-1)[..., fsel]                         # n,c,s,f
        im = np.einsum("ncsf,ndsf->ncdsf", Fz.conj(), Fz, optimize=True).imag
        num = np.abs(im.sum(3))                                            # n,c,d,f
        den = np.abs(im).sum(3)
        wf = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-20)
        for name, m in masks.items():
            w = wf[..., m].mean(-1)
            w[:, idx, idx] = 0.0
            out[name][i:i + chunk] = w
    for n in out:
        out[n] = np.nan_to_num(out[n])
    return out


def band_power_batch(X, chunk=500):
    """log10 relative band power per channel: (N, 19, 5). Unchanged from v7."""
    freqs = np.fft.rfftfreq(WIN, 1 / FS)
    masks = [(freqs >= lo) & (freqs < hi) for lo, hi in FREQ_BANDS.values()]
    tot = (freqs >= 0.5) & (freqs < 45)
    out = np.zeros((len(X), X.shape[1], len(masks)), np.float32)
    for i in range(0, len(X), chunk):
        P = np.abs(np.fft.rfft(X[i:i + chunk] * _HANN512, axis=-1)) ** 2
        T = P[..., tot].sum(-1) + 1e-12
        for b, m in enumerate(masks):
            out[i:i + chunk, :, b] = P[..., m].sum(-1) / T
    return np.log10(out + 1e-6)


def subject_features(W, BP, w_sid, n_sub):
    iu = np.triu_indices(W.shape[1], 1)
    fw = np.zeros((n_sub, len(iu[0])), np.float32)
    fb = np.zeros((n_sub, BP.shape[1] * BP.shape[2]), np.float32)
    for s in range(n_sub):
        m = w_sid == s
        fw[s] = W[m][:, iu[0], iu[1]].mean(0)
        fb[s] = BP[m].mean(0).ravel()
    return {"wPLI": fw, "BandPower": fb, "wPLI+BP": np.hstack([fw, fb])}


# v8: node-level handcrafted features, built from BP/W that v7 already computes.
# [log theta power, log beta power, log theta/beta ratio, node strength], z-scored
# globally across all windows so the CNN-embedding-scale feature doesn't dominate.
BAND_NAMES = list(FREQ_BANDS.keys())  # delta, theta, alpha, beta, gamma
THETA_IDX, BETA_IDX = BAND_NAMES.index("theta"), BAND_NAMES.index("beta")


def compute_node_handcrafted(BP, W):
    theta = BP[:, :, THETA_IDX]
    beta = BP[:, :, BETA_IDX]
    ratio = theta - beta  # log ratio, since BP is already log10
    strength = W.sum(axis=2)
    feats = np.stack([theta, beta, ratio, strength], axis=-1).astype(np.float32)
    mu = feats.mean(axis=(0, 1), keepdims=True)
    sd = feats.std(axis=(0, 1), keepdims=True) + 1e-8
    return (feats - mu) / sd  # (N, 19, 4)


# ----------------------------- DEGREE UTILITIES ------------------------
def topk_mask_batch(W, k):
    N, C, _ = W.shape
    idx = np.argsort(W, axis=2)[:, :, -k:]
    mask = np.zeros_like(W, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=2)
    mask = mask | mask.transpose(0, 2, 1)
    mask &= ~np.eye(C, dtype=bool)[None, :, :]
    return mask


def threshold_mask_batch(W, thresh):
    """v8: alternative to top-k -- keep all edges with |wPLI| > thresh."""
    C = W.shape[1]
    mask = np.abs(W) > thresh
    mask &= ~np.eye(C, dtype=bool)[None, :, :]
    return mask


def avg_degree_from_mask(mask):
    return float(mask.sum(axis=(1, 2)).mean() / mask.shape[1])


def find_degree_matched_k(W_single, target_degree, k_candidates=range(3, 19)):
    best_k, best_diff, log = None, float("inf"), []
    for k in k_candidates:
        deg = avg_degree_from_mask(topk_mask_batch(W_single, k))
        log.append((k, deg))
        if abs(deg - target_degree) < best_diff:
            best_diff, best_k = abs(deg - target_degree), k
    print(f"  [degree match] target={target_degree:.2f} -> " +
          ", ".join(f"k={k}:{d:.2f}" for k, d in log) + f" -> selected k={best_k}")
    return best_k, log


def union_mask_multiband(band_matrices, k):
    masks = [topk_mask_batch(W, k) for W in band_matrices.values()]
    union = masks[0]
    for m in masks[1:]:
        union = union | m
    return union


# ----------------------------- METRICS / STATS ------------------------
def metrics(y, p, thr=0.5):
    y = np.asarray(y); pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, pred),
                auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                f1=f1_score(y, pred, zero_division=0),
                sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
                mcc=matthews_corrcoef(y, pred))


def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed); accs, aucs = [], []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        accs.append(((p[i] >= 0.5) == y[i]).mean())
        if len(set(y[i])) > 1:
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(accs, [2.5, 97.5]), np.percentile(aucs, [2.5, 97.5])


def benjamini_hochberg(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    valid = ~np.isnan(pvals)
    adjusted = np.full(n, np.nan)
    if valid.sum() == 0:
        return adjusted
    idx = np.where(valid)[0]
    p = pvals[idx]; m = len(p)
    order = np.argsort(p)
    a = p[order] * m / np.arange(1, m + 1)
    a = np.clip(np.minimum.accumulate(a[::-1])[::-1], 0, 1)
    out = np.empty(m); out[order] = a
    adjusted[idx] = out
    return adjusted


def nb_corrected_ttest(a, b, n_train, n_test):
    d = np.asarray(a, float) - np.asarray(b, float); k = len(d); v = d.var(ddof=1)
    if k < 2 or v == 0:
        return np.nan, (1.0 if d.mean() == 0 else 0.0)
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * v)
    return float(t), float(2 * t_dist.sf(abs(t), k - 1))


def paired_comparison(folds, name_a, name_b, alpha=0.05, power=0.8, margin=EQUIV_MARGIN):
    m = folds[folds.model == name_a].merge(folds[folds.model == name_b],
                                           on=["repeat", "fold"], suffixes=("_a", "_b"))
    d = (m.acc_a - m.acc_b).values; k = len(d)
    if k < 2:
        return None
    n_tr, n_te = m.n_train_a.mean(), m.n_test_a.mean()
    se = max(float(np.sqrt((1.0 / k + n_te / n_tr) * d.var(ddof=1))), 1e-12)
    mean = float(d.mean())
    p = float(2 * t_dist.sf(abs(mean / se), k - 1))
    tc = t_dist.ppf(1 - alpha / 2, k - 1)
    mde = float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)
    p_lo = float(t_dist.sf((mean + margin) / se, k - 1))
    p_hi = float(t_dist.cdf((mean - margin) / se, k - 1))
    return {"mean_diff": mean, "p_nb": p, "se": se,
            "ci": (mean - tc * se, mean + tc * se), "mde": mde,
            "tost_p": max(p_lo, p_hi), "margin": margin, "n_pairs": k}


def make_splits(sub_lab):
    rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    return [(tr, te) for tr, te in rskf.split(np.zeros(len(sub_lab)), sub_lab)]


# ----------------------------- BASELINES ------------------------------
def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))


def make_svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=False))


def clf_score(clf, Xte):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(Xte)[:, 1]
    return 1.0 / (1.0 + np.exp(-clf.decision_function(Xte)))


BASELINES = {"LR": make_lr, "SVM": make_svm}


# v8.2 fix: window-level training has ~30-40x more rows per fold than
# subject-level training (thousands of windows vs ~110 subjects). An RBF SVC
# at that row count can take tens of minutes PER FOLD (libsvm's RBF solver
# scales poorly with n_samples), which is what made the window-trained-SVM
# step look "stuck" with no per-fold progress printed. LinearSVC (liblinear)
# scales far better with n_samples and is a legitimate, standard choice for
# this many rows -- used ONLY for the window-trained arm, not the original
# subject-level SVM[*] baselines (those keep the RBF kernel unchanged).
def make_svm_window():
    return make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000))


BASELINES_WINDOW = {"LR": make_lr, "SVM": make_svm_window}


def run_baselines(FEATS, y, splits):
    folds, oof = [], []
    for fname, Fm in FEATS.items():
        for mname, mk in BASELINES.items():
            name = f"{mname}[{fname}]"
            for i, (tr, te) in enumerate(splits):
                p = clf_score(mk().fit(Fm[tr], y[tr]), Fm[te])
                r = metrics(y[te], p)
                r.update(model=name, repeat=i // FOLDS, fold=i % FOLDS + 1, n_train=len(tr), n_test=len(te))
                folds.append(r)
                oof += [dict(model=name, repeat=i // FOLDS, sid=int(s), p=float(pp), y=int(y[s]))
                        for s, pp in zip(te, p)]
    return pd.DataFrame(folds), pd.DataFrame(oof)


def permutation_pvalue(Fm, y, mk, obs_acc, B):
    rng = np.random.RandomState(SEED); null = []
    for b in range(B):
        yp = rng.permutation(y); accs = []
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=b).split(Fm, yp):
            p = clf_score(mk().fit(Fm[tr], yp[tr]), Fm[te])
            accs.append(((p >= 0.5) == yp[te]).mean())
        null.append(np.mean(accs))
    null = np.array(null)
    return (1 + np.sum(null >= obs_acc)) / (B + 1), float(null.mean())


# Window-TRAINED classical
# baselines. run_baselines() above trains on subject-level features (each
# subject's windows averaged first). Here the classifier is trained directly
# on window-level rows, and only the TEST-side predictions are pooled to
# subject level (by averaging per-window probability) -- the same pooling the
# deep models use. This is reported as its own named model
# ("...-window-trained") so it gets its own row, its own NB-corrected p-value
# vs. the reference model, and its own permutation test, instead of being an
# informal side-exploration.
def window_features(W, BP):
    iu = np.triu_indices(W.shape[1], 1)
    fw = W[:, iu[0], iu[1]]
    fb = BP.reshape(BP.shape[0], -1)
    return {"wPLI": fw, "BandPower": fb, "wPLI+BP": np.hstack([fw, fb])}


def run_window_trained_baselines(WIN_FEATS, y_sub, w_sid, splits):
    folds, oof = [], []
    for fname, Fm in WIN_FEATS.items():
        for mname, mk in BASELINES_WINDOW.items():
            name = f"{mname}[{fname}]-window-trained"
            t_name = time.time()
            for i, (tr_s, te_s) in enumerate(splits):
                tr_w = np.where(np.isin(w_sid, tr_s))[0]
                te_w = np.where(np.isin(w_sid, te_s))[0]
                clf = mk().fit(Fm[tr_w], y_sub[w_sid[tr_w]])
                p_win = clf_score(clf, Fm[te_w])
                g = pd.DataFrame({'sid': w_sid[te_w], 'p': p_win}).groupby('sid')['p'].mean()
                sid_arr, p_arr = g.index.values, g.values
                r = metrics(y_sub[sid_arr], p_arr)
                r.update(model=name, repeat=i // FOLDS, fold=i % FOLDS + 1,
                         n_train=len(tr_s), n_test=len(te_s))
                folds.append(r)
                oof += [dict(model=name, repeat=i // FOLDS, sid=int(sidv), p=float(pp), y=int(y_sub[sidv]))
                        for sidv, pp in zip(sid_arr, p_arr)]
                if (i + 1) % 10 == 0:
                    print(f"  [{name}] {i+1}/{len(splits)} folds done ({time.time()-t_name:.0f}s so far)")
    return pd.DataFrame(folds), pd.DataFrame(oof)


def run_window_split_comparison(Fm, y_sub, w_sid, subject_splits):
    """Window-trained linear SVM on window-level wPLI features under two split designs.
    'Subject-disjoint' reuses the main outer participant splits. 'Random-window' uses
    repeated stratified FOLDS-fold splits of the windows themselves (same number of folds
    and repeats), so windows of one participant can appear in both training and test
    sets. For each repeat all out-of-fold windows are pooled; window accuracy/AUC are
    computed on windows, and participant accuracy/AUC on the mean window score
    (sigmoid of the SVM decision value; threshold 0.5) of each participant."""
    y_win = y_sub[w_sid]
    schemes = {}
    schemes["Subject-disjoint"] = [(np.where(np.isin(w_sid, tr_s))[0], np.where(np.isin(w_sid, te_s))[0])
                                   for tr_s, te_s in subject_splits]
    rskf = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED)
    schemes["Random-window (not subject-disjoint)"] = [(tr, te) for tr, te in rskf.split(np.zeros(len(y_win)), y_win)]
    rows_rep, rows_fold = [], []
    for scheme, sp in schemes.items():
        t0 = time.time()
        for rep in range(REPEATS):
            p_all = np.full(len(y_win), np.nan)
            for f_i, (tr_w, te_w) in enumerate(sp[rep * FOLDS:(rep + 1) * FOLDS]):
                clf = make_svm_window().fit(Fm[tr_w], y_win[tr_w])
                p_all[te_w] = clf_score(clf, Fm[te_w])
                rows_fold.append(dict(scheme=scheme, repeat=rep, fold=f_i + 1,
                                      window_acc=float(((p_all[te_w] >= 0.5) == y_win[te_w]).mean())))
            assert not np.isnan(p_all).any()
            sub = pd.DataFrame({"sid": w_sid, "p": p_all}).groupby("sid")["p"].mean()
            ys = y_sub[sub.index.values]
            rows_rep.append(dict(scheme=scheme, repeat=rep,
                                 window_acc=float(((p_all >= 0.5) == y_win).mean()),
                                 window_auc=float(roc_auc_score(y_win, p_all)),
                                 participant_acc=float(((sub.values >= 0.5) == ys).mean()),
                                 participant_auc=float(roc_auc_score(ys, sub.values))))
        print(f"  [window split comparison] {scheme} done ({time.time()-t0:.0f}s)")
    rep_df, fold_df = pd.DataFrame(rows_rep), pd.DataFrame(rows_fold)
    out = []
    for scheme in schemes:
        r = rep_df[rep_df.scheme == scheme]; f = fold_df[fold_df.scheme == scheme]
        out.append({"scheme": scheme,
                    "window_acc_mean": r.window_acc.mean() * 100, "window_acc_sd": r.window_acc.std(ddof=1) * 100,
                    "window_auc_mean": r.window_auc.mean(), "window_auc_sd": r.window_auc.std(ddof=1),
                    "participant_acc_mean": r.participant_acc.mean() * 100,
                    "participant_acc_sd": r.participant_acc.std(ddof=1) * 100,
                    "participant_auc_mean": r.participant_auc.mean(), "participant_auc_sd": r.participant_auc.std(ddof=1),
                    "fold_window_acc_mean": f.window_acc.mean() * 100, "fold_window_acc_sd": f.window_acc.std(ddof=1) * 100})
    return pd.DataFrame(out), rep_df, fold_df


def permutation_pvalue_window(Fm, y_sub, w_sid, mk, obs_acc, B):
    """Same logic as permutation_pvalue(), but splits/trains/pools at the
    window level like run_window_trained_baselines(). Uses its own (smaller)
    budget N_PERM_WINDOW, since each permutation retrains on far more rows
    (windows, not subjects) than the subject-level classical baselines."""
    rng = np.random.RandomState(SEED); null = []
    subs_all = np.unique(w_sid)
    for b in range(B):
        yp_sub = rng.permutation(y_sub)
        accs = []
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=b).split(subs_all, yp_sub[subs_all]):
            tr_s, te_s = subs_all[tr], subs_all[te]
            tr_w = np.where(np.isin(w_sid, tr_s))[0]; te_w = np.where(np.isin(w_sid, te_s))[0]
            clf = mk().fit(Fm[tr_w], yp_sub[w_sid[tr_w]])
            p_win = clf_score(clf, Fm[te_w])
            g = pd.DataFrame({'sid': w_sid[te_w], 'p': p_win}).groupby('sid')['p'].mean()
            accs.append(((g.values >= 0.5) == yp_sub[g.index.values]).mean())
        null.append(np.mean(accs))
    null = np.array(null)
    return (1 + np.sum(null >= obs_acc)) / (B + 1), float(null.mean())


# ----------------------------- DEEP MODELS ----------------------------
if HAS_TORCH:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _edge_mask_for(A, k, method, thresh):
        """v8: single place that decides top-k vs threshold edge selection
        for one adjacency matrix A (C,C)."""
        C = A.shape[0]
        if method == "threshold":
            m = np.abs(A) > thresh
        else:
            idx = np.argsort(A, axis=1)[:, -k:]
            m = np.zeros_like(A, dtype=bool)
            m[np.arange(C)[:, None], idx] = True
        m = m | m.T
        np.fill_diagonal(m, False)
        return m

    def make_graphs(X, y_sub, s, W, k, node_feat=None, method=GRAPH_METHOD, thresh=WPLI_THRESHOLD):
        """Single-band graphs (edge_dim=1). v8: method/thresh select top-k vs
        absolute-threshold edges; node_feat (N,19,4) is optionally attached."""
        C = X.shape[1]; graphs = []
        for i in range(len(X)):
            A = W[i]
            S_mask = _edge_mask_for(A, k, method, thresh)
            r, c = np.nonzero(S_mask)
            nf = (torch.from_numpy(node_feat[i]) if node_feat is not None
                  else torch.zeros((C, 0), dtype=torch.float32))
            graphs.append(Data(x=torch.from_numpy(X[i]),
                               edge_index=torch.from_numpy(np.vstack([r, c])).long(),
                               edge_attr=torch.from_numpy(A[r, c]).float().unsqueeze(-1),
                               node_feat=nf,
                               y=torch.tensor(int(y_sub[s[i]])), sid=torch.tensor(int(s[i]))))
        return graphs

    def make_multiband_graphs(X, y_sub, s, band_matrices, k, node_feat=None,
                               method=GRAPH_METHOD, thresh=WPLI_THRESHOLD):
        """Edges = union of each band's top-k/threshold mask; edge_attr =
        per-band wPLI (edge_dim=N_BANDS)."""
        band_names = list(band_matrices.keys())
        graphs = []
        for i in range(len(X)):
            union = np.zeros_like(band_matrices[band_names[0]][i], dtype=bool)
            per_band_A = []
            for name in band_names:
                A = band_matrices[name][i]
                per_band_A.append(A)
                union |= _edge_mask_for(A, k, method, thresh)
            np.fill_diagonal(union, False)
            r, c = np.nonzero(union)
            edge_attr = np.stack([per_band_A[b][r, c] for b in range(len(band_names))], axis=1).astype(np.float32)
            C = X.shape[1]
            nf = (torch.from_numpy(node_feat[i]) if node_feat is not None
                  else torch.zeros((C, 0), dtype=torch.float32))
            graphs.append(Data(x=torch.from_numpy(X[i]),
                               edge_index=torch.from_numpy(np.vstack([r, c])).long(),
                               edge_attr=torch.from_numpy(edge_attr),
                               node_feat=nf,
                               y=torch.tensor(int(y_sub[s[i]])), sid=torch.tensor(int(s[i]))))
        return graphs

    # Fully-connected control. Every channel
    # pair is an edge (no top-k sparsification); edge_attr is still that
    # pair's real wPLI value. If this scores the same as the top-k GraphODE,
    # sparsification itself isn't helping/hurting; if it scores differently,
    # density/topology matters beyond what the degree-matched control shows.
    def make_fully_connected_graphs(X, y_sub, s, W, node_feat=None):
        C = X.shape[1]; graphs = []
        full_mask = ~np.eye(C, dtype=bool)
        r, c = np.nonzero(full_mask)
        for i in range(len(X)):
            A = W[i]
            nf = (torch.from_numpy(node_feat[i]) if node_feat is not None
                  else torch.zeros((C, 0), dtype=torch.float32))
            graphs.append(Data(x=torch.from_numpy(X[i]),
                               edge_index=torch.from_numpy(np.vstack([r, c])).long(),
                               edge_attr=torch.from_numpy(A[r, c]).float().unsqueeze(-1),
                               node_feat=nf,
                               y=torch.tensor(int(y_sub[s[i]])), sid=torch.tensor(int(s[i]))))
        return graphs

    # Random-topology control with the SAME edge count as the top-k graph:
    # for every window, the number of undirected edges of that window's top-k
    # graph (k = K_NEIGHBOURS, after symmetrisation) is counted, and the same
    # number of undirected pairs is drawn uniformly at random. Only the
    # TOPOLOGY is randomised: edge_attr is still the real wPLI value of
    # whichever pairs are chosen. This isolates whether rank-based edge
    # selection matters, at identical edge density.
    def make_random_graphs(X, y_sub, s, W, k, seed=SEED, node_feat=None):
        rng = np.random.RandomState(seed)
        C = X.shape[1]; graphs = []
        iu = np.triu_indices(C, 1)                      # all undirected pairs
        target = topk_mask_batch(W, k)                  # top-k graph of every window
        for i in range(len(X)):
            A = W[i]
            n_und = int(target[i][iu].sum())            # undirected edges to draw
            chosen = rng.choice(len(iu[0]), size=n_und, replace=False)
            mask = np.zeros((C, C), dtype=bool)
            mask[iu[0][chosen], iu[1][chosen]] = True
            mask = mask | mask.T
            r, c = np.nonzero(mask)
            nf = (torch.from_numpy(node_feat[i]) if node_feat is not None
                  else torch.zeros((C, 0), dtype=torch.float32))
            graphs.append(Data(x=torch.from_numpy(X[i]),
                               edge_index=torch.from_numpy(np.vstack([r, c])).long(),
                               edge_attr=torch.from_numpy(A[r, c]).float().unsqueeze(-1),
                               node_feat=nf,
                               y=torch.tensor(int(y_sub[s[i]])), sid=torch.tensor(int(s[i]))))
        return graphs

    class Encoder(nn.Module):
        def __init__(self, out=HIDDEN):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Linear(32, out)

        def forward(self, x):
            return self.fc(self.conv(x.unsqueeze(1)).squeeze(-1))

    def init_w(m):
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu"); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def head(h, p):
        return nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 2))

    class ODEFunc(nn.Module):
        def __init__(self, h, edge_dim=1):
            super().__init__()
            self.gat = GATConv(h, h // 2, heads=2, edge_dim=edge_dim)
            self.ei = self.ea = None

        def forward(self, t, x):
            return torch.tanh(self.gat(x, self.ei, edge_attr=self.ea))

    class GraphODE(nn.Module):
        def __init__(self, h=HIDDEN, p=DROPOUT, edge_dim=1, node_feat_dim=NODE_FEAT_DIM):
            super().__init__()
            self.enc = Encoder(h)
            self.node_feat_dim = node_feat_dim
            if node_feat_dim > 0:
                self.merge = nn.Linear(h + node_feat_dim, h)  # v8: fuse handcrafted node features
            self.gat_in = GATConv(h, h // 4, heads=4, edge_dim=edge_dim)
            self.func = ODEFunc(h, edge_dim=edge_dim)
            self.cls = head(h, p)
            self.register_buffer("t", torch.tensor([0.0, 1.0]))
            self.register_buffer("t_traj", torch.linspace(0.0, 1.0, steps=N_TRAJ))
            self.apply(init_w)

        def _embed(self, d):
            x = self.enc(d.x)
            if self.node_feat_dim > 0:
                x = F.relu(self.merge(torch.cat([x, d.node_feat], dim=-1)))
            return x

        def _ode(self, x, t):
            return odeint(self.func, x, t, method="rk4", options={"step_size": ODE_STEP})

        def forward(self, d, return_att=False, return_trajectory=False):
            x = self._embed(d)
            if return_att:
                x_att, (edge_idx, alpha) = self.gat_in(x, d.edge_index, edge_attr=d.edge_attr,
                                                       return_attention_weights=True)
                x_att = F.relu(x_att)
                self.func.ei, self.func.ea = d.edge_index, d.edge_attr
                x_ode = self._ode(x_att, self.t)[1]
                return self.cls(global_mean_pool(x_ode, d.batch)), edge_idx, alpha
            x = F.relu(self.gat_in(x, d.edge_index, edge_attr=d.edge_attr))
            self.func.ei, self.func.ea = d.edge_index, d.edge_attr
            if return_trajectory:
                return self._ode(x, self.t_traj)
            x = self._ode(x, self.t)[1]
            return self.cls(global_mean_pool(x, d.batch))

    class StackedGAT(nn.Module):
        def __init__(self, h=HIDDEN, p=DROPOUT, edge_dim=1, node_feat_dim=NODE_FEAT_DIM):
            super().__init__()
            self.enc = Encoder(h)
            self.node_feat_dim = node_feat_dim
            if node_feat_dim > 0:
                self.merge = nn.Linear(h + node_feat_dim, h)
            self.g = nn.ModuleList([GATConv(h, h // 4, heads=4, edge_dim=edge_dim) for _ in range(3)])
            self.cls = head(h, p)
            self.apply(init_w)

        def forward(self, d):
            x = self.enc(d.x)
            if self.node_feat_dim > 0:
                x = F.relu(self.merge(torch.cat([x, d.node_feat], dim=-1)))
            for gat in self.g:
                x = F.relu(gat(x, d.edge_index, edge_attr=d.edge_attr))
            return self.cls(global_mean_pool(x, d.batch))

    # CNN-only ablation -- identical encoder and
    # classification head as GraphODE/StackedGAT, but NO graph layer at all
    # (edges are simply ignored). Isolates how much of the model's accuracy
    # comes from the per-channel CNN feature extractor alone, vs. the graph
    # convolution / ODE machinery built on top of it.
    class CNNOnly(nn.Module):
        def __init__(self, h=HIDDEN, p=DROPOUT, node_feat_dim=NODE_FEAT_DIM):
            super().__init__()
            self.enc = Encoder(h)
            self.node_feat_dim = node_feat_dim
            if node_feat_dim > 0:
                self.merge = nn.Linear(h + node_feat_dim, h)
            self.cls = head(h, p)
            self.apply(init_w)

        def forward(self, d):
            x = self.enc(d.x)
            if self.node_feat_dim > 0:
                x = F.relu(self.merge(torch.cat([x, d.node_feat], dim=-1)))
            return self.cls(global_mean_pool(x, d.batch))

    # CNN-LSTM baseline -- same per-channel CNN
    # front end, followed by a bidirectional LSTM over the CNN's output time
    # steps (captures temporal dynamics within a channel) instead of any
    # graph/ODE layer across channels. A second, non-graph deep-learning
    # comparison point alongside CNN-only.
    class CNNLSTM(nn.Module):
        def __init__(self, h=HIDDEN, p=DROPOUT):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 16, 15, stride=2, padding=7), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32), nn.ReLU())
            self.lstm = nn.LSTM(32, h // 2, batch_first=True, bidirectional=True)
            self.cls = head(h, p)
            self.apply(init_w)

        def forward(self, d):
            x = self.conv(d.x.unsqueeze(1))      # (n_nodes_in_batch, 32, T')
            x = x.transpose(1, 2)                # (n_nodes_in_batch, T', 32)
            out, _ = self.lstm(x)
            x = out[:, -1, :]                    # last time step, both directions
            return self.cls(global_mean_pool(x, d.batch))

    def subset(graphs, idx):
        return [graphs[i] for i in idx]

    @torch.no_grad()
    def val_loss(m, loader):
        m.eval(); tot, n = 0.0, 0
        for b in loader:
            b = b.to(dev)
            tot += F.cross_entropy(m(b), b.y, reduction="sum").item(); n += b.y.numel()
        return tot / max(n, 1)

    def fit(make_model, graphs, tr_idx, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        subs = np.unique(W_SID[tr_idx]); labs = SUB_LAB[subs]
        tr_s, va_s = train_test_split(subs, test_size=0.15, stratify=labs, random_state=seed)
        tr = subset(graphs, tr_idx[np.isin(W_SID[tr_idx], tr_s)])
        va = subset(graphs, tr_idx[np.isin(W_SID[tr_idx], va_s)])
        tl = DataLoader(tr, batch_size=BATCH, shuffle=True)
        vl = DataLoader(va, batch_size=128)
        m = make_model().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best, bad, state = 1e9, 0, None
        for ep in range(MAX_EPOCHS):
            m.train()
            for b in tl:
                b = b.to(dev)
                opt.zero_grad()
                F.cross_entropy(m(b), b.y).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
            vloss = val_loss(m, vl)
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        m.load_state_dict(state)
        return m

    # Identical to fit() above, but also returns
    # the per-epoch train/val loss history. Kept as a SEPARATE function (not a
    # flag on fit()) so the main 50-fold training loop used for every reported
    # number is completely untouched; this is called only for the specific
    # (repeat 0, every fold, INTERPRET_MODEL) runs used to produce the
    # training-curve evidence requested by the reviewer.
    def fit_with_history(make_model, graphs, tr_idx, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        subs = np.unique(W_SID[tr_idx]); labs = SUB_LAB[subs]
        tr_s, va_s = train_test_split(subs, test_size=0.15, stratify=labs, random_state=seed)
        tr = subset(graphs, tr_idx[np.isin(W_SID[tr_idx], tr_s)])
        va = subset(graphs, tr_idx[np.isin(W_SID[tr_idx], va_s)])
        tl = DataLoader(tr, batch_size=BATCH, shuffle=True)
        vl = DataLoader(va, batch_size=128)
        m = make_model().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best, bad, state, history = 1e9, 0, None, []
        for ep in range(MAX_EPOCHS):
            m.train(); tr_sum, tr_n = 0.0, 0
            for b in tl:
                b = b.to(dev); opt.zero_grad()
                loss = F.cross_entropy(m(b), b.y)
                loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                tr_sum += loss.item() * b.y.numel(); tr_n += b.y.numel()
            vloss = val_loss(m, vl)
            history.append({'epoch': ep, 'train_loss': tr_sum / max(tr_n, 1), 'val_loss': vloss})
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        m.load_state_dict(state)
        return m, history

    # Like fit(), but training/validation targets
    # are overridden by y_perm (indexed by subject id) instead of the TRUE
    # label baked into each graph's Data.y at construction time -- used only
    # to build the null distribution for the deep-model permutation test.
    def fit_permuted(make_model, graphs, tr_idx, y_perm, w_sid, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        subs = np.unique(w_sid[tr_idx]); labs = y_perm[subs]
        tr_s, va_s = train_test_split(subs, test_size=0.15, stratify=labs, random_state=seed)
        tr = subset(graphs, tr_idx[np.isin(w_sid[tr_idx], tr_s)])
        va = subset(graphs, tr_idx[np.isin(w_sid[tr_idx], va_s)])
        # v8.2 fix: y_perm_t must live on the SAME device as b.sid after
        # b.to(dev) moves every tensor attribute (sid included) to the GPU --
        # indexing a CPU tensor with GPU indices raises a RuntimeError.
        y_perm_t = torch.from_numpy(y_perm.astype(np.int64)).to(dev)
        tl = DataLoader(tr, batch_size=BATCH, shuffle=True)
        vl = DataLoader(va, batch_size=128)
        m = make_model().to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best, bad, state = 1e9, 0, None
        for ep in range(MAX_EPOCHS):
            m.train()
            for b in tl:
                b = b.to(dev); b.y = y_perm_t[b.sid]
                opt.zero_grad()
                F.cross_entropy(m(b), b.y).backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            m.eval(); tot, n = 0.0, 0
            with torch.no_grad():
                for b in vl:
                    b = b.to(dev); b.y = y_perm_t[b.sid]
                    tot += F.cross_entropy(m(b), b.y, reduction="sum").item(); n += b.y.numel()
            vloss = tot / max(n, 1)
            if vloss < best - 1e-4:
                best, bad, state = vloss, 0, {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        m.load_state_dict(state)
        return m

    def _pool_probs(s):
        """v8: window->subject probability pooling. AGG_METHOD default 'mean'
        reproduces v7 exactly; 'median'/'trimmed' are opt-in alternatives that
        down-weight noisy individual-window predictions."""
        if AGG_METHOD == "median":
            return s.median()
        if AGG_METHOD == "trimmed":
            arr = np.sort(s.values); n = len(arr)
            cut = max(1, int(0.1 * n))
            return arr[cut:n - cut].mean() if n - 2 * cut > 0 else arr.mean()
        return s.mean()

    @torch.no_grad()
    def predict(m, graphs, idx):
        m.eval(); ps, ss = [], []
        for b in DataLoader(subset(graphs, idx), batch_size=128):
            b = b.to(dev)
            ps.append(F.softmax(m(b), 1)[:, 1].cpu()); ss.append(b.sid.cpu())
        df = pd.DataFrame({"sid": torch.cat(ss).numpy(), "p": torch.cat(ps).numpy()})
        g = df.groupby("sid")["p"].agg(_pool_probs)
        return g.index.values, g.values

    def model_ckpt_path(name, spec, rep, k):
        return f"{OUT_DIR}/model_{name}_{CFG_TAG}{spec.get('extra', '')}_rep{rep}_fold{k}.pt"

    def run_deep(splits, MODELS):
        # v8.2: model-major loop (was repeat-major) so each model can have its
        # OWN repeat count via spec.get('repeats', REPEATS) -- the primary
        # models (GraphODE, StackedGAT, MultiBand, DegreeMatched) keep the
        # full REPEATS for statistical validity; the new ablation/control
        # arms default to REPEATS_CONTROL (1) to keep a combined rerun
        # affordable.
        for name, spec in MODELS.items():
            n_rep = spec.get('repeats', REPEATS)
            for rep in range(n_rep):
                stem = deep_stem(name, spec, rep)
                ff, fo = f"{OUT_DIR}/{stem}_folds.csv", f"{OUT_DIR}/{stem}_oof.csv"
                need_loss_curves = (rep == 0 and name == INTERPRET_MODEL and
                                    not all(os.path.exists(f"{OUT_DIR}/{stem}_rep0_fold{kk+1}_loss_curve.csv")
                                            for kk in range(FOLDS)))
                if os.path.exists(ff) and os.path.exists(fo) and not need_loss_curves:
                    print(f"[resume] {name} repeat {rep} already done -> skipped"); continue
                if os.path.exists(ff) and os.path.exists(fo) and need_loss_curves:
                    print(f"[resume override] {name} repeat {rep}: results exist but loss curves "
                          f"are missing -> retraining this repeat once to capture them")
                rows, orows, k_log = [], [], []
                for k in range(FOLDS):
                    tr_s, te_s = splits[rep * FOLDS + k]
                    t0 = time.time()

                    if spec.get('degree_matched_per_fold'):
                        # matched_k is selected using
                        # ONLY this fold's training subjects -- the held-out test subjects
                        # of this fold are never used to choose the graph-construction
                        # parameter. The resulting k is then used to build graphs for
                        # every subject (train AND test); applying a train-selected
                        # hyperparameter to test windows is standard practice and is not
                        # itself leakage -- using test-subject signal to CHOOSE it was.
                        train_win = np.where(np.isin(W_SID, tr_s))[0]
                        band_train = {bn: bm[train_win] for bn, bm in band_matrices.items()}
                        mb_target_fold = avg_degree_from_mask(
                            union_mask_multiband(band_train, K_NEIGHBOURS))
                        fold_k, _ = find_degree_matched_k(W[train_win], target_degree=mb_target_fold)
                        graphs = make_graphs(X_ALIVE, SUB_LAB, W_SID, W, fold_k, node_feat=NODE_FEAT)
                        k_log.append({'repeat': rep, 'fold': k + 1,
                                      'train_only_mb_target_degree': mb_target_fold,
                                      'matched_k': fold_k})
                        print(f"  [{name}] rep {rep+1}/{n_rep} fold {k+1}/{FOLDS}: "
                              f"train-only matched_k={fold_k} (target degree={mb_target_fold:.2f})")
                    else:
                        graphs = spec['graphs']

                    tr_idx_k = np.where(np.isin(W_SID, tr_s))[0]
                    if rep == 0 and name == INTERPRET_MODEL:
                        # Capture the loss curve
                        # for every fold of repeat 0 of the interpretability model.
                        m, history = fit_with_history(spec['make'], graphs, tr_idx_k, SEED + rep * 100 + k)
                        torch.save(m.state_dict(), model_ckpt_path(name, spec, rep, k))
                        pd.DataFrame(history).assign(fold=k + 1).to_csv(
                            f"{OUT_DIR}/{stem}_rep0_fold{k+1}_loss_curve.csv", index=False)
                    else:
                        m = fit(spec['make'], graphs, tr_idx_k, SEED + rep * 100 + k)
                    sid, p = predict(m, graphs, np.where(np.isin(W_SID, te_s))[0])
                    r = metrics(SUB_LAB[sid], p)
                    r.update(model=name, repeat=rep, fold=k + 1, n_train=len(tr_s), n_test=len(te_s))
                    rows.append(r)
                    orows += [dict(model=name, repeat=rep, sid=int(s), p=float(pp), y=int(SUB_LAB[s]))
                              for s, pp in zip(sid, p)]
                    print(f"  [{name}] rep {rep+1}/{n_rep} fold {k+1}/{FOLDS} "
                          f"acc={r['acc']*100:.1f}% auc={r['auc']:.3f} ({time.time()-t0:.0f}s)")
                pd.DataFrame(orows).to_csv(fo, index=False)
                pd.DataFrame(rows).to_csv(ff, index=False)
                if k_log:
                    pd.DataFrame(k_log).to_csv(f"{OUT_DIR}/{stem}_fold_matched_k.csv", index=False)

    # One-at-a-time hyperparameter sensitivity
    # sweep for the primary GraphODE model. Uses a SINGLE split (repeat 0,
    # fold 0) rather than the full 50-fold CV, since the point is to show the
    # result is not knife-edge-sensitive to these choices, not to re-derive
    # the main table for every combination.
    def run_hyperparam_sensitivity(graphs, splits, repeat=0):
        """One-at-a-time sensitivity of GraphODE to hidden size, learning rate
        and epoch budget, on ALL FOLDS of one repeat (same splits as the main
        analysis). The default setting is re-run inside the sweep with the same
        procedure, so every setting is compared with it on the same folds
        (corrected paired test, Nadeau-Bengio). Saves the per-fold results,
        including the number of epochs completed, and a summary table."""
        configs = [dict(name="default", hidden=HIDDEN, lr=LR, epochs=MAX_EPOCHS),
                   dict(name="hidden=32", hidden=32, lr=LR, epochs=MAX_EPOCHS),
                   dict(name="hidden=128", hidden=128, lr=LR, epochs=MAX_EPOCHS),
                   dict(name="lr=0.0005", hidden=HIDDEN, lr=5e-4, epochs=MAX_EPOCHS),
                   dict(name="lr=0.002", hidden=HIDDEN, lr=2e-3, epochs=MAX_EPOCHS),
                   dict(name="max_epochs=15", hidden=HIDDEN, lr=LR, epochs=15),
                   dict(name="max_epochs=50", hidden=HIDDEN, lr=LR, epochs=50)]
        rows = []
        for cfg in configs:
            for fold in range(FOLDS):
                tr_s, te_s = splits[repeat * FOLDS + fold]
                tr_idx = np.where(np.isin(W_SID, tr_s))[0]
                te_idx = np.where(np.isin(W_SID, te_s))[0]
                torch.manual_seed(SEED + fold); np.random.seed(SEED + fold)
                subs = np.unique(W_SID[tr_idx]); labs = SUB_LAB[subs]
                tr_sub, va_sub = train_test_split(subs, test_size=0.15, stratify=labs, random_state=SEED + fold)
                tr_g = subset(graphs, tr_idx[np.isin(W_SID[tr_idx], tr_sub)])
                va_g = subset(graphs, tr_idx[np.isin(W_SID[tr_idx], va_sub)])
                tl = DataLoader(tr_g, batch_size=BATCH, shuffle=True)
                vl = DataLoader(va_g, batch_size=128)
                m = GraphODE(h=cfg['hidden'], p=DROPOUT, edge_dim=1).to(dev)
                opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=WEIGHT_DECAY)
                best, bad, state, n_ep = 1e9, 0, None, 0
                for ep in range(cfg['epochs']):
                    m.train()
                    for b in tl:
                        b = b.to(dev); opt.zero_grad()
                        F.cross_entropy(m(b), b.y).backward()
                        nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                    n_ep = ep + 1
                    vl_loss = val_loss(m, vl)
                    if vl_loss < best - 1e-4:
                        best, bad, state = vl_loss, 0, {kk: vv.clone() for kk, vv in m.state_dict().items()}
                    else:
                        bad += 1
                        if bad >= PATIENCE:
                            break
                m.load_state_dict(state)
                sid, pr = predict(m, graphs, te_idx)
                r = metrics(SUB_LAB[sid], pr)
                rows.append({'setting': cfg['name'], 'repeat': repeat, 'fold': fold + 1,
                             'acc': r['acc'], 'auc': r['auc'], 'epochs_completed': n_ep,
                             'n_train': len(tr_s), 'n_test': len(te_s)})
                print(f"  [hyperparam sweep] {cfg['name']} fold {fold+1}/{FOLDS}: "
                      f"acc={r['acc']*100:.1f}% auc={r['auc']:.3f} epochs={n_ep}")
        folds_df = pd.DataFrame(rows)
        folds_df.to_csv(f"{OUT_DIR}/hyperparam_sensitivity_folds.csv", index=False)
        base = folds_df[folds_df.setting == "default"].sort_values('fold')
        summ = []
        for cfg in configs:
            g = folds_df[folds_df.setting == cfg['name']].sort_values('fold')
            row = {'setting': cfg['name'],
                   'fold_acc_mean_%': g.acc.mean() * 100, 'fold_acc_sd_%': g.acc.std(ddof=1) * 100,
                   'mean_auc': g.auc.mean(), 'mean_epochs_completed': g.epochs_completed.mean(),
                   'min_epochs': int(g.epochs_completed.min()), 'max_epochs_completed': int(g.epochs_completed.max())}
            if cfg['name'] != "default":
                _, p_nb = nb_corrected_ttest(g.acc.values, base.acc.values,
                                             base.n_train.mean(), base.n_test.mean())
                row['diff_vs_default_pp'] = (g.acc.values - base.acc.values).mean() * 100
                row['p_corrected_paired'] = p_nb
            summ.append(row)
        summ_df = pd.DataFrame(summ)
        summ_df.to_csv(f"{OUT_DIR}/hyperparam_sensitivity.csv", index=False)
        return summ_df

    # Label-permutation test extended to a deep
    # model. Uses only repeat 0's FOLDS folds (not the full 50) and a much
    # smaller B (default 20 vs >=1000 for the classical baselines) because
    # every permutation here means retraining a GNN FOLDS times from scratch
    # -- an explicit, stated compute trade-off, not a silent shortcut.
    def permutation_pvalue_deep(make_model, graphs, y_sub, w_sid, splits_rep0, obs_acc, B, n_folds=FOLDS):
        """n_folds caps how many of repeat 0's folds are used per permutation
        (default: all FOLDS; a time-constrained rerun passes a smaller number
        via N_PERM_DEEP_FOLDS, since each fold here means retraining a GNN
        from scratch). obs_acc should be computed over the SAME subset of
        folds for a fair comparison -- the caller does this."""
        rng = np.random.RandomState(SEED); null = []
        # Resumable: every finished permutation is appended to a small CSV in
        # OUT_DIR. If the session restarts, finished permutations are re-read
        # and skipped (the random generator is advanced identically), so the
        # same permutations are used as in an uninterrupted run.
        # N_PERM_DEEP_START lets a second notebook compute permutations
        # START+1..B while another run covers 1..START. Permutation number b
        # always uses the same labels and the same training seeds, so parts
        # can be combined.
        start = int(os.environ.get("N_PERM_DEEP_START", 0))
        null_path = f"{OUT_DIR}/deep_perm_null.csv" if start == 0 else f"{OUT_DIR}/deep_perm_null_from{start + 1}.csv"
        if os.path.exists(null_path):
            try:
                prev = pd.read_csv(null_path)
                null = [float(v) for v in prev["null_acc"].tolist()][:max(B - start, 0)]
                print(f"  [deep permutation] resuming: {len(null)}/{B} permutations already done")
            except Exception:
                null = []
        for b in range(B):
            yp = rng.permutation(y_sub)
            if b < start + len(null):
                continue
            accs = []
            for k in range(n_folds):
                tr_s, te_s = splits_rep0[k]
                tr_idx = np.where(np.isin(w_sid, tr_s))[0]
                te_idx = np.where(np.isin(w_sid, te_s))[0]
                m = fit_permuted(make_model, graphs, tr_idx, yp, w_sid, SEED + 50000 + b * 100 + k)
                sid, p = predict(m, graphs, te_idx)
                accs.append(((p >= 0.5) == yp[sid]).mean())
            null.append(np.mean(accs))
            pd.DataFrame({"perm": range(start + 1, start + len(null) + 1), "null_acc": null}).to_csv(null_path, index=False)
            print(f"  [deep permutation] {b+1}/{B} null acc={null[-1]*100:.1f}%")
        null = np.array(null)
        return (1 + np.sum(null >= obs_acc)) / (B + 1), float(null.mean())

    # ---------------------------------------------------------------
    # INTERPRETABILITY (illustrative; ONE model x repeat 0 x all folds)
    # ---------------------------------------------------------------
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import pdist
    from scipy.special import gamma as gamma_fn, digamma
    from scipy.stats import mannwhitneyu

    def estimate_divergence_rate(trajectory, dt):
        steps, n_nodes, _ = trajectory.shape
        if steps < 3:
            return np.nan
        div = np.array([np.log(np.clip(np.linalg.norm(trajectory[t] - trajectory[t - 1], axis=-1), 1e-10, None))
                        for t in range(1, steps)])
        time_axis = np.arange(1, steps) * dt
        return float(np.mean([np.polyfit(time_axis, div[:, n], 1)[0] for n in range(n_nodes)]))

    def estimate_switching_rate(trajectory):
        steps, n_nodes, h_dim = trajectory.shape
        if steps < 3:
            return np.nan
        flat = trajectory.reshape(steps * n_nodes, h_dim)
        try:
            _, _, vt = np.linalg.svd(flat - flat.mean(0, keepdims=True), full_matrices=False)
            pc1 = vt[0]
        except np.linalg.LinAlgError:
            return np.nan
        rates = []
        for node in range(n_nodes):
            signs = np.sign(np.diff(trajectory[:, node, :] @ pc1)); signs = signs[signs != 0]
            rates.append(0.0 if len(signs) < 2 else np.sum(signs[1:] != signs[:-1]) / (len(signs) - 1))
        return float(np.mean(rates))

    def cliffs_delta(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(x) * len(y) == 0:
            return np.nan
        return float((np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :])) / (len(x) * len(y)))

    def run_interpretability(model_name, MODELS, graphs, splits, repeat=0):
        spec = MODELS[model_name]; make_model = spec['make']
        print(f"\n[Interpretability] {model_name} repeat={repeat}: loading/training {FOLDS} fold models...")
        fold_models = []
        for k in range(FOLDS):
            tr_s, te_s = splits[repeat * FOLDS + k]
            path = model_ckpt_path(model_name, spec, repeat, k)
            if os.path.exists(path):
                sd = torch.load(path, map_location=dev)
                print(f"  fold {k+1}/{FOLDS} loaded from checkpoint")
            else:
                m = fit(make_model, graphs, np.where(np.isin(W_SID, tr_s))[0], SEED + repeat * 100 + k)
                sd = {kk: vv.clone() for kk, vv in m.state_dict().items()}
                print(f"  fold {k+1}/{FOLDS} trained")
            fold_models.append({'state_dict': sd, 'val_subjects': set(te_s)})

        def load_fold(fm):
            m = make_model().to(dev); m.load_state_dict(fm['state_dict']); m.eval(); return m

        records = []
        for fm in fold_models:
            m = load_fold(fm)
            val_graphs = subset(graphs, np.where(np.isin(W_SID, list(fm['val_subjects'])))[0])
            if not val_graphs:
                continue
            with torch.no_grad():
                for b in DataLoader(val_graphs, batch_size=32):
                    b = b.to(dev)
                    traj = m(b, return_trajectory=True).cpu().numpy()
                    batch_idx = b.batch.cpu().numpy(); labels = b.y.cpu().numpy()
                    sids = b.sid.cpu().numpy()
                    for g_idx in range(len(labels)):
                        st = traj[:, batch_idx == g_idx, :]
                        if st.shape[1] == 0:
                            continue
                        records.append({
                            'sid': int(sids[g_idx]), 'label': int(labels[g_idx]),
                            'divergence_rate': estimate_divergence_rate(st, ODE_STEP),
                            'switching_rate': estimate_switching_rate(st)})
        bdf = pd.DataFrame(records).groupby(['sid', 'label'], as_index=False).mean(numeric_only=True)
        adhd_g, ctrl_g = bdf[bdf.label == 1], bdf[bdf.label == 0]
        rows = []
        for c in BIOMARKER_COLS:
            a, b = adhd_g[c].dropna().values, ctrl_g[c].dropna().values
            if len(a) < 3 or len(b) < 3:
                rows.append({'biomarker': c, 'p_raw': np.nan, 'cliffs_delta': np.nan}); continue
            _, p = mannwhitneyu(a, b, alternative='two-sided')
            rows.append({'biomarker': c, 'p_raw': p, 'cliffs_delta': cliffs_delta(a, b),
                         'n_adhd': len(a), 'n_ctrl': len(b)})
        bio_stats = pd.DataFrame(rows)
        bio_stats['p_fdr_bh'] = benjamini_hochberg(bio_stats['p_raw'].values)
        bio_stats.to_csv(f"{OUT_DIR}/biomarker_group_stats.csv", index=False)
        print(f"\n[Biomarkers] ADHD (n={len(adhd_g)}) vs Control (n={len(ctrl_g)}), out-of-fold "
              f"(descriptive):")
        print(bio_stats.to_string(index=False))

        attn_sum = {0: np.zeros((19, 19)), 1: np.zeros((19, 19))}
        attn_cnt = {0: np.zeros((19, 19)), 1: np.zeros((19, 19))}
        for fm in fold_models:
            m = load_fold(fm)
            val_graphs = subset(graphs, np.where(np.isin(W_SID, list(fm['val_subjects'])))[0])
            if not val_graphs:
                continue
            with torch.no_grad():
                for b in DataLoader(val_graphs, batch_size=32):
                    b = b.to(dev)
                    _, edge_idx, alpha = m(b, return_att=True)
                    ei = edge_idx.cpu().numpy()
                    bidx = b.batch.cpu().numpy(); labels = b.y.cpu().numpy()
                    keep = ei[0] != ei[1]
                    src, dst = ei[0][keep] % 19, ei[1][keep] % 19
                    edge_graph = bidx[ei[0][keep]]
                    edge_label = labels[edge_graph]
                    w = alpha.mean(dim=-1).cpu().numpy()[keep]
                    for lbl in (0, 1):
                        sel = edge_label == lbl
                        np.add.at(attn_sum[lbl], (src[sel], dst[sel]), w[sel])
                        np.add.at(attn_cnt[lbl], (src[sel], dst[sel]), 1)
        mat_adhd = np.divide(attn_sum[1], attn_cnt[1], out=np.zeros((19, 19)), where=attn_cnt[1] != 0)
        mat_ctrl = np.divide(attn_sum[0], attn_cnt[0], out=np.zeros((19, 19)), where=attn_cnt[0] != 0)
        both = (attn_cnt[1] > 0) & (attn_cnt[0] > 0)
        diff = np.where(both, np.abs(mat_adhd - mat_ctrl), 0.0)
        importance = diff.sum(axis=1) + diff.sum(axis=0)
        order = np.argsort(importance)[::-1]
        importance_df = pd.DataFrame({'channel_idx': order, 'importance_score': importance[order]})
        importance_df.to_csv(f"{OUT_DIR}/channel_importance.csv", index=False)
        print(f"\n[Attention] Top-5 electrode indices (0-18): {order[:5].tolist()} "
              f"(scores: {np.round(importance[order[:5]], 3).tolist()})")

        print("\n[t-SNE] latent ODE trajectory embedding (fold-0 held-out subjects only)...")
        from sklearn.manifold import TSNE
        tsne_result = None
        try:
            m = load_fold(fold_models[0])
            val_idx0 = np.where(np.isin(W_SID, list(fold_models[0]['val_subjects'])))[0]
            rng = np.random.RandomState(SEED)
            sample_idx = rng.choice(val_idx0, size=min(48, len(val_idx0)), replace=False)
            all_traj, all_labels = [], []
            with torch.no_grad():
                for b in DataLoader(subset(graphs, sample_idx), batch_size=8):
                    b = b.to(dev)
                    all_traj.append(m(b, return_trajectory=True).cpu().numpy())
                    all_labels.append(b.y.cpu().numpy()[b.batch.cpu().numpy()])
            concat_traj = np.concatenate(all_traj, axis=1)
            node_labels = np.concatenate(all_labels)
            steps, total_nodes, h_dim = concat_traj.shape
            flat = concat_traj.reshape(steps * total_nodes, h_dim)
            perplexity = min(30, max(5, flat.shape[0] // 3))
            emb = TSNE(n_components=2, random_state=SEED, perplexity=perplexity).fit_transform(flat)
            tsne_result = {'embeddings': emb.reshape(steps, total_nodes, 2),
                           'node_labels': node_labels, 'n_steps': steps}
            print(f"[t-SNE] Done -- {steps} trajectory points x {total_nodes} nodes.")
        except Exception as e:
            print(f"[t-SNE] Skipped due to error: {e}")

        return {'bio_stats': bio_stats, 'bdf': bdf, 'importance_df': importance_df,
                'mat_adhd': mat_adhd, 'mat_ctrl': mat_ctrl, 'tsne': tsne_result}


def load_deep_results(MODELS):
    fl, ol = [], []
    if MODELS:
        for name, spec in MODELS.items():
            for rep in range(REPEATS):
                stem = deep_stem(name, spec, rep)
                ff, fo = f"{OUT_DIR}/{stem}_folds.csv", f"{OUT_DIR}/{stem}_oof.csv"
                if os.path.exists(ff) and os.path.exists(fo):
                    fl.append(pd.read_csv(ff)); ol.append(pd.read_csv(fo))
    else:
        for ff in sorted(glob.glob(f"{OUT_DIR}/deep_*_{CFG_TAG}*_folds.csv")):
            fo = ff.replace("_folds.csv", "_oof.csv")
            if os.path.exists(fo):
                fl.append(pd.read_csv(ff)); ol.append(pd.read_csv(fo))
    if not fl:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(fl, ignore_index=True), pd.concat(ol, ignore_index=True)


# ----------------------------- REPORT ---------------------------------
def summarize(folds, oof, ref_model=None):
    order = list(dict.fromkeys(folds.model))
    ref = folds[folds.model == ref_model] if ref_model in order else None
    rows, raw_pvals = [], []
    for name in order:
        g = folds[folds.model == name]; o = oof[oof.model == name]
        rep_acc = [((og.p >= .5) == og.y).mean() for _, og in o.groupby("repeat")]
        rep_auc = [roc_auc_score(og.y, og.p) for _, og in o.groupby("repeat")]
        avg = o.groupby("sid").agg(p=("p", "mean"), y=("y", "first"))
        (a_lo, a_hi), _ = boot_ci(avg.y.values, avg.p.values)
        row = {"model": name, "repeats": len(rep_acc),
               "fold_acc %": f"{g.acc.mean()*100:.1f} +/- {g.acc.std()*100:.1f}",
               "pooled_acc %": f"{np.mean(rep_acc)*100:.1f} +/- {np.std(rep_acc)*100:.1f}",
               "pooled_auc": f"{np.mean(rep_auc):.3f} +/- {np.std(rep_auc):.3f}",
               "acc_CI95 %": f"[{a_lo*100:.1f}, {a_hi*100:.1f}]",
               "sens %": f"{g.sens.mean()*100:.1f}", "spec %": f"{g.spec.mean()*100:.1f}",
               "mcc": f"{g.mcc.mean():.3f}"}
        if ref is not None and name != ref_model:
            m = g.merge(ref, on=["repeat", "fold"], suffixes=("", "_ref"))
            _, p = nb_corrected_ttest(m.acc, m.acc_ref, m.n_train.mean(), m.n_test.mean())
            row[f"p_vs_{ref_model}"] = f"{p:.3f}"
            raw_pvals.append(p)
        else:
            raw_pvals.append(np.nan)
        rows.append(row)
    if ref is not None:
        fdr = benjamini_hochberg(np.array(raw_pvals))
        for i, row in enumerate(rows):
            if not np.isnan(raw_pvals[i]):
                row[f"p_vs_{ref_model}_fdr"] = f"{fdr[i]:.3f}"
    return pd.DataFrame(rows)


# v8: fixed-alpha (0.5) late-fusion ensemble. alpha is NOT tuned against these
# results (that would leak); it's the naive, defensible default. Reported
# separately -- not injected into the NB-corrected significance table, since
# no fold-id is available for this post-hoc probability blend.
def report_ensemble(oof, name_a, name_b, alpha=0.5):
    if name_a not in set(oof.model) or name_b not in set(oof.model):
        return None
    a = oof[oof.model == name_a].groupby(["repeat", "sid"])["p"].mean().rename("p_a")
    b = oof[oof.model == name_b].groupby(["repeat", "sid"])["p"].mean().rename("p_b")
    y = oof[oof.model == name_a].groupby(["repeat", "sid"])["y"].first()
    m = pd.concat([a, b, y], axis=1).dropna()
    m["p"] = alpha * m.p_a + (1 - alpha) * m.p_b
    rep_acc = [((g.p >= .5) == g.y).mean() for _, g in m.groupby(level="repeat")]
    rep_auc = [roc_auc_score(g.y, g.p) for _, g in m.groupby(level="repeat") if len(set(g.y)) > 1]
    avg = m.groupby(level="sid").agg(p=("p", "mean"), y=("y", "first"))
    (a_lo, a_hi), (auc_lo, auc_hi) = boot_ci(avg.y.values, avg.p.values)
    print(f"\n=== Ensemble (fixed alpha={alpha:.2f}, NOT tuned on these results): "
          f"{name_a} + {name_b} ===")
    print(f"  pooled_acc % = {np.mean(rep_acc)*100:.1f} +/- {np.std(rep_acc)*100:.1f}   "
          f"acc_CI95 % = [{a_lo*100:.1f}, {a_hi*100:.1f}]")
    if rep_auc:
        print(f"  pooled_auc   = {np.mean(rep_auc):.3f} +/- {np.std(rep_auc):.3f}   "
              f"auc_CI95 = [{auc_lo:.3f}, {auc_hi:.3f}]")
    print(f"  NOTE: this is descriptive only. A defensible significance claim for the\n"
          f"  ensemble vs either component would need alpha selected via nested CV\n"
          f"  (tune alpha on inner folds, evaluate on outer folds), not printed here.")
    return {"alpha": alpha, "pooled_acc_mean": float(np.mean(rep_acc)), "pooled_acc_std": float(np.std(rep_acc)),
            "acc_ci95": [float(a_lo * 100), float(a_hi * 100)]}


# ------------------------------ MAIN ----------------------------------
T0 = time.time()
files = find_files()
print(f"ADHD files: {sum(l == 1 for _, l in files)} | Control files: {sum(l == 0 for _, l in files)}")
assert len(files) > 0, "No .mat files found -- check DATA_DIR"

X, W_SID, SUB_LAB, reject_log = load_windows(
    files, artifact_thresh=(ARTIFACT_UV_THRESH if RUN_ARTIFACT_REJECTION else None))
reject_log.to_csv(f"{OUT_DIR}/artifact_rejection_log.csv", index=False)
n_rej_total = int(reject_log.n_rejected.sum())
n_kept_total = int(reject_log.n_kept.sum())
print(f"[artifact rejection] threshold="
      f"{ARTIFACT_UV_THRESH if RUN_ARTIFACT_REJECTION else 'OFF'} uV | kept {n_kept_total} windows | "
      f"rejected {n_rej_total} windows "
      f"({100*n_rej_total/max(n_rej_total+n_kept_total,1):.1f}%) across {len(reject_log)} subjects "
      f"(per-subject detail in artifact_rejection_log.csv)")
if RUN_ICA:
    print(f"[ICA ocular correction] z>{ICA_Z_THRESH}, max {ICA_MAX_REMOVE} components/subject | "
          f"subjects with >=1 component removed: {int((reject_log.n_ic_removed > 0).sum())}/{len(reject_log)} | "
          f"mean removed per subject: {reject_log.n_ic_removed.mean():.2f}")
assert (reject_log.n_kept > 0).all(), (
    "At least one subject lost ALL windows to artifact rejection -- lower ARTIFACT_UV_THRESH "
    "(see artifact_rejection_log.csv for which subject/threshold).")

if EXCLUDE_FRONTAL:
    # Fp1/Fp2 sensitivity arm. Run this as a
    # SEPARATE invocation (EXCLUDE_FRONTAL=1, a different OUT_DIR) and compare
    # its final_table.csv row-by-row against the main (19-channel) run; this
    # is not merged into the same table because dropping channels changes the
    # whole feature/graph pipeline, not just one model.
    X = np.delete(X, FRONTAL_IDX, axis=1)
    print(f"[Fp1/Fp2 sensitivity run] EXCLUDE_FRONTAL=1 -- dropped channels {FRONTAL_IDX}, "
          f"{X.shape[1]} channels remain")
    # the interpretability/attention-map code below hardcodes the 19-channel
    # montage (electrode names, 19x19 attention matrices); it isn't needed for
    # this sensitivity check (only the accuracy numbers in final_table.csv
    # are compared against the main 19-channel run), so skip it here.
    RUN_INTERPRETABILITY = False

n_sub = len(SUB_LAB)
print(f"{n_sub} subjects ({SUB_LAB.sum()} ADHD / {(1-SUB_LAB).sum()} control), {len(X)} windows "
      f"({time.time()-T0:.0f}s) | {REPEATS} repeats x {FOLDS} folds | deep={RUN_DEEP} "
      f"| multiband={RUN_MULTIBAND} | degree_matched={RUN_DEGREE_MATCHED} | cfg={CFG_TAG}")
print(f"  [v8 config] graph_method={GRAPH_METHOD} wpli_threshold={WPLI_THRESHOLD} "
      f"node_handcrafted={NODE_HANDCRAFTED} agg_method={AGG_METHOD} "
      f"weight_decay={WEIGHT_DECAY} dropout={DROPOUT} ode_step={ODE_STEP} "
      f"exclude_frontal={EXCLUDE_FRONTAL}")

print("computing standard wPLI (broadband + 5 bands, one FFT pass)...")
WB = wpli_all(X)
W = WB.pop("broad")
band_matrices = WB
BP = band_power_batch(X)
NODE_FEAT = compute_node_handcrafted(BP, W) if NODE_HANDCRAFTED else None
FEATS = subject_features(W, BP, W_SID, n_sub)
splits = make_splits(SUB_LAB)
print(f"features ready ({time.time()-T0:.0f}s)")

if RUN_WINDOW_SPLIT_COMPARISON:
    print("\n=== Window-trained SVM[wPLI]: subject-disjoint vs random-window splits ===")
    _WF = window_features(W, BP)["wPLI"]
    split_cmp, split_rep, split_fold = run_window_split_comparison(_WF, SUB_LAB, W_SID, splits)
    split_cmp.to_csv(f"{OUT_DIR}/window_split_comparison.csv", index=False)
    split_rep.to_csv(f"{OUT_DIR}/window_split_comparison_by_repeat.csv", index=False)
    split_fold.to_csv(f"{OUT_DIR}/window_split_comparison_by_fold.csv", index=False)
    print(split_cmp.round(3).to_string(index=False))
    if os.environ.get("ONLY_WINDOW_SPLIT_COMPARISON", "0") == "1":
        print("ONLY_WINDOW_SPLIT_COMPARISON=1: stopping after this analysis.")
        raise SystemExit(0)

print("\n=== Baselines ===")
b_folds, b_oof = run_baselines(FEATS, SUB_LAB, splits)
perm = {}
for fname, Fm in FEATS.items():
    for mname, mk in BASELINES.items():
        name = f"{mname}[{fname}]"
        obs = b_folds[(b_folds.model == name) & (b_folds.repeat == 0)].acc.mean()
        perm[name] = permutation_pvalue(Fm, SUB_LAB, mk, obs, N_PERM)
print(summarize(b_folds, b_oof).to_string(index=False))
print(f"Permutation test (baselines; obs = repeat-0 CV acc; B={N_PERM}):")
for k, (pv, nm) in perm.items():
    print(f"  {k:<18} p={pv:.4f}  (null mean acc={nm*100:.1f}%)")

if RUN_WINDOW_TRAINED_SVM:
    print("\n=== Window-trained baselines ===")
    WIN_FEATS = window_features(W, BP)
    w_folds, w_oof = run_window_trained_baselines(WIN_FEATS, SUB_LAB, W_SID, splits)
    b_folds = pd.concat([b_folds, w_folds], ignore_index=True)
    b_oof = pd.concat([b_oof, w_oof], ignore_index=True)
    print(summarize(w_folds, w_oof).to_string(index=False))
    if N_PERM_WINDOW > 0:
        obs_w = w_folds[(w_folds.model == "SVM[wPLI]-window-trained") & (w_folds.repeat == 0)].acc.mean()
        perm["SVM[wPLI]-window-trained"] = permutation_pvalue_window(
            WIN_FEATS["wPLI"], SUB_LAB, W_SID, make_svm_window, obs_w, N_PERM_WINDOW)
        print(f"Permutation test (SVM[wPLI]-window-trained; obs = repeat-0 CV acc; "
              f"B={N_PERM_WINDOW}, smaller than N_PERM because training uses window-level rows):")
        pv, nm = perm["SVM[wPLI]-window-trained"]
        print(f"  SVM[wPLI]-window-trained p={pv:.4f}  (null mean acc={nm*100:.1f}%)")
    else:
        print("Permutation test for SVM[wPLI]-window-trained skipped this run (N_PERM_WINDOW=0, "
              "time budget) -- the formal NB-corrected fold comparison above still applies.")

MODELS = {}
matched_k = None; degree_log = None; mb_avg_degree = None
if RUN_DEEP:
    print("\n=== Building graph representations ===")
    G_single = make_graphs(X, SUB_LAB, W_SID, W, K_NEIGHBOURS, node_feat=NODE_FEAT)
    MODELS["GraphODE"] = {'make': lambda: GraphODE(edge_dim=1), 'graphs': G_single}
    if RUN_STACKEDGAT:
        MODELS["StackedGAT"] = {'make': lambda: StackedGAT(edge_dim=1), 'graphs': G_single}
    print(f"single-band graphs ready ({time.time()-T0:.0f}s)")

    if RUN_MULTIBAND:
        G_multiband = make_multiband_graphs(X, SUB_LAB, W_SID, band_matrices, K_NEIGHBOURS, node_feat=NODE_FEAT)
        MODELS["GraphODE-MultiBand"] = {'make': lambda: GraphODE(edge_dim=N_BANDS), 'graphs': G_multiband}
        print(f"multi-band graphs ready ({time.time()-T0:.0f}s)")

        # Whole-dataset average degree / k*, reported DESCRIPTIVELY ONLY (Figure 5 and
        # the JSON summary below). This is NOT the value used to evaluate the
        # DegreeMatched arm -- that is selected separately inside each outer training
        # fold (train subjects only; see run_deep's 'degree_matched_per_fold' branch),
        # since the original single, whole-dataset selection
        # used test-subject signal to choose a graph-construction parameter.
        mb_avg_degree = avg_degree_from_mask(union_mask_multiband(band_matrices, K_NEIGHBOURS))
        sb_avg_degree = avg_degree_from_mask(topk_mask_batch(W, K_NEIGHBOURS))
        matched_k, degree_log = find_degree_matched_k(W, target_degree=mb_avg_degree)
        print(f"Average node degree -> single-band (k={K_NEIGHBOURS}): {sb_avg_degree:.2f} | "
              f"multi-band (fused): {mb_avg_degree:.2f} | whole-dataset descriptive k*={matched_k} "
              f"(descriptive only -- NOT used below; see per-fold matched_k log)")

        if RUN_DEGREE_MATCHED:
            MODELS["GraphODE-DegreeMatched"] = {'make': lambda: GraphODE(edge_dim=1),
                                                'graphs': None,
                                                'degree_matched_per_fold': True,
                                                'extra': "_foldmatched"}
            print("degree-matched arm registered: matched_k is selected inside run_deep(), "
                  "separately for each of the 50 outer training folds (train subjects only)")

    # Fully-connected and random-topology
    # controls, both at the SAME average degree family as the single-band
    # graph (k neighbours), so any difference vs. GraphODE isolates the
    # effect of edge SELECTION (rank-based top-k) rather than just having
    # some graph structure at all.
    if RUN_FULLY_CONNECTED:
        G_fc = make_fully_connected_graphs(X, SUB_LAB, W_SID, W, node_feat=NODE_FEAT)
        MODELS["GraphODE-FullyConnected"] = {'make': lambda: GraphODE(edge_dim=1), 'graphs': G_fc,
                                             'repeats': REPEATS_CONTROL}
        print(f"fully-connected graphs ready ({time.time()-T0:.0f}s)")
    if RUN_RANDOM_GRAPH:
        G_rand = make_random_graphs(X, SUB_LAB, W_SID, W, K_NEIGHBOURS, node_feat=NODE_FEAT)
        MODELS["GraphODE-RandomGraph"] = {'make': lambda: GraphODE(edge_dim=1), 'graphs': G_rand,
                                          'repeats': REPEATS_CONTROL}
        print(f"random-topology graphs ready ({time.time()-T0:.0f}s)")

    # Non-graph deep-learning baselines, reusing
    # the single-band Data objects (their edges are simply unused by these
    # two architectures).
    if RUN_CNN_BASELINES:
        MODELS["CNN-Only"] = {'make': lambda: CNNOnly(), 'graphs': G_single, 'repeats': REPEATS_CONTROL}
        MODELS["CNN-LSTM"] = {'make': lambda: CNNLSTM(), 'graphs': G_single, 'repeats': REPEATS_CONTROL}
        print("CNN-only and CNN-LSTM baselines registered (reuse single-band graphs; edges unused)")

    X_ALIVE = X   # kept alive (not del'd yet) so run_deep() can build per-fold DegreeMatched graphs
    print(f"\n=== Deep models (resumable) -- {len(MODELS)} arms, primary models x {REPEATS} repeats, "
          f"control/ablation arms x {REPEATS_CONTROL} repeat(s), all x {FOLDS} folds | ODE_STEP={ODE_STEP} ===")
    if os.environ.get("ONLY_HYPERPARAM_SWEEP", "0") == "1":
        print("\n=== Hyperparameter sensitivity sweep only (repeat 0, all folds) ===")
        print(run_hyperparam_sensitivity(G_single, splits).to_string(index=False))
        print("ONLY_HYPERPARAM_SWEEP=1: stopping after this analysis.")
        raise SystemExit(0)
    run_deep(splits, MODELS)

    if RUN_HYPERPARAM_SWEEP:
        print("\n=== Hyperparameter sensitivity sweep "
              "(repeat 0, all folds -- see run_hyperparam_sensitivity docstring) ===")
        hp_df = run_hyperparam_sensitivity(G_single, splits)
        print(hp_df.to_string(index=False))

    del X, X_ALIVE
d_folds, d_oof = load_deep_results(MODELS)

ref_candidate = "GraphODE" if "GraphODE" in MODELS else None
if RUN_DEEP and RUN_PERM_DEEP and ref_candidate:
    # Permutation test extended to a deep model
    # (repeat 0 only, reduced B -- see permutation_pvalue_deep docstring).
    # observed accuracy is averaged over the SAME first N_PERM_DEEP_FOLDS folds
    # of repeat 0 that the null distribution below uses, for a fair comparison.
    obs_acc_deep = d_folds[(d_folds.model == ref_candidate) & (d_folds.repeat == 0) &
                           (d_folds.fold <= N_PERM_DEEP_FOLDS)].acc.mean()
    splits_rep0 = splits[0:FOLDS]
    print(f"\n=== Deep-model permutation test: {ref_candidate} "
          f"(first {N_PERM_DEEP_FOLDS}/{FOLDS} folds of repeat 0 only, time budget) ===")
    pv_deep, nm_deep = permutation_pvalue_deep(
        MODELS[ref_candidate]['make'], MODELS[ref_candidate]['graphs'], SUB_LAB, W_SID,
        splits_rep0, obs_acc_deep, N_PERM_DEEP, n_folds=N_PERM_DEEP_FOLDS)
    perm[ref_candidate] = (pv_deep, nm_deep)
    print(f"  {ref_candidate} p={pv_deep:.4f} (null mean acc={nm_deep*100:.1f}%, B={N_PERM_DEEP}, "
          f"{N_PERM_DEEP_FOLDS} folds of repeat 0 only -- see permutation_pvalue_deep docstring "
          f"for the compute-cost rationale)")

folds = pd.concat([d_folds, b_folds], ignore_index=True)
oof = pd.concat([d_oof, b_oof], ignore_index=True)
ref = "GraphODE" if "GraphODE" in set(folds.model) else None
table = summarize(folds, oof, ref_model=ref)
table["perm_p"] = table.model.map(lambda m: f"{perm[m][0]:.4f}" if m in perm else "-")

print("\n" + "=" * 70 + "\nFINAL TABLE (subject-level, repeated CV, all subjects)\n" + "=" * 70)
print(table.to_string(index=False))
print("\nfold_acc: mean+/-std over repeat x fold | pooled_*: each subject predicted once per repeat, "
      "mean+/-std over repeats | acc_CI95: subject bootstrap on repeat-averaged probs | "
      "p_vs_*: Nadeau-Bengio corrected paired t-test | p_vs_*_fdr: Benjamini-Hochberg across all "
      f"model-vs-reference comparisons (use THIS column) | perm_p: label permutation ({N_PERM} perms)")

cmp_res = None
if "GraphODE-MultiBand" in set(folds.model) and "GraphODE-DegreeMatched" in set(folds.model):
    cmp_res = paired_comparison(folds, "GraphODE-MultiBand", "GraphODE-DegreeMatched")
    print("\n" + "=" * 70)
    print("NOVELTY TEST: Multi-band fused vs degree-matched single-band control")
    print("=" * 70)
    print("(Density-controlled: isolates whether band-resolved information helps, separately\n"
          " from multi-band's graph simply being denser.)")
    print(f"Mean accuracy difference (multi-band - degree-matched): {cmp_res['mean_diff']*100:+.2f} pp")
    print(f"Nadeau-Bengio corrected paired t-test p = {cmp_res['p_nb']:.4f} "
          f"{'(significant, p<0.05)' if cmp_res['p_nb'] < 0.05 else '(NOT significant)'}")
    print(f"NB-corrected 95% CI on the difference: [{cmp_res['ci'][0]*100:+.2f}, {cmp_res['ci'][1]*100:+.2f}] pp")
    print(f"Minimum detectable effect (alpha=.05, power=.80, NB SE): {cmp_res['mde']*100:.2f} pp "
          f"-- true differences smaller than this were unlikely to be detected by this design "
          f"({n_sub} subjects, {REPEATS}x{FOLDS} repeated CV).")
    print(f"TOST equivalence test (margin +/-{cmp_res['margin']*100:.1f} pp): p = {cmp_res['tost_p']:.4f} "
          f"{'-> practically EQUIVALENT' if cmp_res['tost_p'] < 0.05 else '-> equivalence NOT shown (cannot claim \"no difference\")'}")
    if cmp_res['p_nb'] >= 0.05:
        print("-> Multi-band's advantage (if any) is NOT distinguishable from added graph density.\n"
              "   Report single-band as primary and multi-band as exploratory ablation; do not claim\n"
              "   multi-band superiority. 'Not significant' is NOT evidence of equivalence unless TOST passes.")
    else:
        print("-> Multi-band beats the density-matched control: band-resolved information is a genuine\n"
              "   (not merely density) contributor.")
    with open(f"{OUT_DIR}/multiband_vs_degreematched.json", "w") as f:
        json.dump({'mean_acc_diff_pp': cmp_res['mean_diff'] * 100, 'p_value_nb': cmp_res['p_nb'],
                   'ci95_nb_pp': [cmp_res['ci'][0] * 100, cmp_res['ci'][1] * 100],
                   'min_detectable_effect_pp': cmp_res['mde'] * 100,
                   'tost_p': cmp_res['tost_p'], 'equiv_margin_pp': cmp_res['margin'] * 100,
                   'degree_matched_k_whole_dataset_descriptive': matched_k,
                   'note': 'The evaluated DegreeMatched arm selects k per outer training '
                           'fold (see *_fold_matched_k.csv); this k is a whole-dataset '
                           'descriptive value only (Figure 5), not the evaluation parameter.'},
                  f, indent=2)

# v8: descriptive ensemble block (fixed alpha, not tuned on these results)
ens_result = report_ensemble(oof, "GraphODE", "SVM[wPLI]", alpha=0.5)
if ens_result is not None:
    json.dump(ens_result, open(f"{OUT_DIR}/ensemble_graphode_svm.json", "w"), indent=2)

table.to_csv(f"{OUT_DIR}/final_table.csv", index=False)
folds.to_csv(f"{OUT_DIR}/all_fold_results.csv", index=False)
oof.to_csv(f"{OUT_DIR}/all_oof_predictions.csv", index=False)
import sys, sklearn, scipy
_lib_versions = {"python": sys.version.split()[0], "numpy": np.__version__,
                  "pandas": pd.__version__, "scipy": scipy.__version__,
                  "scikit_learn": sklearn.__version__}
if HAS_TORCH:
    try:
        import torch_geometric, torchdiffeq
        _lib_versions.update({
            "torch": getattr(torch, "__version__", "unknown"),
            "torch_geometric": getattr(torch_geometric, "__version__", "unknown"),
            "torchdiffeq": getattr(torchdiffeq, "__version__", "unknown")})
    except Exception as e:
        _lib_versions["torch_stack_version_lookup_error"] = str(e)

json.dump({"cfg_tag": CFG_TAG, "repeats": REPEATS, "folds": FOLDS, "n_subjects": int(n_sub),
           "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "k_neighbours": K_NEIGHBOURS,
           "max_epochs": MAX_EPOCHS, "ode_step": ODE_STEP, "equiv_margin": EQUIV_MARGIN,
           "run_multiband": RUN_MULTIBAND, "run_degree_matched": RUN_DEGREE_MATCHED,
           "degree_matched_k_whole_dataset_descriptive": matched_k, "graph_method": GRAPH_METHOD,
           "wpli_threshold": WPLI_THRESHOLD, "node_handcrafted": NODE_HANDCRAFTED,
           "agg_method": AGG_METHOD, "weight_decay": WEIGHT_DECAY,
           # Additional config/metadata fields:
           "artifact_rejection_uv_thresh": ARTIFACT_UV_THRESH if RUN_ARTIFACT_REJECTION else None,
           "exclude_frontal_sensitivity_run": EXCLUDE_FRONTAL,
           "ica_ocular_correction": ({"z_thresh": ICA_Z_THRESH, "max_remove": ICA_MAX_REMOVE,
                                      "subjects_with_removal": int((reject_log.n_ic_removed > 0).sum())}
                                     if RUN_ICA else None),
           "n_perm_classical": N_PERM, "n_perm_window_trained": N_PERM_WINDOW,
           "n_perm_deep": N_PERM_DEEP, "library_versions": _lib_versions},
          open(f"{OUT_DIR}/config.json", "w"), indent=2)

if RUN_DEEP and RUN_INTERPRETABILITY and INTERPRET_MODEL in MODELS:
    print("\n" + "=" * 70)
    print(f"INTERPRETABILITY: biomarkers + attention + t-SNE for {INTERPRET_MODEL} (repeat 0)")
    print("=" * 70)
    interp = run_interpretability(INTERPRET_MODEL, MODELS, MODELS[INTERPRET_MODEL]['graphs'], splits, repeat=0)
else:
    interp = None

# =========================================================
# FIGURES (300 DPI PNGs) -- unchanged from v7
# =========================================================
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
print("\n" + "=" * 70); print("GENERATING PUBLICATION FIGURES"); print("=" * 70)


def _save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Figure] saved {name}")


def _model_color(name, order):
    return MODEL_PALETTE[order.index(name) % len(MODEL_PALETTE)]


CH_NAMES = CHANNEL_NAMES_DEFAULT if (USE_NAMED_CHANNELS and len(CHANNEL_NAMES_DEFAULT) == 19) \
    else [f"Ch{i}" for i in range(19)]

try:
    order = list(dict.fromkeys(folds.model))
    plot_rows = []
    for name in order:
        o = oof[oof.model == name]
        ra = np.array([((og.p >= .5) == og.y).mean() for _, og in o.groupby("repeat")])
        plot_rows.append((name, ra.mean(), ra.std()))
    plot_rows.sort(key=lambda r: r[1])
    names_sorted = [r[0] for r in plot_rows]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(names_sorted) + 1.2))
    ypos = np.arange(len(names_sorted))
    ax.barh(ypos, [r[1] * 100 for r in plot_rows], xerr=[r[2] * 100 for r in plot_rows],
            color=[_model_color(n, order) for n in names_sorted], edgecolor="white", height=0.65,
            error_kw=dict(ecolor="#333333", capsize=3, lw=1))
    ax.axvline(50, color=COLOR_MUTED, linestyle=":", lw=1, label="Chance (50%)")
    ax.set_yticks(ypos); ax.set_yticklabels(names_sorted)
    ax.set_xlabel("Pooled subject-level accuracy (%)  [mean $\\pm$ SD across repeats]")
    ax.set_title("Model Comparison -- Repeated Subject-Level CV")
    ax.set_xlim(40, 100); ax.legend(loc="lower right")
    _save(fig, "fig1_model_comparison.png")
except Exception as e:
    print(f"  [Figure] fig1 skipped: {e}")

try:
    roc_models = [m for m in ["GraphODE", "GraphODE-MultiBand", "GraphODE-DegreeMatched",
                              "StackedGAT", "LR[wPLI]"] if m in set(oof.model)]
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, name in enumerate(roc_models):
        o = oof[oof.model == name].groupby("sid").agg(p=("p", "mean"), y=("y", "first"))
        fpr, tpr, _ = roc_curve(o.y.values, o.p.values)
        ax.plot(fpr, tpr, lw=2, color=MODEL_PALETTE[i % len(MODEL_PALETTE)],
                label=f"{name} (AUC={roc_auc_score(o.y.values, o.p.values):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (pooled OOF, repeat-averaged)")
    ax.legend(loc="lower right")
    _save(fig, "fig2_roc_curves.png")
except Exception as e:
    print(f"  [Figure] fig2 skipped: {e}")

try:
    o = oof[oof.model == INTERPRET_MODEL].groupby("sid").agg(p=("p", "mean"), y=("y", "first"))
    cm = confusion_matrix(o.y.values, (o.p.values >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.3, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Control", "ADHD"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Control", "ADHD"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix -- {INTERPRET_MODEL}\n(pooled OOF, repeat-averaged)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "fig3_confusion_matrix.png")
except Exception as e:
    print(f"  [Figure] fig3 skipped: {e}")

if degree_log is not None:
    try:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.plot([x[0] for x in degree_log], [x[1] for x in degree_log], "o-", color=COLOR_PRIMARY, lw=2,
                label="Single-band avg. degree")
        ax.axhline(mb_avg_degree, color=COLOR_ACCENT, linestyle="--", lw=1.5,
                   label=f"Multi-band fused avg. degree ({mb_avg_degree:.2f})")
        ax.axvline(matched_k, color=COLOR_MUTED, linestyle=":", lw=1.5,
                   label=f"Whole-dataset descriptive k={matched_k} (illustrative; the evaluated "
                         f"arm selects k per outer training fold -- see *_fold_matched_k.csv)")
        ax.set_xlabel("Top-k neighbours"); ax.set_ylabel("Average node degree")
        ax.set_title("Degree-Matching: Single-Band k Selection (whole-dataset, illustrative)")
        ax.legend()
        _save(fig, "fig4_degree_matching.png")
    except Exception as e:
        print(f"  [Figure] fig4 skipped: {e}")

try:
    nov = [m for m in ["GraphODE", "GraphODE-MultiBand", "GraphODE-DegreeMatched"] if m in set(folds.model)]
    label_map = {"GraphODE": "Single-band\n(default k)", "GraphODE-MultiBand": "Multi-band\n(fused)",
                 "GraphODE-DegreeMatched": "Single-band\n(degree-matched, per-fold k)"}
    color_map = {"GraphODE": COLOR_PRIMARY, "GraphODE-MultiBand": COLOR_ACCENT, "GraphODE-DegreeMatched": "#7f7f7f"}
    means = [folds[folds.model == n].acc.mean() * 100 for n in nov]
    stds = [folds[folds.model == n].acc.std() * 100 for n in nov]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(nov))
    ax.bar(x, means, yerr=stds, capsize=4, color=[color_map[n] for n in nov], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([label_map[n] for n in nov])
    ax.set_ylabel("Fold-level accuracy (%)")
    ax.set_title("Novelty Test: Does Multi-Band Beat a Density-Matched Control?")
    if cmp_res is not None:
        y_max = max(m + s for m, s in zip(means, stds)) + 3
        i_mb, i_dm = nov.index("GraphODE-MultiBand"), nov.index("GraphODE-DegreeMatched")
        ax.plot([i_mb, i_mb, i_dm, i_dm], [y_max, y_max + 1.5, y_max + 1.5, y_max], color="black", lw=1)
        lab = (f"p={cmp_res['p_nb']:.3f}" + (" (n.s.)" if cmp_res['p_nb'] >= 0.05 else " *") +
               f"\nNB 95% CI: [{cmp_res['ci'][0]*100:+.1f}, {cmp_res['ci'][1]*100:+.1f}] pp")
        ax.text((i_mb + i_dm) / 2, y_max + 2, lab, ha="center", fontsize=9.5)
        ax.set_ylim(0, y_max + 11)
    _save(fig, "fig5_novelty_test.png")
except Exception as e:
    print(f"  [Figure] fig5 skipped: {e}")

if interp is not None:
    try:
        bdf = interp['bdf']; bio_stats = interp['bio_stats']
        fig, axes = plt.subplots(1, len(BIOMARKER_COLS), figsize=(7.5, 4.2))
        for ax, c in zip(axes, BIOMARKER_COLS):
            a = bdf[bdf.label == 1][c].dropna().values
            b = bdf[bdf.label == 0][c].dropna().values
            parts = ax.violinplot([b, a], showmeans=True, showextrema=False)
            for pc, col in zip(parts['bodies'], [COLOR_CTRL, COLOR_ADHD]):
                pc.set_facecolor(col); pc.set_alpha(0.55)
            rng = np.random.RandomState(0)
            ax.scatter(1 + rng.uniform(-0.06, 0.06, len(b)), b, s=10, color=COLOR_CTRL, alpha=0.5)
            ax.scatter(2 + rng.uniform(-0.06, 0.06, len(a)), a, s=10, color=COLOR_ADHD, alpha=0.5)
            ax.set_xticks([1, 2]); ax.set_xticklabels(["Control", "ADHD"])
            row = bio_stats[bio_stats.biomarker == c]
            p_fdr = row.p_fdr_bh.values[0] if len(row) else np.nan
            ax.set_title(f"{c}\n(p_fdr={p_fdr:.3f})", fontsize=10)
        fig.suptitle("Latent-Dynamics Biomarkers -- ADHD vs Control (out-of-fold)", y=1.04)
        fig.tight_layout()
        _save(fig, "fig6_biomarkers.png")
    except Exception as e:
        print(f"  [Figure] fig6 skipped: {e}")

    try:
        mat_adhd, mat_ctrl = interp['mat_adhd'], interp['mat_ctrl']
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        vmax = max(mat_adhd.max(), mat_ctrl.max())
        for ax, mat, title in [(axes[0], mat_adhd, "ADHD Cohort"), (axes[1], mat_ctrl, "Control Cohort")]:
            im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xticks(range(19)); ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=7)
            ax.set_yticks(range(19)); ax.set_yticklabels(CH_NAMES, fontsize=7)
            ax.set_title(f"{title} Attention (OOF)"); ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        _save(fig, "fig7_attention_maps.png")
    except Exception as e:
        print(f"  [Figure] fig7 skipped: {e}")

    try:
        imp = interp['importance_df'].head(10).copy()
        imp['label'] = imp['channel_idx'].map(lambda i: CH_NAMES[i])
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.bar(imp['label'], imp['importance_score'], color=COLOR_ACCENT, edgecolor="white")
        ax.set_ylabel("|ADHD - Control| attention difference (summed)")
        ax.set_title("Top-10 Electrodes by ADHD-vs-Control Attention Divergence")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _save(fig, "fig8_channel_importance.png")
    except Exception as e:
        print(f"  [Figure] fig8 skipped: {e}")

    try:
        tsne = interp['tsne']
        if tsne is not None:
            emb, node_labels, n_steps = tsne['embeddings'], tsne['node_labels'], tsne['n_steps']
            fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 3.2), sharex=True, sharey=True)
            axes = [axes] if n_steps == 1 else axes
            for s in range(n_steps):
                pts = emb[s]
                axes[s].scatter(pts[node_labels == 0, 0], pts[node_labels == 0, 1], s=8, color=COLOR_CTRL,
                                alpha=0.6, label="Control" if s == 0 else None)
                axes[s].scatter(pts[node_labels == 1, 0], pts[node_labels == 1, 1], s=8, color=COLOR_ADHD,
                                alpha=0.6, label="ADHD" if s == 0 else None)
                axes[s].set_title(f"ODE t={s/(n_steps-1):.1f}", fontsize=10)
                axes[s].set_xticks([]); axes[s].set_yticks([])
            axes[0].legend(loc="upper left", fontsize=8)
            fig.suptitle(f"Latent Trajectory t-SNE -- {INTERPRET_MODEL} (held-out windows, illustrative)", y=1.05)
            fig.tight_layout()
            _save(fig, "fig9_tsne_trajectory.png")
        else:
            print("  [Figure] fig9 skipped: t-SNE was not computed")
    except Exception as e:
        print(f"  [Figure] fig9 skipped: {e}")

print(f"\nDONE in {(time.time()-T0)/60:.1f} min. Saved to {OUT_DIR} (figures in {FIG_DIR})")