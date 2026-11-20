"""Recording-length analysis (descriptive), computed from pipeline outputs.

Usage:  python recording_length_analysis.py <results_dir> [model]
Reads  <results_dir>/artifact_rejection_log.csv  (windows kept per participant)
and    <results_dir>/all_oof_predictions.csv     (out-of-fold probabilities).
Recording length in seconds = kept windows x 4 s (non-overlapping 512-sample
windows at 128 Hz). Writes <results_dir>/recording_length_analysis.txt and
<results_dir>/recording_length_by_subject.csv.
"""
import sys, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

args = [a for a in sys.argv[1:] if not a.startswith("-") and not a.endswith(".json")]
res = os.environ.get("RESULTS_DIR") or (args[0] if args else ".")
model = os.environ.get("MODEL") or (args[1] if len(args) > 1 else "GraphODE")
for f in ("artifact_rejection_log.csv", "all_oof_predictions.csv"):
    if not os.path.exists(os.path.join(res, f)):
        raise SystemExit(f"{f} not found in '{res}'. Give the results folder, e.g. RESULTS_DIR=/path/to/results python recording_length_analysis.py")
WIN_S = 4
log = pd.read_csv(os.path.join(res, "artifact_rejection_log.csv"))
oof = pd.read_csv(os.path.join(res, "all_oof_predictions.csv"))
oof = oof[oof.model == model]

sid_col = "sid" if "sid" in log.columns else log.columns[0]
d = log.rename(columns={sid_col: "sid"})[["sid", "n_kept"]].copy()
lab = oof.groupby("sid").y.first().reset_index().rename(columns={"y": "label"})
d = d.merge(lab, on="sid")
d["dur_s"] = d.n_kept * WIN_S
d["tert"] = pd.qcut(d.dur_s.rank(method="first"), 3, labels=["short", "mid", "long"])
d.to_csv(os.path.join(res, "recording_length_by_subject.csv"), index=False)

out = []
P = out.append
a, c = d[d.label == 1], d[d.label == 0]
for name, g in (("ADHD", a), ("Control", c)):
    P(f"{name}: n={len(g)} median={g.dur_s.median():.0f}s IQR=[{g.dur_s.quantile(.25):.0f},{g.dur_s.quantile(.75):.0f}] range=[{g.dur_s.min()},{g.dur_s.max()}] windows={g.n_kept.sum()}")
P(f"Mann-Whitney p={mannwhitneyu(a.dur_s, c.dur_s).pvalue:.4f}")
P(str(d.groupby(["tert", "label"], observed=True).size().unstack()))
P(str(d.groupby("tert", observed=True).dur_s.agg(["min", "max", "count"])))

oof = oof.assign(correct=((oof.p >= .5).astype(int) == oof.y).astype(int))
acc = oof.merge(d[["sid", "tert"]], on="sid").groupby("tert", observed=True).correct.mean()
P("Accuracy by third (all repeats): " + ", ".join(f"{k}={v*100:.1f}%" for k, v in acc.items()))
s = oof.groupby("sid").agg(p=("p", "mean"), y=("y", "first")).reset_index().merge(d[["sid", "dur_s"]], on="sid")
P(f"{model} subject AUC={roc_auc_score(s.y, s.p):.3f} | recording length alone AUC={roc_auc_score(s.y, s.dur_s):.3f}")
for k, nm in ((1, "ADHD"), (0, "Control")):
    x = s[s.y == k]; r = spearmanr(x.dur_s, x.p)
    P(f"Spearman(length, probability) {nm}: rho={r[0]:.2f} p={r[1]:.3f}")
x = s[(s.dur_s >= 104) & (s.dur_s <= 184)]
P(f"Overlap range 104-184 s: n={len(x)} ADHD={int(x.y.sum())} AUC={roc_auc_score(x.y, x.p):.3f}")
txt = "\n".join(out); print(txt)
open(os.path.join(res, "recording_length_analysis.txt"), "w").write(txt)
