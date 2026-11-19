# Graph Neural ODE for EEG-Based ADHD Classification

Code for a graph neural ordinary differential equation (Graph ODE) model
that classifies Attention-Deficit/Hyperactivity Disorder (ADHD) from
resting-state EEG, using subject-level, repeated cross-validation and a
density-matched control to separate the effect of multi-band connectivity
information from the effect of graph density alone.

## Method summary

- 19-channel resting-state EEG is band-pass filtered (0.5-50 Hz) and split
  into fixed-length windows.
- Functional connectivity is estimated with the weighted phase-lag index
  (wPLI), computed broadband and in five standard frequency bands
  (delta, theta, alpha, beta, gamma) from a single FFT pass.
- Each window is represented as a graph: channels are nodes, and edges are
  the top-k strongest wPLI connections (k nearest neighbours per channel).
- A per-channel 1-D CNN encodes the raw waveform at each node, followed by
  a graph attention (GAT) input layer and a GAT-based ODE function
  integrated with `torchdiffeq`, and a classification head.
- Baselines compared against: a stacked-GAT model (no ODE layer), a
  CNN-only model (no graph layer), a CNN-LSTM model (temporal, no graph
  layer), a fully-connected-graph control, a random-topology-graph control,
  a degree-matched single-band control (selected per cross-validation fold,
  using only that fold's training subjects), classical logistic
  regression/SVM on wPLI and band-power features (both subject-level and
  window-level training), and a multi-band fused-graph variant.
- Evaluation uses repeated stratified subject-level k-fold cross-validation
  (5 repeats x 10 folds), Nadeau-Bengio corrected paired significance
  tests, Benjamini-Hochberg FDR correction across comparisons, TOST
  equivalence testing, and label-permutation tests.

## Repository contents

- `v8_main_pipeline.py` -- the full pipeline: data loading, feature
  extraction, graph construction, model definitions, training/evaluation
  loop, statistical tests, and figure generation.
- `requirements.txt` -- exact library versions used to produce the
  reported results.

## Running the pipeline

```bash
pip install -r requirements.txt
python v8_main_pipeline.py
```

Key environment variables (all optional; defaults reproduce the results
reported in the paper):

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_PATH` | - | Path to the EEG dataset folder |
| `OUT_DIR` | `/kaggle/working/results_v8` | Output directory |
| `REPEATS` / `FOLDS` | 5 / 10 | Repeated cross-validation design |
| `N_PERM` | 1000 | Permutations for the classical-baseline permutation test |
| `RUN_ARTIFACT_REJECTION` | 0 | Amplitude-based artifact rejection on/off |
| `EXCLUDE_FRONTAL` | 0 | Drop the two most eye-blink-prone channels (sensitivity analysis) |

All outputs (per-model metrics, out-of-fold predictions, the final
comparison table, and all figures) are written to `OUT_DIR`.

## Data availability

The EEG dataset used in this study is [TO BE CONFIRMED BY THE AUTHORS --
state here whether it is a public dataset with a citation/DOI, or private
with a contact for access].

## Citation

If you use this code, please cite:

Ramya, S., & Bommareddy, L. (2026). Graph Neural Ordinary Differential
Equations for Electroencephalography-Based Attention-Deficit/Hyperactivity
Disorder Classification: Subject-Level Evaluation with Density-Matched
Controls. *Frontiers in Psychiatry*. (manuscript in review)
