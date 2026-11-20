# Graph Neural ODE for EEG-Based ADHD Classification

Code for a graph neural ordinary differential equation (Graph ODE) model
that classifies Attention-Deficit/Hyperactivity Disorder (ADHD) from
EEG recorded during a visual attention task, using subject-level, repeated cross-validation and a
density-matched control to separate the effect of multi-band connectivity
information from the effect of graph density alone.

## Method summary

- 19-channel task EEG (visual attention task) is band-pass filtered (0.5-50 Hz) and split
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
  layer), a fully-connected-graph control, a random-topology-graph control (same number of edges as the top-k graph in every window),
  a degree-matched single-band control (selected per cross-validation fold,
  using only that fold's training subjects), classical logistic
  regression/SVM on wPLI and band-power features (both subject-level and
  window-level training), and a multi-band fused-graph variant.
- Evaluation uses repeated stratified subject-level k-fold cross-validation
  (5 repeats x 10 folds), Nadeau-Bengio corrected paired significance
  tests, Benjamini-Hochberg FDR correction across comparisons, TOST
  equivalence testing, and label-permutation tests.

## Repository contents

- `full-pipeline.py` -- the full pipeline: data loading, feature
  extraction, graph construction, model definitions, training/evaluation
  loop, statistical tests, and figure generation.
- `requirements.txt` -- exact library versions used to produce the
  reported results.
- `eeg_quality_report.py` -- per-recording signal-quality statistics
  (missing values, flat channels, amplitude spread, windows whose
  amplitude exceeds 3x and 5x the recording median). Writes
  `eeg_quality_by_subject.csv` and `eeg_quality_summary.txt`.
- `recording_length_analysis.py` -- recording length and number of windows
  by group, and accuracy by recording-length tertile, from the saved
  out-of-fold predictions. Usage: `python recording_length_analysis.py <results_dir> [model]`.
- `make_sid_file_map.py` -- writes `sid_file_map.csv`, which links each
  participant index (`sid`) in the result files to its `.mat` file and label.
- `all_oof_predictions.csv`, `all_fold_results.csv`, `final_table.csv` -- saved
  outputs of the main subject-level comparison (out-of-fold predictions,
  fold-level results and summary table). The sensitivity runs (ICA artifact
  correction, Fp1/Fp2 exclusion, hyperparameter sweep, random-graph control)
  are regenerated with the environment variables listed below.

## Environment

Python 3.12.13; numpy 2.0.2, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.6.1,
torch 2.10.0 (CUDA 12.8 build), torch_geometric 2.8.1, torchdiffeq 0.2.5.
Each model was trained on a single CUDA device. The random seed is 42. The
cross-validation splits come from scikit-learn's `RepeatedStratifiedKFold`
(10 folds, 5 repeats, seed 42) on the participant labels, so they are the same
in every run; `data_splits_test_subjects.csv` lists the test participants of
every repeat (0 to 4) and fold (1 to 10); the training participants are all others.

## Running the pipeline

```bash
pip install -r requirements.txt
python full-pipeline.py
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
| `RUN_ICA` | 0 | ICA-based ocular-artifact correction (components correlated with Fp1/Fp2 are removed) |
| `ICA_Z_THRESH` / `ICA_MAX_REMOVE` | 3.0 / 2 | Component-selection threshold and maximum components removed per subject |
| `RUN_WINDOW_SPLIT_COMPARISON` | 1 | Window-trained SVM on wPLI under subject-disjoint vs random-window splits (`window_split_comparison.csv`) |
| `ONLY_WINDOW_SPLIT_COMPARISON` | 0 | If 1, run only that comparison and stop |
| `ONLY_HYPERPARAM_SWEEP` | 0 | If 1, run only the hyperparameter sensitivity sweep and stop |
| `RUN_STACKEDGAT` | 1 | Include the StackedGAT comparison model (0 to skip it) |
| `RUN_RANDOM_GRAPH` / `RUN_FULLY_CONNECTED` | 1 | Random-topology and fully-connected graph control models |
| `RUN_CNN_BASELINES` | 1 | CNN-only and CNN-LSTM control models |
| `RUN_WINDOW_TRAINED_SVM` | 1 | Window-trained SVM (saves per-subject predictions and fold-level results) |
| `RUN_INTERPRETABILITY` | 1 | Latent trajectory descriptors (divergence rate, switching rate), attention maps, t-SNE |
| `RUN_PERM_DEEP` | 1 | Label-permutation test for GraphODE |
| `N_PERM_DEEP` / `N_PERM_DEEP_FOLDS` | 5 / 3 | Number of permutations and number of folds of repeat 0 used in that test (the reported run used 50 permutations and 3 folds) |
| `N_PERM_DEEP_START` | 0 | If set (for example 50), skips permutations 1 to 50 so that a second run can compute the remaining ones. Finished permutations are saved in `deep_perm_null.csv` and re-used when a run is restarted |
| `RUN_HYPERPARAM_SWEEP` | 1 | One-at-a-time sensitivity of GraphODE to hidden size, learning rate and epoch budget on all 10 folds of repeat 0 (`hyperparam_sensitivity_folds.csv`, `hyperparam_sensitivity.csv`) |

All outputs (per-model metrics, out-of-fold predictions, the final
comparison table, and all figures) are written to `OUT_DIR`.

## Data availability

The publicly available EEG dataset analyzed in this study, "EEG Data for
ADHD/Control Children," can be accessed through IEEE DataPort at
<https://ieee-dataport.org/open-access/eeg-data-adhd-control-children>
(Motie Nasrabadi et al., 2020; DOI:
[10.21227/rzfh-zn36](https://doi.org/10.21227/rzfh-zn36)).

> Motie Nasrabadi, A., Allahverdy, A., Samavati, M., and Mohammadi, M. R.
> (2020). EEG data for ADHD / control children. *IEEE DataPort*.
> doi:10.21227/rzfh-zn36

The saved out-of-fold predictions, fold-level results and summary table of the
main comparison are in the files `all_oof_predictions.csv`, `all_fold_results.csv` and `final_table.csv` of this repository. They can also be regenerated by running the pipeline against the
dataset above, as described above. Deep-model results can differ slightly
between complete runs because GPU computations are not fully deterministic;
the classical baselines are exactly reproducible.

## Citation

If you use this code, please cite:

Ramya, S., & Bommareddy, L. (2026). Graph Neural Ordinary Differential
Equations for Electroencephalography-Based Attention-Deficit/Hyperactivity
Disorder Classification: Subject-Level Evaluation with Density-Matched
Controls. *Frontiers in Psychiatry*. (manuscript in review)
