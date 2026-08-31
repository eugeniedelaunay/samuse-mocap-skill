# SAMuSe — Motion-Capture Musician Skill Classification

Reproducible, one-command pipeline for classifying musician skill level
(novice / advanced_beginner / competent / expert) from motion-capture
recordings of violin and clarinet performers, using both classical ML on
engineered kinematic features and an RNN on raw joint-position sequences.

## What the pipeline does

1. **Preprocess** (`preprocess.py`) — cleans raw MoCap CSVs: drops
   placeholder/empty columns, linearly interpolates short gaps (≤5 frames),
   applies a zero-phase Butterworth low-pass filter with joint-group-specific
   cutoffs (distal joints filtered less aggressively than proximal/trunk joints).
2. **Unified PCA evaluation** (`unified_pca_evaluation.py`) — fits shared
   Principal Movements (12a-style) and per-block PCA consistency (11b-style)
   features across violin and clarinet on a common 50-signal joint-angle set.
3. **Feature extraction** (`extract_features3.py`) — computes ROM, ROM rate,
   windowed SPARC (smoothness), stroke-segmented LDJ (jerk), and merges in the
   shared PCA features from step 2, per trial.
4. **Feature diagnostics** (`analyze_features2.py`) — QC report: missingness,
   near-zero variance, outliers, univariate association with skill,
   multicollinearity, instrument confounds. Exploratory only — not used to
   pick final features inside the actual CV loops.
5. **Classical training, trial-wise** (`train_skill_classifiers.py`) —
   logistic regression, linear SVM, random forest on engineered features,
   `StratifiedKFold` (same participant may appear in train and test).
6. **Classical training, participant-wise** (`train_skill_classifiers_grouped.py`)
   — same models, `StratifiedGroupKFold` (no participant in both train and test).
7. **RNN training** (`train_rnn_skill.py`) — GRU/LSTM directly on raw
   joint-position sequences, both CV protocols, 2-class (novice vs. expert)
   and 3-class (novice / advanced_beginner+competent / expert) tasks, pooled
   across instruments. Hyperparameters are set directly in `config.yaml` /
   CLI flags — there is no automated hyperparameter search in this pipeline.

### Known reference results (for sanity-checking your own run, not ground truth to reproduce exactly)

| Setup | Balanced accuracy |
|---|---|
| Classical, participant-wise, 3-class, shared-PCA random forest | 0.670 |
| RNN, participant-wise, 3-class | 0.534 (likely data-size limited, ~45 participants) |
| RNN, trial-wise, 3-class | 0.932–0.959 (same participant can appear in train+test — not a generalization estimate) |

If your run lands far outside these ranges, something in the environment,
config, or data likely differs from the reference setup — check `config.yaml`
diffs first before assuming a code bug.

## Setup

```powershell
# Clone and enter the repo
git clone <repo-url>
cd samuse-github

# Create and activate a virtual environment (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

On Linux/macOS, replace the activation line with `source .venv/bin/activate`.

If you don't have a GPU or don't want to install PyTorch, you can skip the
RNN stage entirely (see below) and only install the non-RNN lines from
`requirements.txt`.

### Getting the data

- **Raw MoCap recordings are NOT in this repo** (participant privacy). Ask
  your PI or the data steward for access, then place them under `data/raw/`
  (or point `paths.raw_mocap_dir` in `config.yaml` at wherever you keep them).
- **Everything else needed to run training is already committed**:
  `data/metadata/participants_instrument_skill_toshare.xlsx`,
  `data/pca/unified_pca_participant_features.csv`, and
  `data/features_extracted/trial_features.csv` are de-identified and small.
  This means you can run stages 5–7 (classical + RNN training) immediately
  after cloning, without ever touching raw MoCap data, if you just want to
  reproduce the modeling results.

## Running the pipeline

Everything is driven by `config.yaml` — edit the `paths:` section once to
point at your local data locations, then run:

```powershell
python run_pipeline.py --config config.yaml
```

This runs all seven stages in dependency order and stops immediately with a
clear error (naming the failing stage and command) if anything breaks.

### Common variations

```powershell
# Classical-only run (skip the RNN stage entirely — much faster, no GPU/PyTorch needed)
python run_pipeline.py --config config.yaml --skip-rnn

# Resume from a specific stage after fixing an error (avoids re-running earlier stages)
python run_pipeline.py --config config.yaml --only extract_features,analyze_features,train_classical

# See exactly what would run without executing anything
python run_pipeline.py --config config.yaml --dry-run
```

Valid stage names for `--only`: `preprocess`, `unified_pca`, `extract_features`,
`analyze_features`, `train_classical`, `train_grouped`, `train_rnn`.

## Expected runtime

Approximate, on a typical laptop CPU (~45 participants, both instruments).
These are rough estimates, not benchmarks — recalibrate after your first run:

| Stage | Typical time |
|---|---|
| Preprocess | 5–15 min |
| Unified PCA evaluation | 2–5 min |
| Feature extraction | 5–10 min |
| Feature diagnostics | under 1 min |
| Classical training (all tasks/modes/configs) | 5–15 min |
| RNN training (CPU, all tasks/protocols) | 1–3 hours |
| RNN training (GPU) | 15–40 min |

The RNN stage dominates total runtime. Use `--skip-rnn` if you only need
classical results, or a machine with a CUDA GPU for the full run.

## Where outputs land

All output paths are set in `config.yaml` under `paths:` and are relative to
the repo root by default:

- `data/preprocessed/` — cleaned trial CSVs + `preprocessing_summary.csv`
- `results/unified_pca_evaluation/` — shared PM loadings, consistency and
  variability features, `unified_pca_participant_features.csv`
- `data/features_extracted/` — `trial_features.csv` (committed once stable)
- `data/features_analysis/` — QC reports (missingness, outliers, ranking)
- `results/training_results/` — trial-wise classical results (per run:
  `experiment_summary.csv`, `fold_metrics.csv`, `confusion_matrix_total.csv`)
- `results/training_results_grouped/` — participant-wise classical results
- `results/training_results_rnn/` — RNN fold metrics, confusion matrices,
  `experiment_summary.json` per task/protocol combination

None of the `results/` or `data/preprocessed/` directories are committed to
git — they're fully regenerable by re-running the pipeline (see `.gitignore`).

## What's committed vs. not

| Data | Committed? | Why |
|---|---|---|
| Raw MoCap recordings | **No** | Participant privacy — identifiable motion data |
| Preprocessed trial CSVs | No | Large, fully regenerable from raw + `preprocess.py` |
| QC/diagnostic reports | No | Regenerable from `trial_features.csv` |
| Training result folders | No | Regenerable from committed feature CSVs |
| `participants_instrument_skill_toshare.xlsx` | **Yes** | De-identified, small |
| `unified_pca_participant_features.csv` | **Yes** | De-identified, small |
| `trial_features.csv` | **Yes** | De-identified, small |
| `config.yaml`, all `src/*.py` | **Yes** | Code, not data |

If in doubt about whether a new derived file is safe to commit, check that it
contains no participant names, no raw frame-by-frame joint coordinates, and no
column that could re-identify a participant when cross-referenced with other
public information.

## Configuration reference

`config.yaml` sections and what they control:

- `paths` — every input/output directory and the metadata file location.
  This is the only section most users need to edit.
- `preprocessing` — sampling rate, filter order, gap-fill limit, per-joint-group
  Butterworth cutoffs.
- `feature_extraction` — SPARC window size/step, cutoffs, conditions to keep.
- `cross_validation` — number of folds, random seed (shared across all scripts
  for exact reproducibility of splits).
- `classical_models` — correlation pruning threshold, per-protocol random
  forest hyperparameters (deliberately different between trial-wise and
  participant-wise — see inline comment).
- `rnn` — architecture (hidden dim, layers, dropout), training (batch size,
  epochs, patience, learning rate, weight decay), device selection. All RNN
  hyperparameters here are fixed values chosen manually — there is no
  automated tuning stage in this pipeline.
- `unified_pca` — variance targets, number of Principal Movements saved/shown.

## Known issues and open work

- **`train_rnn_skill.py` needs a config-loading pass.** It currently reads
  all hyperparameters from CLI flags with hardcoded defaults; `run_pipeline.py`
  passes `config.yaml` values explicitly as flags, but the script itself
  doesn't yet import `config.yaml` directly. This is fine functionally but
  means editing `config.yaml`'s `rnn:` section only takes effect through the
  orchestrator, not when running `train_rnn_skill.py` standalone with no flags.
- **Device selection**: pass `--device` explicitly on shared/lab machines with
  multiple GPUs to avoid contention; `"auto"` picks CUDA if available, else CPU.
- **RNN participant-wise result (0.534 balanced accuracy) is close to chance**
  for 3-class — treat this as a known limitation of sample size (~45
  participants), not a training bug, unless a code change specifically
  addresses data volume (e.g., data augmentation, transfer learning, or
  pooling additional recording sessions).

## For new lab members

1. Install `requirements.txt` in a fresh virtual environment.
2. Get raw data access from your PI, or skip straight to step 3 if you only
   need to reproduce modeling results (the derived CSVs are already committed).
3. Run `python run_pipeline.py --config config.yaml --skip-rnn` first — it's
   fast and validates your environment/paths before committing to an RNN run.
4. Once that succeeds, run the full pipeline (or just `--only train_rnn`) for
   RNN results.
5. Compare your `experiment_summary.csv` / `.json` files against the reference
   results table above. Large deviations usually mean a `config.yaml` path or
   parameter mismatch, not a fundamental problem — check there first.
