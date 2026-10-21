# Reproducibility release: IoT-oriented dual-pipeline framework

This directory accompanies the revised manuscript:

> **An IoT-Oriented Dual-Pipeline Machine-Learning Framework for Turbofan Health Monitoring and Inventory Backorder Risk Prediction**

It contains the experiment scripts, selected derived result files, LaTeX tables, and paper figures. It does **not** redistribute the two public datasets. Instructions for obtaining them are in `data/README.md` and `CMaps/README.md`.

## What the code reproduces

- **IoT pipeline (NASA C-MAPSS FD001):** causal, per-engine ARIMA residual features; past-only rolling features; engine-grouped 5-fold cross-validation; a 70/30 engine-grouped holdout; a paired 45-versus-48-feature ablation; permutation importance and SHAP.
- **External domain-shift tests:** FD002, FD003, and FD004 using the same 15 FD001 sensor channels and the same 48-feature schema, with no external tuning.
- **Backorder pipeline (public Kaggle data):** a 42-configuration resampling/classifier screen on a 240,000/60,000 split of the 300,000-row training subset; five-fold cross-validation of the top three configurations and three prespecified baselines; one-time evaluation on the official 242,075-row test split; bootstrap CIs; McNemar and DeLong tests; precision-recall/AP; Platt calibration; cost-sensitive thresholds; base-rate prior correction; permutation importance and SHAP.
- **Paper artefacts:** the nine LaTeX tables in `tables/` and the ten figures used in the manuscript in `figures/`.

The paper reports a null result for the three ARIMA residual features: the 45-versus-48 ablation is not statistically detectable. The code retains that comparison.

## Environment

Tested with Python 3.13.2 on Windows 11 with the package versions in `requirements.txt`:

```text
numpy 2.3.3
pandas 2.3.2
scikit-learn 1.9.0
xgboost 3.4.1
imbalanced-learn 0.14.2
statsmodels 0.15.0
scipy 1.18.1
shap 0.54.1
matplotlib 3.10.6
seaborn 0.13.2
joblib 1.5.3
```

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

All random seeds are fixed to `42`. `BO_N_JOBS` (default `4`) controls the number of parallel jobs in the backorder pipeline; `N_JOBS` in `iot_revision.py` is also `4`. Reduce these values on machines with limited memory.

## Data layout

Place the public files at the following paths, relative to this directory:

```text
CMaps/
  train_FD001.txt
  train_FD002.txt
  train_FD003.txt
  train_FD004.txt

data/
  Training_BOP.csv
  Testing_BOP.csv
```

The scripts expect the original whitespace-separated C-MAPSS training files and the original Kaggle CSV files, including the header row of the backorder data. The `sku` column is dropped by the backorder pipeline because it is a unique row identifier, not a product key.

See `CMaps/README.md` and `data/README.md` for source and licence notes. The datasets are not included here.

## Running the experiments

Run commands from this directory (the scripts resolve paths relative to their own location).

### 1. IoT pipeline

```bash
python iot_revision.py --only all
```

This performs the FD001 analysis and then the external FD002-FD004 tests serially. To run only one part:

```bash
python iot_revision.py --only FD001
python iot_revision.py --only external
```

The per-engine ARIMA residual computation is the runtime bottleneck. A parallel external runner is provided:

```bash
python iot_external_parallel.py
```

It uses the same splits and seeds as `iot_revision.py` and writes the same `results/iot_FD00*.json` files.

After the FD001 run has written `results/iot_features_FD001.csv`, run the paired ablation:

```bash
python iot_ablation.py
```

### 2. Backorder pipeline

```bash
python bo_revision.py
```

This is the longest step. It writes the validation grid, cross-validation folds, final predictions, calibration/cost artefacts, permutation/SHAP outputs, and `results/bo_revision.json`. It reuses `results/bo_validation_grid.csv` if that file exists, so an interrupted run can resume after the screening stage. Delete that file to force a fresh 42-configuration screen.

### 3. Tables and figures

```bash
python make_tables.py
python make_figures_revision.py
```

`make_tables.py` uses the result JSON/CSV files in `results/` and writes the nine `.tex` tables to `tables/`.

`make_figures_revision.py` uses both the saved result files and larger intermediates produced by the pipelines (`iot_features_FD001.csv`, `iot_curve_*.npz`, `bo_curves.json`, `bo_oof_predictions.csv`). Those large intermediates and the full prediction files are omitted from this release to keep it small. The figure PDFs used in the manuscript are included in `figures/`; rerun the pipelines first if you need to regenerate them from scratch.

## Included artefacts

```text
release/
  iot_revision.py              IoT pipeline, causal ARIMA features, CV/holdout
  iot_external_parallel.py     parallel FD002-FD004 runner
  iot_ablation.py              45-vs-48 paired ablation
  bo_revision.py               backorder screening/CV/test/calibration/cost
  stats_utils.py               metrics, bootstrap, McNemar, DeLong
  make_tables.py               LaTeX table generator
  make_figures_revision.py     figure generator
  requirements.txt
  results/                     selected JSON/CSV result artefacts
  tables/                      nine LaTeX tables
  figures/                     ten manuscript figures (PDF)
  data/README.md               backorder data instructions
  CMaps/README.md              C-MAPSS data instructions
```

Selected results include the FD001/FD002/FD003/FD004 metrics, the 45-vs-48 ablation tests, ARIMA orders and diagnostics, CV folds, holdout predictions, permutation/SHAP tables, the full 42-row validation grid, backorder CV folds, final/bootstrap/McNemar/DeLong summaries, calibration and cost thresholds. The large per-record prediction files (`bo_final_predictions.csv`, `bo_oof_predictions.csv`), full feature matrices, and full ROC/PR curve dump (`bo_curves.json`) are regenerated by the scripts and are not shipped here.

## Scope and caveats

- C-MAPSS is a simulation-generated benchmark, not field data.
- The two pipelines are independently validated components. Their integration is a proposed blueprint, not an end-to-end validated logistics system.
- The backorder task is row-level; the Kaggle `sku` field cannot support product-grouped validation.
- No field deployment, real-time control, or inventory-cost optimisation is claimed.
- SHAP and permutation importance are associational, not causal.
- Cost-sensitive thresholds are selected on out-of-fold predictions and reported as sensitivity analyses, not deployed operating points.

## Code availability

This directory is the supplementary release submitted with the manuscript. A public repository URL will be added upon acceptance. The release contains no proprietary data and no credentials.
