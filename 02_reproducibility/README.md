# DATA3001 S01-G5 — Reproducible final submission

This folder contains the English-facing code, processed data, frozen predictions,
figures and validation checks for the surface-drifter forecasting project. The
group report is still being edited and is intentionally kept outside this
reproducibility package.

## Start here: instructor review in five steps

All commands below must be run from the `02_reproducibility` directory.

### Step 1 — Create the Python environment

Python 3.10 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 2 — Verify the submitted evidence

```bash
python run_reproduction.py --mode verify
```

This is the recommended first check. It does **not** retrain the models. It
recomputes the headline statistics from the frozen case-level outputs and checks:

- required files and SHA256 checksums;
- all canonical notebooks and RQ1 Python exports;
- the 3,591 / 769 / 757 training, calibration and test split;
- absence of drifter-ID leakage across partitions;
- the frozen RQ1, RQ2 and RQ3 headline results.

A successful review ends with:

```text
Overall submission validation: PASS
```

The machine-readable result is saved in `validation/latest_validation.json`.

### Step 3 — Read the project evidence in report order

Open Jupyter Lab:

```bash
jupyter lab
```

Then review these files:

1. `00_shared/01_model_development_and_trial_log.ipynb` — complete model
   development history, including unsuccessful and exploratory trials.
2. `00_shared/00_report_figures.ipynb` — index of the report figures and tables.
3. `RQ1/notebooks/` — point-forecast baselines, candidate models, optimisation
   and robustness checks.
4. `RQ2/notebooks/` — ocean-regime assignment, regional predictability and
   Core-only retraining.
5. `RQ3/notebooks/` — conformal calibration and regime-specific coverage.

The exact analytical order is listed in
[`docs/EXECUTION_ORDER.md`](docs/EXECUTION_ORDER.md). Every RQ1 notebook also
has a matching plain-Python export under `RQ1/scripts/`.

### Step 4 — Preview a full rerun

Before starting a long run, print every command without executing it:

```bash
python run_reproduction.py --mode full --stage all --dry-run
```

The stages can also be previewed separately:

```bash
python run_reproduction.py --mode full --stage rq1 --dry-run
python run_reproduction.py --mode full --stage rq2 --dry-run
python run_reproduction.py --mode full --stage rq3 --dry-run
```

### Step 5 — Re-run models when required

Run only the required research question:

```bash
python run_reproduction.py --mode full --stage rq1
python run_reproduction.py --mode full --stage rq2
python run_reproduction.py --mode full --stage rq3
```

Or run the complete submitted pipeline:

```bash
python run_reproduction.py --mode full --stage all
```

RQ1 is the slowest stage because it refits grouped-cross-validation models and
ensembles; runtime depends on CPU and memory and can take hours. RQ2 uses frozen
RQ1 case-level predictions and included environmental caches. RQ3 uses the
frozen FNN predictions and the reserved calibration partition. No GPU is
required by the submitted NumPy/scikit-learn pipelines.

Full execution rewrites some outputs and executed notebooks. Use a fresh copy of
the submission if the original frozen files must remain byte-for-byte unchanged.

## Where each stage writes its results

| Stage | Main inputs | Main outputs |
|---|---|---|
| RQ1 | `data/processed/` | `output/rq1/` |
| RQ2 | `output/rq1/`, `RQ2/data/` | `RQ2/results/` |
| RQ3 | frozen FNN predictions and calibration cases | `RQ3/optimised_fnn/output/` |
| Report evidence | frozen RQ outputs | `../01_report/figures/` and `00_shared/00_report_figures.ipynb` |

## Expected headline results

The automated validator recomputes these values from the included result files:

| Component | Expected result |
|---|---|
| Dataset | 562,944 hourly observations and 5,117 forecast cases |
| Frozen split | 3,591 training, 769 calibration and 757 test cases |
| RQ1 FNN | 72-hour RMSE 61.69 km |
| RQ1 original Random Forest | 72-hour RMSE 62.13 km |
| RQ2 Shelf / Kuroshio Core FNN | 40.83 / 85.86 km RMSE |
| RQ2 Core-only FNN retraining | 82.12 km RMSE; 4.35% reduction |
| RQ3 global conformal method | 91.02% coverage at 72 h; 101.87 km radius |
| RQ3 Kuroshio Core | 80.58% global versus 89.93% Mondrian coverage |

Small last-decimal differences may occur because the displayed values are
rounded. The validator uses the unrounded source files and documented tolerances.

## Submission map

| Path | Purpose |
|---|---|
| `00_shared/00_report_figures.ipynb` | Index of figures and tables used in the report |
| `00_shared/01_model_development_and_trial_log.ipynb` | Model-selection history, unsuccessful trials and later audits |
| `RQ1/notebooks/`, `RQ1/scripts/` | Baselines, candidate models, optimisation and RQ1 checks |
| `RQ2/notebooks/`, `RQ2/scripts/`, `RQ2/data/` | Ocean-regime and regional-predictability analysis |
| `RQ2/results/optimised_fnn_analysis/` | Final Core-only optimised-FNN refit and sensitivity results |
| `RQ3/notebooks/`, `RQ3/scripts/` | Conformal uncertainty-analysis workflows |
| `RQ3/optimised_fnn/output/` | Canonical final RQ3 outputs |
| `data/processed/` | Frozen leakage-safe processed inputs used across the RQs |
| `output/rq1/` | RQ1 predictions, metrics, CV summaries and diagnostics |
| `../01_report/figures/` | Report-ready figures grouped by RQ |
| `RESULTS_MANIFEST.csv` | Headline claims and their machine-readable sources |
| `MANIFEST_SHA256.txt` | Integrity checksums for the submitted files |
| `validation_summary.json` | Frozen summary of the most recent validation run |

## How the model-development record should be interpreted

The frozen RQ1 comparison contains two physical baselines plus pooled VAR(1),
Ridge, weighted KNN, FNN, Extra Trees and the original Random Forest. The FNN
was selected as the common point estimator because it had the lowest frozen
72-hour RMSE (61.69 km), while Random Forest was statistically competitive
(62.13 km).

Later tuning and diagnostic work is retained to make the trial-and-error process
visible, but it is not silently promoted to confirmatory evidence. In particular,
the tuned Random Forest was slightly worse than the original Random Forest, and
the residual-boosting result was produced after held-out inspection. These are
labelled exploratory in the trial log.

## Reproducibility safeguards

- Complete drifter IDs are assigned to only one partition.
- Calibration cases are excluded from RQ1 fitting and reserved for RQ3.
- Identical test-case IDs are used for paired model comparisons.
- RQ2 evaluates the same frozen FNN predictions by ocean regime.
- Bootstrap resampling is clustered by drifter ID where dependence matters.
- All 44 canonical notebooks are English-facing.
- Cached environmental inputs required for the submitted RQ2 results are included.
- Large raw upstream archives are not duplicated; provenance and reconstruction
  instructions are provided in [`data/README.md`](data/README.md).

## Troubleshooting

**`FileNotFoundError` for `data/processed`:** run the command from the
`02_reproducibility` directory, not from inside an RQ subfolder.

**`jupyter: command not found`:** reactivate `.venv` and rerun
`python -m pip install -r requirements.txt`.

**Checksum failure after intentionally rerunning code:** regenerated outputs no
longer match the frozen package. To establish a new local manifest, run:

```bash
python scripts/generate_manifest.py
python run_reproduction.py --mode verify
```

For additional detail, see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and
[`docs/KNOWN_DIFFERENCES.md`](docs/KNOWN_DIFFERENCES.md).
