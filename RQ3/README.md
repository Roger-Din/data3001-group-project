# DATA3001 — RQ3: Conformal Prediction for Ocean Drifter Trajectories

## Research question

How reliably can a fixed 72-hour trajectory forecasting model quantify positional uncertainty, and does its coverage vary across ocean regimes?

## Scope

This folder contains the complete **English-language RQ3 analysis code**: global split conformal prediction, regime-specific Mondrian calibration, a drifter-max sensitivity analysis, a paired test-drifter cluster bootstrap, and an independent integrity audit. The scripts reuse the fixed RQ1 feedforward neural network; **they do not retrain it**.

## Required inputs (stored elsewhere in the DATA3001 repository)

Run from the `DATA3001/` project root. The following existing files must be present:

| Relative path | Purpose |
| --- | --- |
| `data/processed/forecast_origins.parquet` | Forecast cases, drifter IDs, and partitions |
| `data/processed/region_A_observations.parquet` | Observed trajectories |
| `output/rq1/en/fnn_model_parameters.npz` | Fixed RQ1 FNN weights |
| `output/rq1/en/fnn_test_case_metrics.csv` | Saved RQ1 test errors for validation |
| `RQ2/data/forecast_case_regimes.csv` | RQ2 ocean regime labels |

Do not commit large raw datasets or sensitive data without permission. The five `Unclassified` regime labels are confined to the training partition and are retained in the input dataset.

## Installation and execution

```bash
# From DATA3001/ with the desired Python environment active:
pip install -r RQ3/requirements.txt
python RQ3/run_all.py
```

Alternatively, run each stage independently:

```bash
python RQ3/01_global_conformal.py
python RQ3/02_regime_and_drifter_aware.py
python RQ3/03_paired_cluster_bootstrap.py
python RQ3/04_integrity_and_drifter_audit.py
```

`run_all.py` runs the audit (which also runs stages 01–02) and then the bootstrap (which reruns stages 01–02). This intentionally favors simple, independently runnable scripts over caching. It can take several minutes.

## Code inventory

| File | Responsibility |
| --- | --- |
| `01_global_conformal.py` | Reconstruct FNN forecasts, global 80/90/95% pointwise regions, drifter-bootstrap intervals, regime diagnostics, simultaneous trajectory band, and plots |
| `02_regime_and_drifter_aware.py` | Global/Mondrian/drifter-max 72-hour comparisons, case-level and drifter-level coverage, and coverage plot |
| `03_paired_cluster_bootstrap.py` | 2,000 paired resamples of complete held-out test drifters, with fixed calibration radii |
| `04_integrity_and_drifter_audit.py` | Case-ID alignment, split leakage, regime completeness, independent threshold and coverage recomputation, drifter-level audit |
| `run_all.py` | One-command orchestration |
| `RESULTS_AND_LIMITATIONS.md` | Verified numerical results and interpretation |

All generated tables and figures are written to **`RQ3/output/`**. This directory is created automatically.

## What to commit to GitHub

Commit the Python scripts, `README.md`, `RESULTS_AND_LIMITATIONS.md`, `requirements.txt`, `.gitignore`, and **the generated `output/` CSV and PNG files after running the pipeline locally**. The ZIP does not contain generated CSV/PNG outputs because the full input data and model weights are not available here. Do not upload placeholder results.

Suggested Git workflow from the repository root:

```bash
python RQ3/run_all.py
# Confirm: Audit: 42/42 checks passed.
git add RQ3/
git commit -m "Add reproducible RQ3 conformal prediction analysis"
```

If `git status` shows unrelated files, review the staged changes before committing. Keep `output/rq1/en/` and the RQ2 regime table in their existing locations.

## Interpretation and reproducibility

- Case-level 90% coverage is **not** 90% all-case coverage for a complete drifter.
- Drifter-max calibration aggregates maximum endpoint error per calibration drifter *within a regime*. This is a sensitivity analysis, not a universal finite-sample guarantee.
- Calibration cases from the same drifter are dependent. Conventional case-exchangeable conformal guarantees do not automatically apply.
- Bootstrap confidence intervals resample test drifters only, conditional on the already estimated calibration radii.
- The Mondrian extension was motivated by inspection of global regime undercoverage. Treat method comparisons as exploratory and disclose test-informed method development.
- A passing integrity audit validates numerical consistency and data alignment, **not** the assumptions required for distribution-free coverage.
