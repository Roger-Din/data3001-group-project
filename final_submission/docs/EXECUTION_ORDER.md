# Execution order

## Recommended review path

1. Run `python run_reproduction.py --mode verify`.
2. Read `00_shared/01_model_development_and_trial_log.ipynb` for the complete model-development history.
3. Read `00_shared/00_report_figures.ipynb` for the report evidence index.
4. Review the RQ1, RQ2 and RQ3 notebooks in the order below.

## Full analytical order

### Stage 0 — data

The repository already includes the frozen processed inputs. Rebuilding them requires the raw GDP hourly archive and `scripts/build_processed_dataset.py`. The validator checks the expected dimensions and drifter-level partitioning before any downstream result is accepted.

### Stage 1 — RQ1 point forecasts

Core notebooks run in numerical order under `RQ1/notebooks/core/`:

1. velocity persistence
2. spatial mean flow
3. multi-output Ridge
4. FNN
5. pooled VAR(1)
6. weighted KNN

Tree ensembles and optimisation experiments are under `RQ1/notebooks/extensions/`. The frozen comparison uses the original Random Forest and the selected optimised FNN. Later exploratory models remain in the archive but do not replace the frozen conclusion.

### Stage 2 — RQ2 regional predictability

The numbered scripts in `RQ2/scripts/` document environmental-feature construction and regime assignment. Cached bathymetry/current products are included, so headline reproduction does not need to contact an external API. `21_reproduce_report_fnn_by_regime.py` merges the frozen optimised-FNN case errors with the final regime labels. `22_reproduce_core_optimised_fnn.py` independently refits the selected FNN on Core-only training cases and reproduces the 82.12 km extension result. Scripts 14--20 preserve broader model comparisons and regional extension experiments.

### Stage 3 — RQ3 conformal calibration

Run `RQ3/scripts/RQ3_optimised_fnn_final.py`, followed by `RQ3/scripts/06_validation_and_paired_bootstrap.py`. Final canonical outputs are written under `RQ3/optimised_fnn/output/`. Older alternative-pipeline results are retained separately under `RQ3/results/exploratory_original_pipeline/`.

### Stage 4 — report evidence

Use `00_shared/00_report_figures.ipynb` and `RQ3/scripts/make_report_figures.py` to assemble report-ready figures. Finally rerun the validator and checksum generator.
