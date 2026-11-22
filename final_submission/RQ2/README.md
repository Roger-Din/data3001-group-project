# RQ2 — Regional predictability

RQ2 asks whether the frozen point forecast has the same accuracy in different ocean environments and whether region-specific retraining adds useful predictive skill.

## Main pipeline

- Scripts `01`–`05`: processed-data and bathymetry checks.
- Scripts `06`–`12d`: GLORYS climatology, connected-current components and case-level current features.
- Script `13`: final regime labels — Shelf, Kuroshio Core, Transition and Other Offshore.
- Scripts `14`–`17`: original six-model regional comparison, drifter-clustered inference and figures.
- Scripts `18`–`20`: exploratory region-conditioned training and lead-time analyses.
- Script `21`: exact reconstruction of the report's frozen optimised-FNN regional metrics.
- Script `22`: reproducible Kuroshio-Core retraining of the frozen optimised-FNN architecture.

## Two experiments that must not be confused

1. **Regional evaluation without retraining:** the globally fitted FNN is kept frozen and evaluated within each regime. This gives 40.83 km RMSE on Shelf and 85.86 km in the Kuroshio Core.
2. **Kuroshio-only retraining:** the selected FNN architecture is refitted using only 646 Kuroshio-Core training cases and evaluated on the same 139 Core test cases. Its RMSE is 82.12 km, a 4.35% improvement over the frozen global FNN.

The second result does not replace the project's common global point estimator. It is an exploratory RQ2 result showing that some regional adaptation is possible.

## Important result locations

- `results/report_optimised_fnn_regime_summary.csv`: frozen global FNN by regime.
- `results/optimised_fnn_analysis/core_retraining_comparison.csv`: final Core-only optimised-FNN comparison.
- `results/optimised_fnn_analysis/regional_retraining_five_models_summary.csv`: regional refits for five learned models.
- `data/kuroshio_all_models/`: earlier original-model comparison retained for provenance.
- `figures/main/`: main RQ2 figures.
- `figures/optimised_fnn_analysis/`: final Core-retraining and sensitivity figures.

See `../docs/RQ2_ZIP_COMPARISON.md` for how the late RQ2 hand-off changed the interpretation.
