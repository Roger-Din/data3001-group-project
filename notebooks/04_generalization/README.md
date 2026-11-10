# External spatial generalization / 外部空间泛化

This folder tests whether champion models trained in Region A remain useful in geographically distinct ocean regions. It is a stricter **zero-shot transfer** experiment: external cases are never used for hyperparameter selection or refitting.

## Notebook order

### 1. `01_region_selection_and_cache_CN.ipynb`

- uses ten fixed ocean tiles from the original seeded region-selection run;
- downloads/verifies NOAA observations from 2018–2022;
- excludes every drifter ID observed in Region A;
- keeps one complete 96-hour case per external drifter;
- saves a reproducible local cache.

The current cache contains 686 external cases. The approximately 62 MB cache is generated data and is deliberately not committed to GitHub.

### 2. `02_champion_models_spatial_generalization_CN.ipynb`

- trains fixed champion configurations on the 3,591 Region-A training cases only;
- applies Optimised FNN, Optimised Weighted KNN and Optimised Random Forest directly to the ten external regions;
- compares them with velocity persistence and a Region-A mean-flow fallback;
- reports 72-hour RMSE, median error, median ITE and skill relative to persistence;
- resamples external drifters to form bootstrap intervals for the average region-level RMSE difference.

Boosting residual is omitted because its fitted definition requires GEBCO, GLORYS, Kuroshio-distance, regime and drogue covariates that are absent from the external cache. Replacing these inputs would define a different model.

## Current result

| Model | Mean of ten region-level 72 h RMSEs | Change versus persistence |
|---|---:|---:|
| Optimised Random Forest | **61.51 km** | **18.41% improvement** |
| Velocity persistence | 75.40 km | reference |
| Optimised Weighted KNN | 82.14 km | 8.94% worse |
| Region-A mean flow | 87.06 km | 15.47% worse |
| Optimised FNN | 293.19 km | 288.85% worse |

Random Forest beats persistence by an average 13.88 km across regions; the external-drifter bootstrap 95% interval is −18.56 to −9.11 km. The result supports Random Forest as the strongest zero-shot transfer model in this experiment. It does not establish worldwide performance: the conclusion is restricted to these ten observed tiles and this time window.

## Outputs

Compact results are in [`../../output/generalization/champion_spatial/`](../../output/generalization/champion_spatial/): case-level errors, region-level metrics, the overall summary, the bootstrap comparison and the summary figure.

Run both notebooks from the repository root. Notebook 01 needs network access when the cache is absent; Notebook 02 can then reuse the saved local cache.
