# Notebook guide / Notebook 导航

This directory is the canonical entry point for the executable analyses. The folders follow the research workflow rather than the date on which a notebook was created.

本目录是项目 Notebook 的主入口。文件夹按照研究流程排列，而不是按照文件创建时间排列。

## Folder map

| Order | Folder | Main purpose | Main outputs | Status |
|---|---|---|---|---|
| 00 | [`00_preliminary/`](00_preliminary/) | Region-A extraction, QC, continuous segments, 24 h → 72 h forecast-case construction and proposal-stage analysis | Processed-data audit and preliminary figures | Complete |
| 01 | [`01_rq1_forecasting/`](01_rq1_forecasting/) | Physical baselines, initial learned models, champion-model optimisation and held-out forecast evaluation | Lead-time errors, RMSE, median error, ITE, forecast skill and drogue analysis | Complete |
| 02 | [`02_rq2_regime_analysis/`](02_rq2_regime_analysis/) | Compare fixed models across Shelf, Kuroshio Core, Transition and Other Offshore regimes; explore regional refitting and normalised errors | Regime-specific error curves, tables and bootstrap intervals | Complete with extensions |
| 03 | [`03_rq3_conformal_prediction/`](03_rq3_conformal_prediction/) | Convert point forecasts into calibrated circular prediction/search regions | Coverage, radius, area and regime-reliability diagnostics | Complete |
| 04 | [`04_generalization/`](04_generalization/) | Test whether Region-A champion models transfer to geographically distinct ocean regions without retraining | Ten-region zero-shot RMSE, median error, ITE and bootstrap comparison | Complete extension |

## Recommended reading order

1. `00_preliminary/regionA_final.ipynb` — where the Region-A data and 5,117 cases come from.
2. `01_rq1_forecasting/RQ1_training_and_test_audit_CN.ipynb` — partitioning and leakage audit.
3. `01_rq1_forecasting/zh/01...06` — six original models.
4. `01_rq1_forecasting/zh/10...14` — champion optimisation and common-metric comparison.
5. `01_rq1_forecasting/analysis/15_champion_models_drogue_effect_CN.ipynb` — attached/lost drogue analysis.
6. `02_rq2_regime_analysis/core/RQ2_complete_analysis_CN.ipynb` — complete RQ2 workflow.
7. `02_rq2_regime_analysis/core/20_regional_lead_time_analysis_CN.ipynb` — errors from 1 to 72 hours within each regime.
8. `02_rq2_regime_analysis/extensions/22_champion_models_by_regime_CN.ipynb` — frozen champions by regime.
9. `03_rq3_conformal_prediction/RQ3_conformal_prediction_CN.ipynb` — initial-FNN conformal analysis.
10. `03_rq3_conformal_prediction/RQ3_champion_conformal_CN.ipynb` — champion-model conformal comparison.
11. `04_generalization/01_region_selection_and_cache_CN.ipynb`, then `02_champion_models_spatial_generalization_CN.ipynb` — external-region transfer.

## Shared experimental rules

- A forecast case contains 24 observed hours and the following 72 target hours.
- Complete drifter IDs, not individual windows, define training/calibration/test partitions.
- Hyperparameters are selected with drifter-grouped cross-validation inside the training partition.
- The 769-case calibration partition is reserved for RQ3.
- The 757-case held-out test partition is used for final evaluation only.
- All point-forecast models are compared using identical cases and Haversine separation errors.
- RMSE emphasises large errors; median error describes the typical case; ITE summarises the full 72-hour trajectory.

## Where generated files go

| Analysis | Output directory |
|---|---|
| RQ1 models and champions | `output/rq1/en/` and `output/rq1/zh/` |
| Champion drogue analysis | `output/rq1/zh/drogue_champion_analysis/` |
| RQ2 core outputs | `RQ2/data/`, `RQ2/figures/` and `RQ2/champion_models/output/` |
| RQ3 outputs | `output/rq3/` and `RQ3/champion_models/output/` |
| External spatial generalization | `output/generalization/champion_spatial/` |

Large raw/processed data and the 62 MB external-region cache remain local and are intentionally not committed. The compact CSV/PNG results needed to inspect the conclusions are tracked.

## Running notebooks

Start Jupyter from the repository root so relative paths resolve correctly:

```bash
source .venv/bin/activate
jupyter lab
```

Open the relevant notebook and use **Run → Run All Cells**. Some notebooks reuse existing model-output CSV files and finish quickly; model-training and external-data notebooks take longer. Read the folder-specific README before rerunning an analysis.
