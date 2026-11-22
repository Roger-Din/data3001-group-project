# RQ1 forecasting notebooks / RQ1 轨迹预测 notebooks

This folder contains the complete, leakage-safe workflow for Research Question 1. The notebooks are available in two matched language versions with identical code, data partitions and numerical results.

本目录包含研究问题 RQ1 的完整、防止数据泄漏的预测流程。中英文版本使用相同代码、相同数据划分和相同数值结果。

## Language versions / 语言版本

- [`en/`](en/): English notebooks and English figures for teaching staff and formal reporting.
- [`zh/`](zh/): 中文讲解、中文图表，用于组内学习和交流。

Run the notebooks in numerical order within one language folder. Each language writes to its own result directory, so running one version never overwrites the other.

请在同一个语言文件夹内按编号运行。中英文输出相互独立，不会互相覆盖。

## Notebook sequence

1. `01_velocity_persistence_baseline.ipynb` — constant forecast-origin velocity baseline.
2. `02_spatial_mean_flow_baseline.ipynb` — training-only multiresolution spatial mean-flow baseline.
3. `03_multioutput_ridge_trajectory_model.ipynb` — 98-input, 144-output Ridge model with drifter-grouped cross-validation.
4. `04_fnn_trajectory_model.ipynb` — nonlinear fully connected neural network implemented in NumPy.
5. `05_pooled_var1_velocity_model.ipynb` — pooled first-order vector autoregression fitted to training-only velocity transitions.
6. `06_weighted_knn_trajectory_regression.ipynb` — nonparametric weighted analogue forecast with grouped selection of neighbour count and distance power.
7. `07_extra_trees_trajectory_model_CN.ipynb` — Extra Trees trajectory regression experiment.
8. `08_random_forest_trajectory_model_CN.ipynb` — Random Forest trajectory regression experiment.
9. `09_physical_residual_extra_trees_CN.ipynb` — physical-baseline residual learning with Extra Trees.
10. `10_optimised_random_forest_CN.ipynb` — grouped-CV Random Forest optimisation.
11. `11_optimised_fnn_CN.ipynb` — grouped-CV FNN architecture and regularisation optimisation.
12. `12_optimised_weighted_knn_CN.ipynb` — weighted KNN distance and neighbour optimisation.
13. `13_gradient_boosting_residual_CN.ipynb` — persistence-residual gradient boosting and extreme-case diagnostics.
14. `14_champion_models_comparison_CN.ipynb` — common-metric comparison of the four champion models and physical baselines.

## Fixed experimental protocol

- Input: 24 hourly observations ending at the forecast origin.
- Output: east/north displacement at each of the next 72 hours.
- Cases: 3,591 training, 769 calibration and 757 test cases.
- Partitioning: complete drifter IDs, never individual observations or forecast windows.
- Model selection: grouped folds inside the training partition only.
- Calibration: deliberately unused here and reserved for RQ3 conformal prediction.
- Final evaluation: the same 757 held-out test cases and Haversine separation error for every method.

## Current held-out results

| Method | 24 h median (km) | 48 h median (km) | 72 h median (km) | 72 h RMSE (km) |
|---|---:|---:|---:|---:|
| Velocity persistence | 16.47 | 36.54 | 61.35 | 90.10 |
| Spatial mean flow | 21.69 | 39.96 | 56.42 | 80.98 |
| Pooled VAR(1) | 16.00 | 36.27 | 55.89 | 77.89 |
| Multi-output Ridge | 12.32 | 28.27 | 44.75 | 65.91 |
| Weighted KNN | 12.03 | 27.00 | 41.99 | 65.21 |
| FNN | **11.32** | **26.10** | **41.72** | **62.15** |

The original FNN has the lowest error among the six initial models. The extension notebooks add four champion candidates: Optimised FNN, Optimised Weighted KNN, Optimised Random Forest and Boosting residual. Model selection remains based on drifter-grouped cross-validation inside the training partition; the held-out test set is used only for the final comparison.

Generated files are documented in [`../../output/rq1/README.md`](../../output/rq1/README.md).

## Folder contents

| Subfolder/file | Purpose |
|---|---|
| `en/` | English versions of the six original models. |
| `zh/` | Chinese versions, tree-model extensions, optimised champions and the final champion leaderboard. |
| `analysis/` | Cross-model diagnostics that consume saved test predictions rather than selecting new hyperparameters. |
| `extensions/` | Time-series experiments such as AR/VAR order search. |
| `RQ1_training_and_test_audit_CN.ipynb` | Verifies partition roles and confirms that the comparison uses the same 757 test cases. |

## Champion-model extensions

- `zh/10_optimised_random_forest_CN.ipynb` — two-stage grouped-CV Random Forest search.
- `zh/11_optimised_fnn_CN.ipynb` — FNN architecture, regularisation and training optimisation.
- `zh/12_optimised_weighted_knn_CN.ipynb` — neighbour count, distance power and spatial weighting search.
- `zh/13_gradient_boosting_residual_CN.ipynb` — learns corrections to a physical baseline and diagnoses extreme errors.
- `zh/14_champion_models_comparison_CN.ipynb` — compares RMSE, median error, ITE and worst-10% performance on common cases.
- `analysis/15_champion_models_drogue_effect_CN.ipynb` — tests attached versus lost drogue cases for all four champions.

The drogue notebook first reports the raw difference, then uses drifter-cluster bootstrap and an adjusted association controlling for regime, climatological current speed, water depth and forecast-origin speed. Lost-drogue cases have larger raw errors, but all adjusted 95% intervals include zero; the current evidence is therefore insufficient to claim that drogue status independently causes poorer forecasts.

Its compact results are stored in [`../../output/rq1/zh/drogue_champion_analysis/`](../../output/rq1/zh/drogue_champion_analysis/).
