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

FNN currently has the lowest held-out error. Weighted KNN is a close nonparametric comparator, while Ridge provides the strongest simple linear learned benchmark.

Generated files are documented in [`../../output/rq1/README.md`](../../output/rq1/README.md).
