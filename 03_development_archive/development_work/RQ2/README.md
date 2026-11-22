# RQ2 directory guide / RQ2 文件导航

This directory contains the complete regional-predictability workflow. The
numbered Python files at the top level are the reproducible pipeline; teaching
and interpretation notebooks are grouped under `notebooks/`.

本目录包含RQ2的完整区域可预测性分析。根目录中的编号Python文件是可复现主流程；用于阅读和解释的notebook统一放在`notebooks/`中。

## Start here / 建议阅读顺序

1. `notebooks/core/RQ2_complete_analysis_CN.ipynb` — RQ2完整中文讲解。
2. `notebooks/core/20_regional_lead_time_analysis_CN.ipynb` — 六模型、四区域、1–72小时median与RMSE曲线。
3. `notebooks/extensions/18_regime_conditioned_forecasting_CN.ipynb` — 固定Ridge，比较全局、区域、洋流条件化和残差策略。
4. `notebooks/extensions/19_Kuroshio全模型区域训练_中文.ipynb` — 只用Kuroshio数据重训Ridge、FNN、KNN和VAR(1)。

## Reproducible pipeline / 编号脚本分类

### A. Data and bathymetry: `01`–`05`

- 数据可用性检查、forecast origins地图、GEBCO水深匹配与抽样检查。

### B. GLORYS currents and Kuroshio definition: `06`–`12d`

- 下载和检查GLORYS、构建1994–2022 climatology、识别连通的Kuroshio current components并给case添加洋流特征。

### C. Regime labels: `13`

- 将forecast origins定义为Shelf、Kuroshio Core、Transition或Other Offshore。

### D. Main RQ2 inference: `14`–`17`

- 合并六模型误差、按drifter cluster bootstrap、模型优势和敏感性分析、最终图片。

### E. Extensions: `18`–`20`

- `18`: Ridge策略实验。
- `19`: Kuroshio-only全学习模型重训。
- `20`: 六个全区域模型在四区域中的1–72小时误差曲线。

## Data folders / 数据目录

- `data/forecast_case_*.csv`: case-level bathymetry/current/regime labels.
- `data/rq2_*.csv`: original RQ2 statistical outputs.
- `data/rq2_extension/`: Ridge strategy experiment.
- `data/kuroshio_all_models/`: Kuroshio-only model refits.
- `data/regional_lead_time/`: six-model, four-regime hourly errors and 24/48/72 summaries.
- `data/*.nc`: GEBCO/GLORYS-derived gridded data used by the pipeline.

## Figure folders / 图片目录

- `figures/17_fig*.{png,pdf}`: main RQ2 report figures.
- `figures/rq2_extension/`: Ridge strategy comparison.
- `figures/kuroshio_all_models/`: Kuroshio-only refit comparison.
- `figures/regional_lead_time/`: four-regime median and RMSE growth curves.

## Important interpretation boundary

`20_regional_lead_time_curves.py` evaluates the same globally trained models on
regional test subsets. `19_kuroshio_all_models.py` retrains models using only
Kuroshio training cases. These answer different questions and should not be
presented as the same experiment.
