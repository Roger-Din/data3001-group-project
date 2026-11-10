# RQ2 - Regime-dependent predictability

This folder compares forecast performance across independently defined physical regimes. The four regimes are Shelf, Kuroshio Core, Transition and Other Offshore, derived from bathymetry and long-term GLORYS flow structure rather than from test errors.

Regime definitions must not be derived from test-set forecast errors.

## Folder contents

| Location | Purpose |
|---|---|
| `core/RQ2_complete_analysis_CN.ipynb` | Chinese end-to-end explanation of data checks, GEBCO/GLORYS labels, regime assignment and fixed-model comparison. |
| `core/RQ2_complete_analysis.ipynb` | English counterpart of the core analysis. |
| `core/20_regional_lead_time_analysis_CN.ipynb` | Median error and RMSE curves from lead hour 1 to 72 within each regime. |
| `extensions/18_regime_conditioned_forecasting_CN.ipynb` | Explores regional, current-conditioned and current-residual strategies. |
| `extensions/19_Kuroshio全模型区域训练_中文.ipynb` | Retrains candidate models using Kuroshio-only training data and compares with global training. |
| `extensions/21_teacher_feedback_normalised_evaluation_CN.ipynb` | Adds normalised and relative metrics so fast-flow regimes are not judged only on absolute kilometre error. |
| `extensions/22_champion_models_by_regime_CN.ipynb` | Applies the four frozen RQ1 champions to the same held-out cases and compares them by regime. |

## Interpretation boundary

The core RQ2 question is answered by evaluating the same globally trained models in different regimes. This isolates where a fixed forecasting system is more or less predictable. Regional retraining notebooks answer a different extension question: whether local training improves forecasts within a regime. These two designs should not be mixed in the final interpretation.

## Champion-model extension

- `extensions/22_champion_models_by_regime_CN.ipynb` evaluates the four frozen RQ1 champion models and two physical baselines on the same 757 held-out test cases.
- It reports regime-specific 72-hour median error, RMSE, median ITE, worst-10% RMSE and drifter-cluster bootstrap intervals.
- Generated CSV and PNG outputs are stored in `RQ2/champion_models/output/`.

This is a fixed-model spatial generalisation analysis. It does not refit a separate model inside each regime.

Core figures and tables remain under `RQ2/data/` and `RQ2/figures/` because the numbered scripts use those relative paths. Champion outputs are under `RQ2/champion_models/output/`.
