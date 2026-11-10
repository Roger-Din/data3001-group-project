# RQ2 - Regime-dependent predictability

This folder is reserved for analysis comparing forecast performance across independently defined physical regimes, such as bathymetry-informed or established flow-structure regions.

Regime definitions must not be derived from test-set forecast errors.

## Champion-model extension

- `extensions/22_champion_models_by_regime_CN.ipynb` evaluates the four frozen RQ1 champion models and two physical baselines on the same 757 held-out test cases.
- It reports regime-specific 72-hour median error, RMSE, median ITE, worst-10% RMSE and drifter-cluster bootstrap intervals.
- Generated CSV and PNG outputs are stored in `RQ2/champion_models/output/`.

This is a fixed-model spatial generalisation analysis. It does not refit a separate model inside each regime.
