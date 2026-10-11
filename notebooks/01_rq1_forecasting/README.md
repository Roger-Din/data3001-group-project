# RQ1 - Forecasting accuracy

This folder contains the model-ready forecasting workflow for RQ1:

1. construct 24-step inputs and 72-step targets;
2. implement velocity-persistence and spatial mean-flow baselines;
3. train the selected statistical or machine-learning model;
4. evaluate all frozen methods on the held-out test drifters;
5. report separation error, RMSE, integrated trajectory error and forecast skill.

Only training drifters may be used for model fitting and feature estimation. Calibration data may be used for model selection or uncertainty calibration; test data remains untouched until final evaluation.

## Notebooks

- `01_velocity_persistence_baseline.ipynb`: constructs all 72 lead times, integrates the origin velocity, calculates Haversine separation error, reports held-out test metrics and exports poster-ready figures.
