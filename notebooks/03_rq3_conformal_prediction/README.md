# RQ3 - Forecast uncertainty

## Main notebook

- `RQ3_conformal_prediction_CN.ipynb`: executed Chinese teaching notebook covering the complete RQ3 workflow.
- `RQ3_conformal_prediction_CN.py`: Jupytext source for reproducible editing and regeneration of the notebook.

The notebook uses the fixed RQ1 FNN as the point predictor. The 769 calibration cases determine finite-sample conformal thresholds; the 757 untouched test cases are used once for final empirical coverage and prediction-region size evaluation.

It includes:

1. lead-time-specific 80%, 90% and 95% circular prediction regions;
2. empirical test coverage, radius and spherical-cap area from 1 to 72 h;
3. drifter-cluster bootstrap intervals for coverage;
4. coverage diagnostics across Shelf, Kuroshio Core, Transition and Other Offshore regimes;
5. pointwise versus simultaneous whole-trajectory coverage;
6. a representative trajectory visualization and detailed assumptions/limitations;
7. report-ready English Methods and Results wording.

Generated tables and figures are saved under `output/rq3/`.

## Run

From the project root:

```bash
source .venv/bin/activate
jupyter lab
```

Then open and run `notebooks/03_rq3_conformal_prediction/RQ3_conformal_prediction_CN.ipynb`.
