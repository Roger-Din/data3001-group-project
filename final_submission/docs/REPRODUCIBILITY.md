# Reproducibility guide

## Two levels of reproduction

**Evidence reproduction** is the default. It recomputes every headline statistic from frozen case-level predictions and verifies the split, notebooks and final tables. This is fast and is run with:

```bash
python run_reproduction.py --mode verify
```

**Model reproduction** reruns the selected notebook/script pipelines. It is slower because grouped cross-validation and ensembles are refitted. Preview the commands first:

```bash
python run_reproduction.py --mode full --stage all --dry-run
```

## Expected headline results

- Dataset: 562,944 hourly observations; 5,117 forecast cases.
- Split: 3,591 training, 769 calibration, 757 test cases.
- RQ1: FNN 72-hour RMSE 61.69 km; original Random Forest 62.13 km.
- RQ2: FNN 72-hour RMSE 40.83 km on Shelf and 85.86 km in the Kuroshio Core.
- RQ2 extension: Core-only retraining reduces optimised-FNN RMSE from 85.86 to 82.12 km (4.35%).
- RQ3: global 90% conformal coverage 91.02% at 72 hours with 101.87 km radius; Kuroshio global coverage 80.58% and Mondrian coverage 89.93%.

Small last-decimal differences can arise from display rounding. The automated checks use tolerances and the unrounded source files.

## Reproducibility safeguards

- Identical case IDs are used for paired model comparisons.
- Complete drifter IDs are assigned to one partition only.
- Calibration cases are excluded from RQ1 fitting and reserved for RQ3.
- RQ2 keeps the RQ1 FNN frozen and evaluates the same predictions by regime.
- Bootstrap resampling is clustered by drifter ID.
- Later experiments seen after held-out inspection are labelled exploratory.

## Hardware and runtime

Evidence verification should take minutes on a standard laptop. Complete RQ1 grouped-CV searches may take substantially longer depending on CPU, memory and whether every exploratory model is rerun. No GPU is required by the submitted NumPy/scikit-learn pipelines.
