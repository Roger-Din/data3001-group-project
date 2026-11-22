# DATA3001 Surface-Drifter Forecasting Project

This repository is arranged for assessment. Start with the report, then use the
reproducibility package to verify or rerun the analysis. Historical development
files are retained separately and are not required to reproduce the submitted
results.

## Instructor quick start

1. Open [`01_report/`](01_report/) for the submitted report and its final figures.
2. Open [`02_reproducibility/README.md`](02_reproducibility/README.md) for the
   complete verification and rerun instructions.
3. From `02_reproducibility/`, run:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   python run_reproduction.py --mode verify
   ```

   A successful check ends with `Overall submission validation: PASS`.

## Repository structure

```text
data3001-group-project/
├── README.md
├── 01_report/
│   ├── README.md
│   └── figures/
│       ├── RQ1/
│       ├── RQ2/
│       └── RQ3/
├── 02_reproducibility/
│   ├── README.md
│   ├── run_reproduction.py
│   ├── 00_shared/
│   ├── RQ1/
│   ├── RQ2/
│   ├── RQ3/
│   ├── data/
│   ├── output/
│   ├── scripts/
│   └── validation/
└── 03_development_archive/
    └── README.md
```

## Project definition and headline results

- Region: 125–135°E, 25–35°N, using NOAA Global Drifter Program data.
- Forecast design: 24 hourly observations predict the following 72 hours.
- Leakage-safe split: 3,591 training, 769 calibration and 757 test cases.
- RQ1: frozen FNN 72-hour RMSE = 61.69 km; original Random Forest = 62.13 km.
- RQ2: Shelf / Kuroshio Core FNN RMSE = 40.83 / 85.86 km.
- RQ3: global conformal coverage at 72 hours = 91.02%.

The exact machine-readable evidence is documented in
[`02_reproducibility/RESULTS_MANIFEST.csv`](02_reproducibility/RESULTS_MANIFEST.csv).
