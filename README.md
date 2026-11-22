# DATA3001 Surface-Drifter Forecasting Project

This project forecasts 72 hours of surface-drifter motion from the preceding
24-hour trajectory in the East China Sea. It compares physical and learned point
forecasts, studies regional predictability, and evaluates conformal uncertainty
regions.

## Repository layout

The repository is deliberately divided into only two main folders:

```text
data3001-group-project/
├── README.md
├── .gitignore
│
├── final_submission/          # ★ frozen, reproducible, marker-facing package
│   ├── README.md              # five-step instructor run guide
│   ├── run_reproduction.py    # verify or rerun the submitted analyses
│   ├── requirements.txt
│   ├── 00_shared/             # report figures and model-development log
│   ├── RQ1/                   # point forecasting and model selection
│   ├── RQ2/                   # regional predictability
│   ├── RQ3/                   # conformal uncertainty analysis
│   ├── data/processed/        # frozen leakage-safe shared inputs
│   ├── output/rq1/            # RQ1 predictions and metrics
│   ├── figures/report/        # report-ready figures
│   ├── docs/                  # execution and reproducibility documentation
│   ├── RESULTS_MANIFEST.csv   # final claims mapped to source files
│   ├── MANIFEST_SHA256.txt    # package integrity checksums
│   └── validation_summary.json
│
└── development_work/          # project process, experiments and historical files
    ├── README.md              # development archive guide
    ├── notebooks/             # preliminary, RQ1, RQ2, RQ3 and transfer work
    ├── RQ2/                   # original RQ2 scripts, data and figures
    ├── RQ3/                   # original RQ3 scripts and outputs
    ├── data/                  # compact development data audit
    ├── output/                # historical generated outputs
    ├── scripts/preliminary/   # early processing utilities
    └── project_progress_overview.ipynb
```

## Which folder should be used?

| User | Start here | Purpose |
|---|---|---|
| Marker / instructor | [`final_submission/`](final_submission/) | Review and reproduce the frozen final analysis |
| Team member reviewing the process | [`development_work/`](development_work/) | Inspect earlier notebooks, experiments and intermediate outputs |

The group report is still being edited and is intentionally not included in the
repository's final reproducibility package.

## Quick verification for the marker

From the repository root:

```bash
cd final_submission
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_reproduction.py --mode verify
```

A successful check ends with:

```text
Overall submission validation: PASS
```

For full installation, review order, expected outputs and model rerun commands,
read the [`final_submission/README.md`](final_submission/README.md) instructor
guide.

## Final-submission contents

### Shared evidence

- `00_shared/00_report_figures.ipynb` collects the report figures and tables.
- `00_shared/01_model_development_and_trial_log.ipynb` documents model
  selection, unsuccessful trials and later exploratory work.

### RQ1 — Point forecast accuracy

Contains the two physical baselines, pooled VAR(1), Ridge, weighted KNN, FNN,
Extra Trees, Random Forest, optimisation experiments and robustness diagnostics.
Every canonical RQ1 notebook has a corresponding Python export.

### RQ2 — Regional predictability

Contains the ocean-regime definitions, regional error analyses, normalised skill,
lead-time comparisons and Core-only retraining experiments. Cached environmental
inputs needed for the submitted results are included.

### RQ3 — Forecast uncertainty

Contains global and regime-aware conformal calibration, coverage validation,
clustered bootstrap comparisons and report-ready uncertainty tables and figures.
The canonical outputs are under `RQ3/optimised_fnn/output/`.

## Frozen project design

| Item | Definition |
|---|---|
| Study region | 125–135°E, 25–35°N |
| Data period | 1994–2022 |
| Input | 24 consecutive hourly observations |
| Target | Following 72 hourly positions |
| Complete forecast case | 96 consecutive hourly observations |
| Partition unit | Complete drifter ID |
| Dataset | 562,944 observations; 5,117 forecast cases |
| Split | 3,591 training; 769 calibration; 757 test cases |

Complete drifter IDs remain in one partition, preventing the same physical
drifter from appearing in model fitting and held-out evaluation.

## Research questions

- **RQ1:** How does trajectory separation error grow over 72 hours, and which
  learned models improve on velocity persistence and spatial mean flow?
- **RQ2:** How does forecast performance differ among the Shelf, Kuroshio Core,
  Transition and Other Offshore regimes?
- **RQ3:** Can global and regime-aware conformal regions achieve reliable
  empirical coverage without becoming operationally excessive?

## Frozen headline results

| Component | Result |
|---|---:|
| RQ1 selected FNN 72-hour RMSE | 61.69 km |
| RQ1 original Random Forest 72-hour RMSE | 62.13 km |
| RQ2 Shelf FNN RMSE | 40.83 km |
| RQ2 Kuroshio Core FNN RMSE | 85.86 km |
| RQ2 Core-only FNN retraining | 82.12 km; 4.35% reduction |
| RQ3 global 90% conformal coverage at 72 h | 91.02% |
| RQ3 global 72-hour radius | 101.87 km |
| RQ3 Kuroshio Core global / Mondrian coverage | 80.58% / 89.93% |

The selected FNN has the lowest numerical frozen RQ1 RMSE, while Random Forest
is statistically competitive. Later experiments remain visible in the trial log
but are labelled exploratory when held-out information had already been viewed.

## Data source

The project uses the NOAA Global Drifter Program one-hour interpolated
quality-controlled dataset:

- [NOAA hourly drifter data](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php)
- [NOAA NCEI dataset record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3AAOML-GDP-1hr)
- Dataset DOI: [10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)

Large upstream raw archives are not duplicated. The processed inputs required
to verify and rerun the submitted analyses are included under
`final_submission/data/processed/`.

## Reproducibility safeguards

- Training, calibration and test drifters are disjoint.
- Normalisation, model selection and mean-flow estimation use training data only.
- Calibration cases are reserved for RQ3 conformal calibration.
- All models use identical held-out cases for paired comparison.
- Drifter-clustered resampling preserves within-drifter dependence.
- Confirmatory frozen results are separated from later exploratory development.
- SHA256 checksums identify missing or changed submission files.

## Responsible interpretation

These are retrospective results on historical drifter data, not an operational
search-and-rescue system. Operational use would require real-time forcing,
prospective validation and additional safety review.

## Reference

S. Elipot, R. Lumpkin, R. C. Perez, J. M. Lilly, J. J. Early, and A. M.
Sykulski, “A global surface drifter data set at hourly resolution,” *Journal of
Geophysical Research: Oceans*, 121(5), 2937–2966, 2016.
[doi:10.1002/2016JC011716](https://doi.org/10.1002/2016JC011716)
