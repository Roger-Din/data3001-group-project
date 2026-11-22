# DATA3001 Surface-Drifter Forecasting Project

This project forecasts 72 hours of surface-drifter motion from the preceding
24-hour trajectory in the East China Sea and evaluates how forecast accuracy
and uncertainty vary across ocean regimes.

> **Marker / reviewer: start in [`final_submission/`](final_submission/).**
> It is the authoritative, English-facing and reproducible submission package.
> The other root-level folders preserve the team's development history and
> should not be used in place of the frozen final results.

## Quick review

```bash
cd final_submission
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_reproduction.py --mode verify
```

Successful verification ends with:

```text
Overall submission validation: PASS
```

For full installation, review and rerun instructions, use the
[`final_submission/README.md`](final_submission/README.md) five-step guide.

## Repository status map

| Status | Location | Meaning |
|---|---|---|
| **Authoritative submission** | `final_submission/` | Frozen inputs, canonical notebooks/scripts, final outputs, report figures and validation checks |
| **Development workspace** | `notebooks/`, root `RQ2/`, root `RQ3/` | Working notebooks and intermediate analyses retained to show project development |
| **Generated / historical outputs** | root `output/` | Earlier model outputs, diagnostics and extension results |
| **Preliminary utilities** | `scripts/preliminary/` | Early data-processing utilities; not the main submission entry point |
| **Compact root data audit** | root `data/` | Data dictionary and compact QC summary; complete frozen processed inputs are under `final_submission/data/` |

The report is still being edited and is intentionally not stored inside the
final submission package.

## Complete repository structure

The tree below shows every functional area without expanding hundreds of result
CSVs, model parameter files and generated figures individually.

```text
data3001-group-project/
├── README.md                              # this repository overview
├── .gitignore
│
├── final_submission/                     # ★ AUTHORITATIVE SUBMISSION
│   ├── README.md                         # instructor five-step run guide
│   ├── requirements.txt                  # Python dependencies
│   ├── environment.yml                   # alternative Conda environment
│   ├── run_reproduction.py               # single verification/rerun entry point
│   ├── reproduction_pipeline.json
│   │
│   ├── 00_shared/
│   │   ├── 00_report_figures.ipynb       # report figures and tables index
│   │   ├── 01_model_development_and_trial_log.ipynb
│   │   │                                  # model selection and trial-and-error record
│   │   └── notebooks/
│   │       ├── preliminary/              # retained early analysis notebooks
│   │       └── presentation/             # presentation-support notebooks
│   │
│   ├── RQ1/                              # point forecasting and model selection
│   │   ├── notebooks/
│   │   │   ├── core/                     # two baselines + Ridge, FNN, VAR(1), KNN
│   │   │   ├── extensions/               # trees, optimisation and later trials
│   │   │   └── analysis/                 # comparisons, seed and drogue audits
│   │   └── scripts/                      # Python exports matching all RQ1 notebooks
│   │       ├── core/
│   │       ├── extensions/
│   │       └── analysis/
│   │
│   ├── RQ2/                              # regional predictability
│   │   ├── README.md
│   │   ├── notebooks/
│   │   │   ├── core/                     # main regime and lead-time analyses
│   │   │   ├── extensions/               # sensitivity and feedback extensions
│   │   │   └── champion_models/          # selected-model regional analyses
│   │   ├── scripts/                      # numbered reproducible RQ2 pipeline
│   │   ├── data/                         # cached regime/environment inputs
│   │   ├── results/                      # canonical regional result tables
│   │   └── figures/                      # RQ2 diagnostic and report figures
│   │
│   ├── RQ3/                              # conformal uncertainty analysis
│   │   ├── notebooks/
│   │   │   ├── core/                     # global and regime-aware conformal analysis
│   │   │   ├── analysis/                 # validation and multi-nominal checks
│   │   │   └── champion_models/
│   │   ├── scripts/                      # calibration, bootstrap and figure scripts
│   │   ├── config/                       # frozen optimised-FNN parameters
│   │   ├── optimised_fnn/output/         # ★ canonical final RQ3 outputs
│   │   └── results/
│   │       ├── report_tables/            # report-ready tables
│   │       └── exploratory_original_pipeline/
│   │                                          # labelled non-canonical earlier pipeline
│   │
│   ├── data/
│   │   ├── README.md                     # provenance and reconstruction notes
│   │   └── processed/                    # frozen leakage-safe shared inputs
│   ├── data_dictionary/                  # variable dictionary
│   ├── output/rq1/                       # RQ1 predictions, metrics and diagnostics
│   ├── figures/report/                   # report-ready RQ1, RQ2 and RQ3 figures
│   ├── docs/                             # execution order and reproducibility notes
│   ├── scripts/                          # dataset build, checksum and validation tools
│   ├── RESULTS_MANIFEST.csv              # headline claims → machine-readable sources
│   ├── MANIFEST_SHA256.txt               # package integrity checksums
│   ├── notebook_manifest.csv             # canonical notebook inventory
│   └── validation_summary.json           # frozen PASS/FAIL validation summary
│
├── notebooks/                            # DEVELOPMENT WORKSPACE
│   ├── README.md
│   ├── 00_preliminary/                   # data checks and proposal-stage work
│   ├── 01_rq1_forecasting/
│   │   ├── en/                           # English RQ1 development notebooks
│   │   ├── zh/                           # original bilingual development work
│   │   └── analysis/                     # model comparison and drogue diagnostics
│   ├── 02_rq2_regime_analysis/
│   │   ├── core/
│   │   ├── extensions/
│   │   └── figure/
│   ├── 03_rq3_conformal_prediction/
│   └── 04_generalization/                # external zero-shot transfer experiments
│
├── RQ2/                                  # RQ2 WORKING DIRECTORY / HISTORY
│   ├── README.md
│   ├── 01_check_data.py ... 20_regional_lead_time_curves.py
│   ├── champion_models/
│   ├── data/
│   ├── docs/
│   └── figures/
│
├── RQ3/                                  # RQ3 WORKING DIRECTORY / HISTORY
│   ├── champion_models/
│   └── output/
│
├── data/                                 # ROOT-LEVEL COMPACT DATA AUDIT
│   ├── dictionary/
│   └── processed/qc_summary.csv
│
├── output/                               # DEVELOPMENT OUTPUT ARCHIVE
│   ├── rq1/
│   ├── rq3/
│   ├── generalization/
│   └── pdf/
│
├── scripts/preliminary/                  # preliminary utility scripts
└── project_progress_overview.ipynb       # historical project overview
```

## What the marker should review

| Review goal | Recommended file or command |
|---|---|
| Verify all frozen evidence | `cd final_submission && python run_reproduction.py --mode verify` |
| Understand the complete workflow | `final_submission/README.md` |
| See execution order | `final_submission/docs/EXECUTION_ORDER.md` |
| Inspect model trial and selection history | `final_submission/00_shared/01_model_development_and_trial_log.ipynb` |
| Find report figures and tables | `final_submission/00_shared/00_report_figures.ipynb` |
| Trace a numerical claim to its source | `final_submission/RESULTS_MANIFEST.csv` |
| Review RQ1 | `final_submission/RQ1/notebooks/` |
| Review RQ2 | `final_submission/RQ2/notebooks/` and `final_submission/RQ2/results/` |
| Review RQ3 | `final_submission/RQ3/notebooks/` and `final_submission/RQ3/optimised_fnn/output/` |

## Research design

| Item | Frozen project definition |
|---|---|
| Study region | 125–135°E, 25–35°N |
| Data period | 1994–2022 |
| Input | 24 consecutive hourly observations |
| Target | Following 72 hourly positions |
| Complete forecast case | 96 consecutive hourly observations |
| Partition unit | Complete drifter ID |
| Dataset | 562,944 observations; 5,117 forecast cases |
| Split | 3,591 training; 769 calibration; 757 test cases |

Complete drifter IDs stay in one partition, preventing the same physical drifter
from appearing in training and held-out evaluation.

## Research questions

- **RQ1 — Point accuracy:** How does separation error grow over 72 hours, and
  which learned models improve on velocity persistence and spatial mean flow?
- **RQ2 — Regional predictability:** How does forecast performance differ among
  the Shelf, Kuroshio Core, Transition and Other Offshore regimes?
- **RQ3 — Forecast uncertainty:** Can global and regime-aware conformal regions
  achieve reliable empirical coverage without becoming operationally excessive?

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
is statistically competitive. Later tuning and diagnostic work is retained in
the submission but explicitly labelled exploratory where held-out information
had already been inspected.

## Data source

The project uses the NOAA Global Drifter Program one-hour interpolated
quality-controlled dataset:

- [NOAA hourly drifter data](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php)
- [NOAA NCEI dataset record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3AAOML-GDP-1hr)
- Dataset DOI: [10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)

Large raw upstream archives are not duplicated. The processed inputs necessary
to verify and rerun the submitted analyses are included under
`final_submission/data/processed/`, with provenance notes in
`final_submission/data/README.md`.

## Reproducibility principles

- Training, calibration and test drifters are disjoint.
- Normalisation, model selection and mean-flow estimation use training data only.
- Calibration cases are reserved for RQ3 conformal calibration.
- Every method uses identical held-out forecast cases for paired comparison.
- Drifter-clustered resampling preserves within-drifter dependence.
- Confirmatory frozen results are separated from later exploratory development.
- SHA256 checksums identify missing or changed submission files.

## Responsible interpretation

The results are retrospective evaluations on historical drifter data. They do
not constitute an operational search-and-rescue system. Operational use would
require real-time forcing, prospective validation and additional safety review.

## Reference

S. Elipot, R. Lumpkin, R. C. Perez, J. M. Lilly, J. J. Early, and A. M.
Sykulski, “A global surface drifter data set at hourly resolution,” *Journal of
Geophysical Research: Oceans*, 121(5), 2937–2966, 2016.
[doi:10.1002/2016JC011716](https://doi.org/10.1002/2016JC011716)
