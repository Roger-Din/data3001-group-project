# Development work archive

This folder preserves the files created during data preparation, model
development, experimentation and presentation work. It is included to make the
team's workflow and trial-and-error process transparent.

It is **not** the submission entry point. For the frozen English-facing analysis,
use [`../final_submission/`](../final_submission/) instead.

## Folder map

| Path | Contents |
|---|---|
| `notebooks/00_preliminary/` | Early data checks, proposal analysis and exploratory work |
| `notebooks/01_rq1_forecasting/` | RQ1 model-development notebooks and comparisons |
| `notebooks/02_rq2_regime_analysis/` | RQ2 regime analysis and extensions |
| `notebooks/03_rq3_conformal_prediction/` | RQ3 conformal prediction development |
| `notebooks/04_generalization/` | External zero-shot spatial-transfer experiments |
| `RQ2/` | Original numbered RQ2 scripts, cached inputs, figures and champion analyses |
| `RQ3/` | Original RQ3 scripts, notebooks and output tables |
| `data/` | Compact root-level data dictionary and QC audit |
| `output/` | Historical RQ1, RQ3 and generalisation outputs |
| `scripts/preliminary/` | Early processing utilities |
| `project_progress_overview.ipynb` | Historical project-progress record |

## Important interpretation note

Some files in this archive were created at different stages of the project and
may contain older naming conventions, bilingual notes or paths based on the
former repository layout. They are retained as development evidence, not as the
canonical reproduction workflow.

The following files under `../final_submission/` supersede this archive for
assessment and reproduction:

- `README.md` — instructor run guide;
- `00_shared/01_model_development_and_trial_log.ipynb` — curated trial history;
- `RQ1/`, `RQ2/`, `RQ3/` — canonical English-facing workflows;
- `RESULTS_MANIFEST.csv` — final claims and source files;
- `validation_summary.json` — frozen validation status.
