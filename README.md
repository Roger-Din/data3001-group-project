# DATA3001 Project Proposal

## Predicting Where a Drifter Goes Next

**Assessment:** Project proposal / README (Group, 15%, Week 4)<br>
**Project direction:** IV — Predicting where a drifter goes next<br>
**Study region:** Revised Kuroshio Extension (30–40°N, 142–152°E)<br>
**Proposal status:** Week 4 proposal; model results are not yet reported

## 1. Executive summary

This project will predict the future location of a surface-ocean drifter from its recent trajectory. Given the preceding 24 hours of a drifter track, we will forecast its position at lead times from 1 hour to 7 days (168 hours). This directly addresses the project prompt's examples of an hour, a day and a week.

We will compare a learned gradient-boosting model against two physically interpretable baselines required by the project brief:

1. **Persistence:** the drifter continues at its most recently observed velocity.
2. **Mean-flow advection:** the drifter follows the historical mean flow at its location and time of year.

The final product will be a forecast evaluation on completely held-out drifters. All models will be evaluated on the same forecast origins, and their position error will be plotted against forecast lead time on the same axes. The train, validation and test division will be made by drifter ID rather than by individual rows, preventing observations from the same drifter from leaking across data splits.

## 2. Motivation

Forecasting drifting-object trajectories is relevant to search and rescue, pollutant and debris tracking, and the interpretation of Lagrangian ocean observations. The task is difficult because a drifter's motion reflects spatially varying currents, eddies, tides, wind and waves. Forecast error also accumulates with lead time.

The NOAA Global Drifter Program (GDP) provides quality-controlled hourly locations and velocity estimates for satellite-tracked surface drifters. This makes it possible to construct a supervised-learning problem in which recent motion is used to predict future displacement, while retaining simple physical baselines for comparison.

## 3. Research question and objectives

### Primary research question

> Given 24 hours of recent trajectory information, can a gradient-boosting model predict the position of a drifter in the Kuroshio Extension up to 7 days ahead more accurately than constant-velocity persistence and historical mean-flow advection when evaluated on unseen drifters?

### Secondary questions

- How does great-circle separation error grow with forecast lead time?
- At which lead times, if any, does the learned model improve on each baseline?
- How does forecast performance vary by season and drogue status?
- Are the model's error estimates well calibrated across lead times?

### Objectives

- Build a reproducible pipeline for extracting continuous trajectory windows.
- Implement the persistence and mean-flow baselines before fitting a learned model.
- Train a gradient-boosted regression model using only information available at each forecast origin.
- Evaluate every method on the same held-out drifters and forecast origins.
- Report both overall forecast accuracy and variation across lead time, season and drifter condition.

### Working hypotheses

- **H1:** Position error will increase with forecast lead time for every method.
- **H2:** The learned model will have lower median separation error than persistence and mean-flow advection at medium-to-long lead times (24–168 hours).
- **H3:** Forecasts will be less accurate in dynamically energetic parts of the Kuroshio Extension and after drogue loss.

These are hypotheses to test, not claims about results already obtained.

## 4. Data

### 4.1 Source

The project uses the NOAA Global Drifter Program one-hour interpolated quality-controlled dataset. It contains hourly latitude, longitude, eastward and northward velocity, sea-surface temperature and associated uncertainty variables, together with trajectory-level metadata. NOAA provides the data through NCEI, ERDDAP and cloud access.

- [NOAA hourly drifter data](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php)
- [NOAA NCEI dataset record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3AAOML-GDP-1hr)
- Dataset DOI: [10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)

The complete dataset used during initial exploration contains approximately 197.2 million hourly observations from 19,396 trajectories and is about 16 GB when opened through `xarray`. Only the selected regional subset will be loaded for modelling.

### 4.2 Selected region

The proposed study region is the **revised Kuroshio Extension box**:

| Boundary | Value |
|---|---:|
| Longitude | 142°E to 152°E |
| Latitude | 30°N to 40°N |
| Observed time range in current extract | March 1990 to October 2022 |

This region was chosen after applying the same screening pipeline to four candidate boxes. It provides the strongest overall balance of observation volume, number of drifters, long continuous segments and potential forecast origins.

### 4.3 Preliminary feasibility results

The following figures are descriptive screening results from the current position-only regional extract, not model-performance results:

| Measure | Revised Kuroshio Extension |
|---|---:|
| Valid hourly position records | 798,679 |
| Unique drifters | 516 |
| Exactly one-hour consecutive intervals | 99.76% |
| Continuous trajectory segments | 2,451 |
| Segments at least 96 hours long | 1,218 |
| Segments at least 168 hours long | 986 |
| Segments at least 192 hours long | 924 |
| Segments at least 288 hours long | 730 |
| Drifters contributing at least one usable 192-hour window | 450 |
| Approximate weekly forecast origins under preliminary 192-hour spacing | 3,328 |

A 192-hour segment is the minimum needed for the proposed 24-hour lookback and 168-hour forecast horizon. The preliminary 192-hour spacing prevents the target periods of adjacent weekly evaluation windows from overlapping. It is a screening assumption, not a claim of statistical independence; the final spacing will also consider the empirical velocity autocorrelation or Lagrangian integral time scale.

The ten per cent of drifters with the most candidate origins currently contribute about 36% of those origins. Consequently, results will be aggregated and uncertainty will be estimated at the drifter level, rather than treating every hourly row as an independent observation.

### 4.4 Variables

| Variable | Role in the project |
|---|---|
| `ID` / `id` | Grouping key for trajectories and leakage-safe splitting |
| `time` | Ordering, seasonal features and forecast lead time |
| `latitude`, `longitude` / `lat`, `lon` | Targets, trajectory features and position-error calculation |
| `ve`, `vn` | Velocity features and persistence baseline |
| `sst` | Optional environmental feature after quality control |
| `drogue_lost_date` or derived drogue status | Stratification and optional predictor |
| `err_lat`, `err_lon`, `err_ve`, `err_vn` | Observation-quality checks and sensitivity analysis |
| location and quality flags | Exclusion or sensitivity rules where available |

The current regional CSV files contain only ID, time, latitude and longitude. The velocity, drogue and uncertainty variables will therefore be obtained from the same NOAA dataset before model fitting.

## 5. Forecast design

### 5.1 Prediction unit

For each eligible forecast origin at time \(t\):

- input: observations from \(t-23\) hours through \(t\), giving 24 hourly positions;
- target: the drifter position at future lead times \(h=1,\ldots,168\) hours;
- headline checkpoints: 1 hour, 24 hours and 168 hours, matching the project prompt.

Trajectories will be represented in local east-north displacement coordinates relative to the forecast origin. This avoids treating longitude and latitude as ordinary Cartesian responses. Predicted displacements will be converted back to geographic coordinates for evaluation.

### 5.2 Data cleaning and window construction

1. Select observations within the study box and required time range.
2. Parse timestamps in UTC, sort by drifter ID and time, and remove exact duplicates.
3. Remove records with invalid coordinates or unusable quality flags.
4. Split a trajectory whenever adjacent timestamps are not exactly one hour apart. We will not interpolate across missing intervals.
5. Construct windows only inside continuous segments long enough for both lookback and target periods.
6. Derive motion features using only data at or before the forecast origin.
7. Retain drogue status and observation uncertainty for subgroup and sensitivity analyses.

Requiring the observed positions to remain inside the box for the entire target period could create boundary-selection bias. The intended extraction will therefore identify forecast origins inside the study region and retain their subsequent targets even if they leave the box, where data access permits.

### 5.3 Baseline 1: constant-velocity persistence

The persistence forecast assumes that the latest observed velocity remains constant:

\[
\widehat{\mathbf{x}}_{t+h}=\mathbf{x}_t+h\mathbf{v}_t.
\]

The primary implementation will use NOAA's most recent quality-controlled `ve` and `vn`. A position-derived velocity from the final valid displacements will be used as a documented fallback and sensitivity check.

### 5.4 Baseline 2: mean-flow advection

The mean-flow forecast will advect the drifter through an estimate of the historical mean velocity field. The field will be calculated from training drifters only, using spatial cells and month or season. At each integration step, the forecast position determines the next local mean velocity.

Sparse cells will use a pre-specified fallback hierarchy: coarser spatial cell, regional seasonal mean, then regional overall mean. Cell size and integration step will be selected on the validation data, never on the test set.

### 5.5 Learned model

The proposed model is gradient-boosted regression. Separate eastward and northward displacements will be predicted at each required horizon. Candidate features include:

- recent eastward and northward displacements;
- latest, mean and variability of recent velocity;
- recent acceleration and turning angle;
- forecast-origin latitude and longitude;
- cyclic hour-of-day and month-of-year encodings;
- drogue status;
- sea-surface temperature and recent SST change, where sufficiently complete;
- observation-error summaries.

The initial feature set will be deliberately compact. Additional features will be accepted only when they improve validation performance and are available at forecast time. Hyperparameters, spatial resolution and feature choices will be selected using the training and validation data only.

### 5.6 Optional uncertainty estimates

If time permits, quantile gradient boosting or residual quantiles computed from validation drifters will provide prediction intervals. Interval coverage and width will be reported by lead time. This component is secondary to the required point-forecast comparison.

## 6. Train, validation and test strategy

Random row-level splitting would leak nearly identical observations from the same trajectory into multiple sets. The split will therefore be performed once at the **drifter-ID level**:

| Split | Proposed share of drifters | Purpose |
|---|---:|---|
| Training | 60% | Fit the mean-flow field and learned model |
| Validation | 20% | Select features, hyperparameters and design choices |
| Test | 20% | One final, untouched evaluation |

The allocation will be reproducible with a fixed seed and stratified as far as practical by observation era and number of usable windows. No test drifter may contribute to feature normalisation, climatology construction, hyperparameter selection, missing-value rules or stopping decisions.

During development, grouped cross-validation within the combined training/validation pool will be used to check that conclusions are not an artefact of one split. The held-out test set will remain sealed until the analysis pipeline is frozen.

Forecast origins from a given drifter are correlated. We will estimate the velocity decorrelation scale from training data and thin origins accordingly for the main evaluation. Sensitivity to the thinning interval will also be reported.

## 7. Evaluation plan

### 7.1 Primary metric

The primary error at lead time \(h\) is great-circle separation distance in kilometres between predicted and observed positions:

\[
e_h = 2R\arcsin\!\left(\sqrt{\sin^2\!\left(\frac{\Delta\phi}{2}\right)+
\cos(\phi)\cos(\widehat{\phi})\sin^2\!\left(\frac{\Delta\lambda}{2}\right)}\right),
\]

where \(R\) is Earth's mean radius, \(\phi\) is latitude and \(\lambda\) is longitude.

### 7.2 Required comparison

The main figure will show error against lead time for all three methods on the same held-out origins:

- constant-velocity persistence;
- mean-flow advection;
- gradient-boosting model.

The plot will show median separation error at every lead hour from 1 to 168, with drifter-clustered 95% bootstrap confidence intervals. Mean error and the 25th, 75th and 90th percentiles will be supplied in a table or supplementary figure because the error distribution is likely to be skewed.

### 7.3 Additional analyses

- Skill relative to each baseline: \(1-E_{model}/E_{baseline}\).
- Error at 1, 6, 12, 24, 48, 72 and 168 hours, with 1, 24 and 168 hours treated as the headline results.
- Paired error differences on identical test origins.
- Performance by season, drogue status and spatial subregion.
- Per-drifter error distributions so long-lived drifters do not dominate conclusions.
- If prediction intervals are produced: empirical coverage and interval width by lead time.

Any subgroup with too few independent drifters will be labelled exploratory rather than used for strong conclusions.

## 8. Risks and mitigation

| Risk | Consequence | Mitigation |
|---|---|---|
| Leakage between overlapping windows | Over-optimistic test error | Split by drifter before constructing modelling datasets; thin evaluation origins |
| Unequal contribution by drifter | A few long tracks dominate metrics | Aggregate by drifter and use clustered bootstrap intervals |
| Missing hours | Invalid input/target alignment | Break tracks at every non-hourly gap; do not bridge gaps |
| Boundary truncation | Biased trajectories near the region edge | Select origins in the box but retain valid future positions outside it |
| Drogue loss changes dynamics | Mixed physical regimes | Include/stratify by drogue status and run sensitivity analyses |
| Mean-flow leakage | Test information enters a baseline | Estimate the flow field from training drifters only |
| Non-stationarity across 1990–2022 | Historical performance may not transfer | Stratify split by era and report performance by period where sample size allows |
| Sparse early years or spatial cells | Unstable estimates | Use minimum-count rules and documented fallback pooling |
| Forecasts cross longitude conventions | Incorrect distances | Standardise longitude representation and use spherical distance calculations |

## 9. Expected outputs

The completed project will provide:

1. A reproducible data-cleaning and trajectory-window pipeline.
2. A documented drifter-level train/validation/test split.
3. Implementations of persistence and mean-flow advection.
4. A gradient-boosted trajectory forecast model.
5. The required held-out forecast-error-versus-lead-time figure with all methods on the same axes.
6. Tables of overall and subgroup performance with drifter-level uncertainty.
7. A concise discussion of limitations, practical relevance and future improvements.

Success does not require the learned model to win at every horizon. A valid outcome is a leakage-safe comparison that identifies where added model complexity does or does not improve on the physical baselines.

## 10. Work plan

| Stage | Planned work | Output |
|---|---|---|
| Week 4 | Finalise question, region, feasibility analysis and proposal | Approved README/proposal |
| Week 5 | Re-query selected variables; complete QC and continuous-segment construction | Analysis-ready trajectory data |
| Week 6 | Freeze drifter split; implement persistence and mean-flow baselines | Baseline forecasts and tests |
| Week 7 | Engineer track features and train gradient-boosting models | Validated learned model |
| Week 8 | Run sealed test evaluation and uncertainty analysis | Main results figure and tables |
| Week 9 | Sensitivity analysis, interpretation and limitations | Draft report/notebook |
| Final week | Reproducibility check, editing and presentation preparation | Final submission |

## 11. Team responsibilities

The final allocation should be confirmed by the group before submission. The role structure below separates ownership while retaining peer review of all outputs.

| Team member | Primary responsibility | Review responsibility |
|---|---|---|
| Roger Ding | Data access, regional screening and reproducible trajectory pipeline | Baselines and documentation |
| *Add member name* | Persistence and mean-flow baseline implementation | Data quality checks |
| *Add member name* | Feature engineering and gradient-boosting model | Evaluation code |
| *Add member name* | Statistical evaluation, visualisation and report integration | Model interpretation |

All members will contribute to the research decisions, interpretation, final README/report and presentation. Code changes will be made through individual Git branches and reviewed before merging.

## 12. Reproducibility and repository structure

```text
data3001-project/
├── README.md                         # Week 4 project proposal
├── 01_data_check_clean.ipynb        # Dataset overview and early feasibility work
├── explore.ipynb                    # Reproducible regional screening notebook
├── explore.py                       # Script version of regional screening
├── proposal_method_and_initial_analysis.ipynb
│                                     # Preliminary Region A QC and 24 h-to-72 h feasibility analysis
├── data/
│   ├── regions/                     # Candidate regional extracts (not committed)
│   └── processed/
│       └── qc_summary.csv           # Compact audit trail of Region A QC and sample construction
└── data_dictionary/
    └── gdp1h_variable_dictionary.csv
```

Large source and derived data files will not be committed unnecessarily. Random seeds, package versions, geographic bounds, filtering rules, split IDs and model settings will be recorded. A fresh run should be able to recreate every reported table and figure from documented inputs.

### Preliminary Region A analysis files

`proposal_method_and_initial_analysis.ipynb` documents the earlier Region A feasibility study using 125–135°E, 25–35°N and a 24-observation input / 72-observation target design. It explains the source variables, quality checks, continuous-segment construction, leakage-safe drifter split, candidate forecasting methods and initial visualisations. Its numerical results are calculated from the local `RegionA.csv` file.

`data/processed/qc_summary.csv` is the compact audit table produced by the Region A processing workflow. It records row counts, duplicate and missing-value checks, retained QC flags, segment availability, velocity autocorrelation, the selected forecast-origin spacing and the resulting train/calibration/test counts.

These files are retained as reproducible preliminary work. They should not be confused with the revised Kuroshio Extension 24-hour-to-168-hour design described in the main proposal above. The raw `RegionA.csv` and generated Parquet tables are intentionally excluded from Git because they are large derived data files.

## 13. Ethical and responsible use

The source data are observational and contain no personal information. The main responsibility is scientific validity: avoiding data leakage, communicating forecast uncertainty and not presenting retrospective test performance as guaranteed operational skill. Search-and-rescue or pollution-response use would require operational validation beyond the scope of this coursework.

## 14. References

- Elipot, S., Lumpkin, R., Perez, R. C., Lilly, J. M., Early, J. J., & Sykulski, A. M. (2016). A global surface drifter data set at hourly resolution. *Journal of Geophysical Research: Oceans, 121*, 2937–2966. [https://doi.org/10.1002/2016JC011716](https://doi.org/10.1002/2016JC011716)
- Elipot, S., Sykulski, A., Lumpkin, R., Centurioni, L., & Pazos, M. (2022). A dataset of hourly sea surface temperature from drifting buoys. *Scientific Data, 9*, 567. [https://doi.org/10.1038/s41597-022-01670-2](https://doi.org/10.1038/s41597-022-01670-2)
- Elipot, S., Sykulski, A., Lumpkin, R., Centurioni, L., & Pazos, M. (2022). *Hourly location, current velocity, and temperature collected from Global Drifter Program drifters world-wide*. NOAA National Centers for Environmental Information. [https://doi.org/10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)
