# DATA3001 Surface-Drifter Forecasting Project

## Predicting the next 72 hours from a 24-hour trajectory

This group project uses hourly observations from the NOAA Global Drifter Program (GDP) to study short-term surface-drifter trajectory forecasting in **Region A**.

| Design item | Current project definition |
|---|---|
| Study region | 125–135°E, 25–35°N |
| Observed time range | 1994–2022 |
| Input | 24 consecutive hourly observations |
| Target | The following 72 hourly positions |
| Complete case | 96 consecutive hourly observations |
| Split unit | Drifter ID |
| Current stage | RQ1--RQ3 complete; champion, drogue and external spatial-generalization extensions evaluated |

All analysis, processing outputs and planned model comparisons in this repository use the **24-hour-to-72-hour Region A design**.

## 1. Research questions and objectives

> How accurately can surface-drifter trajectories in the East China Sea be forecast at lead times of up to 72 hours, and how reliably can the uncertainty of those forecasts be quantified?

- **RQ1.** How does trajectory separation error grow with lead time, and how do learned forecasting methods compare with persistence and mean-flow baselines?
- **RQ2.** Does predictability differ between contrasting dynamical regimes, particularly Kuroshio-influenced and lower-energy waters?
- **RQ3.** Can forecast uncertainty be represented by prediction regions with empirically validated coverage, and does coverage remain reliable across regimes?

Forecast cases use 24 hours of track history to predict the subsequent 72 hours. All methods will be compared on identical held-out drifters and forecast origins. A separate calibration set is reserved for conformal prediction and uncertainty evaluation.

### Current headline findings

- The original FNN is the strongest of the six initial Region-A models (72-hour RMSE 62.15 km).
- Optimised FNN, Optimised Weighted KNN, Optimised Random Forest and Boosting residual form the champion set used for later diagnostics.
- Kuroshio Core has the largest absolute errors, motivating both normalised metrics and regime-specific reliability checks.
- Conformal prediction converts point forecasts into empirically calibrated search regions and reports both coverage and region size.
- Lost-drogue cases have larger raw errors, but adjusted drifter-cluster intervals do not establish an independent drogue-status effect.
- In ten external zero-shot regions, Optimised Random Forest is the only learned champion that reliably beats velocity persistence; the Region-A FNN does not transfer safely without adaptation.

The detailed folder and execution guide is [`notebooks/README.md`](notebooks/README.md).

## 2. Motivation

Forecasting drifting-object trajectories is relevant to search and rescue, pollutant and debris tracking, and understanding near-surface ocean circulation. The problem is difficult because drifter motion reflects spatially varying currents, eddies, tides, wind, waves and changes in drogue status.

The project therefore combines reproducible quality control, leakage-safe data partitioning, simple physical baselines and more flexible forecasting models.

## 3. Data source and variables

The data come from the NOAA Global Drifter Program one-hour interpolated quality-controlled dataset.

- [NOAA hourly drifter data](https://www.aoml.noaa.gov/phod/gdp/hourly_data.php)
- [NOAA NCEI dataset record](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc%3AAOML-GDP-1hr)
- Dataset DOI: [10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)

### Core forecasting variables

| Variable | Meaning | Use |
|---|---|---|
| `ID` | Drifter identifier | Grouping, splitting and leakage prevention |
| `time` | UTC observation time | Ordering and lead-time construction |
| `latitude`, `longitude` | Geographic position | History, target and position-error calculation |
| `ve`, `vn` | Eastward and northward velocity | Motion features and velocity-persistence baseline |

### Quality-control and supporting variables

| Variable | Meaning |
|---|---|
| `err_lat`, `err_lon` | Position uncertainty information |
| `err_ve`, `err_vn` | Velocity uncertainty information |
| `location_type` | GPS or Argos positioning |
| `drogue_lost_date` | Recorded drogue-loss date |
| `DrogueType` | Drogue type metadata |
| `DrogueCenterDepth` | Drogue centre depth |
| `sst`, `flg_sst` | Sea-surface temperature and its quality flag |

## 4. Current Region A dataset

| Quantity | Result |
|---|---:|
| Hourly observations | 562,944 |
| Unique drifters | 521 |
| Time coverage | 1994–2022 |
| Exact duplicate rows | 0 |
| Duplicate `ID + time` rows | 0 |
| Missing core forecasting values | 0 |
| Observations outside Region A after processing | 0 |
| Continuous trajectory segments | 2,058 |
| Segments with at least 96 observations | 939 (45.6%) |

Both GPS and Argos observations are represented. Spatial and temporal coverage are uneven, so later evaluation should report performance across locations, periods and physical regimes rather than relying only on one pooled error value.

## 5. Quality-control approach

The raw regional CSV is preserved. Processing standardises data types, sorts observations by `ID` and `time`, removes unusable key records and adds review flags.

The current QC audit reports:

| Check | Result | Current treatment |
|---|---:|---|
| Speed above 2 m s⁻¹ | 245 observations | Retain and review |
| Speed above 3 m s⁻¹ | 0 observations | Retain and review |
| Land-mask flag | 941 observations | Retain pending coastal review |
| Missing position-error information | 1,052 observations | Retain with missingness flag |
| Missing velocity-error information | 24,632 observations | Retain with missingness flag |

A flag is not an automatic declaration that an observation is wrong. Speeds above 2 m s⁻¹ may be physically plausible in a strong current, while a coarse land mask can misclassify near-coastal observations. The main analysis will be accompanied by sensitivity checks where appropriate.

## 6. Continuous trajectories and forecast cases

Observations are ordered within each drifter. A new segment begins whenever adjacent records are not consecutive hourly observations. No forecast case crosses a time gap.

Each case contains:

- 24 history observations, ending at the forecast origin;
- 72 future target observations;
- 96 total observations, spanning 95 elapsed hours.

An eligible segment must contain at least 96 observations. Hourly sliding across all eligible segments would produce **443,295 potential windows**, but adjacent windows would share almost all their observations and cannot be treated as independent cases.

## 7. Velocity autocorrelation and origin spacing

Origin spacing is supported by a training-only vector velocity autocorrelation analysis. Eastward and northward velocities are demeaned within each eligible continuous segment before their pooled vector autocorrelation is calculated.

| Dependence diagnostic | Result |
|---|---:|
| Lag-1 velocity autocorrelation | 0.953983 |
| First lag at or below \(1/e\) | 32 hours |
| Lagrangian integral timescale \(T_L\) | 29.562 hours |

The approximately 30-hour \(T_L\) is a characteristic velocity-memory timescale, not a claim that observations become perfectly independent after 30 hours. Because a complete forecast case contains 96 observations, the selected origin spacing is

\[
\max(96,\lceil T_L\rceil)=96\text{ hours}.
\]

This removes direct observation overlap between adjacent cases. Residual dependence may still exist between cases from the same drifter and will be handled through drifter-level partitioning and clustered evaluation.

## 8. Final forecast-origin index

Applying 96-hour spacing to the 939 eligible segments produces **5,117 forecast cases**.

All cases have been checked for:

- a unique forecast-case ID;
- exactly 24 history observations;
- exactly 72 target observations;
- no crossing of segment boundaries;
- no direct overlap with the next selected case in the same segment.

Of the 5,117 cases, 5,025 contain neither a high-speed flag nor a land-mask flag. The remaining 92 cases are retained for review and sensitivity analysis rather than automatically discarded.

## 9. Leakage-safe data partitions

Partitioning is performed by complete drifter ID, not by individual rows or forecast windows. All observations from one physical drifter remain in exactly one partition.

| Partition | Assigned drifters | Drifters with usable cases | Eligible segments | Forecast cases |
|---|---:|---:|---:|---:|
| Training | 364 | 311 | 642 | 3,591 |
| Calibration | 78 | 66 | 144 | 769 |
| Test | 79 | 71 | 153 | 757 |
| **Total** | **521** | **448** | **939** | **5,117** |

- **Training:** fit baselines and learned models; use grouped cross-validation within training drifters for model selection.
- **Calibration:** calibrate lead-time-specific conformal prediction regions after the forecasting method is fixed.
- **Test:** perform the final comparison once, after all modelling choices are fixed.

The test drifters must not influence feature normalisation, mean-flow construction, hyperparameter selection, QC thresholds or stopping decisions.

## 10. Planned forecasting methods

### 10.1 Velocity-persistence baseline

Retain the latest observed eastward and northward velocities and integrate them forward over the 72-hour horizon. This is not the same as keeping the drifter stationary.

### 10.2 Spatial mean-flow baseline

Estimate a spatial average velocity field using training drifters only, then advect each case through the corresponding grid-cell flow. Sparse cells will use a documented fallback based on coarser or regional training-only averages.

### 10.3 Learned and statistical methods

Four complementary methods have now been implemented with identical inputs, outputs and held-out evaluation:

- multi-output Ridge as the regularised linear benchmark;
- a fully connected neural network (FNN) for nonlinear trajectory regression;
- pooled VAR(1) for explicit temporal velocity dynamics;
- weighted K-nearest-neighbour trajectory regression as a nonparametric analogue method.

All model selection is performed with drifter-grouped folds inside the training partition. The FNN currently has the lowest held-out error (72-hour median 41.72 km; RMSE 62.15 km), followed closely by weighted KNN and Ridge.

## 11. Evaluation plan

The primary outcome is geographic separation between the predicted and observed position, measured in kilometres. Every method will be evaluated on the same test drifters and forecast origins.

Planned reporting includes:

- separation error at every lead time from 1 to 72 hours;
- headline errors at 24, 48 and 72 hours;
- median, mean and RMSE summaries;
- average or integrated error across the complete 72-hour trajectory;
- forecast skill relative to each baseline;
- optional along-track and cross-track error;
- drifter-clustered uncertainty intervals;
- sensitivity to the 92 cases containing high-speed or land-mask flags.

Where sample sizes permit, performance will also be compared across spatial sectors, observation periods and bathymetry-informed dynamical regimes. Conformal prediction will generate lead-time-specific prediction regions, evaluated using both empirical coverage and prediction-region area.

## 12. Current progress

### Completed

- Region A extraction, quality-control inventory and continuous-segment construction
- Drifter-level training/calibration/test split and non-overlapping forecast origins
- Model-ready 24-hour inputs and 72-hour trajectory targets
- Velocity-persistence and training-only spatial mean-flow baselines
- Multi-output Ridge with grouped penalty selection
- Fully connected neural network with grouped architecture and regularisation selection
- Pooled VAR(1) with training-only velocity transitions and stability diagnostics
- Weighted KNN trajectory regression with grouped neighbour selection
- Common held-out evaluation on 757 test cases using separation error, RMSE and ITE
- Matched English and Chinese notebooks, tables and poster-ready figures
- Tree-based and residual-learning extensions with four champion configurations
- RQ2 fixed-model comparison across four physical ocean regimes
- RQ2 lead-time, normalised-error and regional-refitting extensions
- RQ3 global and regime-aware conformal prediction
- Champion-model drogue-status analysis with adjusted cluster bootstrap
- Ten-region zero-shot spatial-generalization experiment

### Remaining

- Manual review and sensitivity analysis for high-speed and land-mask flags
- Operational validation with real-time environmental forcing
- Replication on additional time periods and external regions
- Final report synthesis and concise uncertainty reporting

## 13. Repository structure

```text
data3001-group-project/
├── README.md
├── notebooks/
│   ├── 00_preliminary/
│   ├── 01_rq1_forecasting/
│   │   ├── README.md
│   │   ├── en/                 # English teacher-facing notebooks (01--06)
│   │   ├── zh/                 # 中文原始模型与冠军模型
│   │   └── analysis/           # cross-model and drogue diagnostics
│   ├── 02_rq2_regime_analysis/ # core regime analysis and extensions
│   ├── 03_rq3_conformal_prediction/ # initial and champion conformal analyses
│   └── 04_generalization/      # external zero-shot transfer
├── scripts/
├── data/
│   ├── dictionary/
│   └── processed/              # Large generated tables remain local
└── output/
    ├── rq1/                    # RQ1 metrics, figures and drogue analysis
    ├── rq3/                    # coverage and prediction-region outputs
    └── generalization/         # external-region metrics and figures
```

Large raw and derived datasets are intentionally excluded from Git. In particular, `RegionA.csv` and generated Parquet tables remain local. The tracked `qc_summary.csv` provides a compact audit trail of the processing results.

The explanatory notebook `notebooks/00_preliminary/proposal_method_and_initial_analysis.ipynb` recalculates the main descriptive findings from the local Region A data and distinguishes completed preprocessing from future modelling work. New work should be saved in the numbered RQ folder that it answers; exploratory or superseded work remains in `notebooks/00_preliminary/`.

## 14. Reproducibility rules

- Keep the raw regional CSV unchanged.
- Record the random seed used for drifter splitting.
- Estimate normalisation, climatology and mean-flow quantities from training drifters only.
- Keep calibration drifters separate if conformal prediction is used.
- Do not inspect test performance while choosing models or thresholds.
- Apply identical forecast origins and evaluation metrics to every method.
- Report cases and drifters separately; do not describe all cases as independent individuals.

## 15. Responsible interpretation

The source data contain no personal information. The main responsibilities are methodological: prevent leakage, represent dependence honestly, report uncertainty and avoid presenting retrospective test performance as guaranteed operational skill. Search-and-rescue or pollution-response use would require operational validation beyond the scope of this coursework.

## 16. References

[1] S. Elipot, R. Lumpkin, R. C. Perez, J. M. Lilly, J. J. Early, and A. M. Sykulski, “A global surface drifter data set at hourly resolution,” *Journal of Geophysical Research: Oceans*, vol. 121, no. 5, pp. 2937–2966, 2016. [doi:10.1002/2016JC011716](https://doi.org/10.1002/2016JC011716)

[2] S. Elipot, A. Sykulski, R. Lumpkin, L. Centurioni, and M. Pazos, “Hourly location, current velocity, and temperature collected from Global Drifter Program drifters world-wide,” NOAA National Centers for Environmental Information, Dataset, 2022. [doi:10.25921/x46c-3620](https://doi.org/10.25921/x46c-3620)

[3] H.-J. Lie and C.-H. Cho, “Seasonal circulation patterns of the Yellow and East China Seas derived from satellite-tracked drifter trajectories and hydrographic observations,” *Progress in Oceanography*, vol. 146, pp. 121–141, 2016. [doi:10.1016/j.pocean.2016.06.004](https://doi.org/10.1016/j.pocean.2016.06.004)

[4] H. S. Huntley, B. L. Lipphardt Jr., and A. D. Kirwan Jr., “Lagrangian predictability assessed in the East China Sea,” *Ocean Modelling*, vol. 36, nos. 1–2, pp. 163–178, 2011. [doi:10.1016/j.ocemod.2010.11.001](https://doi.org/10.1016/j.ocemod.2010.11.001)

[5] M. D. Grossi, S. Jegelka, P. F. J. Lermusiaux, and T. M. Özgökmen, “Surface drifter trajectory prediction in the Gulf of Mexico using neural networks,” *Ocean Modelling*, vol. 196, Art. no. 102543, 2025. [doi:10.1016/j.ocemod.2025.102543](https://doi.org/10.1016/j.ocemod.2025.102543)

[6] Y. Zhou, L. Lindemann, and M. Sesia, “Conformalized adaptive forecasting of heterogeneous trajectories,” in *Proceedings of the 41st International Conference on Machine Learning*, vol. 235, pp. 62002–62056, 2024. [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/zhou24l.html)
