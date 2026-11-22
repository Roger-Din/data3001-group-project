# RQ3 Results and Limitations

These figures were obtained from the user's successful local DATA3001 run on 2 November 2026. The packaged English-language code has been syntax-checked, but its full numerical outputs must be regenerated and verified locally after replacement.

## Dataset and integrity

- 5,117 forecast cases from 448 usable drifters.
- Training: 3,591 cases, 311 drifters; calibration: 769 cases, 66 drifters; test: 757 cases, 71 drifters.
- No drifter ID overlap across training, calibration, and test partitions.
- All 757 test case IDs agree between RQ1 and RQ3; maximum absolute 72-hour error difference: 6.395e-13 km.
- Five `Unclassified` cases occur only in training; all calibration and test cases belong to one of four regimes.
- The corrected integrity audit reported **42/42 checks passed**.

## Global pointwise conformal prediction

| Horizon | 90% radius (km) | Test coverage |
| --- | ---: | ---: |
| 24 h | 27.696 | 89.0% |
| 48 h | 67.304 | 91.8% |
| 72 h | 105.196 | 91.8% |

At 72 h, the drifter-cluster bootstrap 95% interval for global 90% coverage was approximately 89.5%–93.9%.

## 72-hour regime comparison

| Regime | Test cases | Global coverage | Mondrian coverage | Global radius (km) | Mondrian radius (km) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kuroshio Core | 139 | 79.1% | 89.9% | 105.196 | 141.227 |
| Shelf | 112 | 96.4% | 91.1% | 105.196 | 62.869 |
| Transition | 165 | 89.7% | 91.5% | 105.196 | 117.177 |
| Other Offshore | 341 | 96.5% | 93.0% | 105.196 | 93.331 |

Both Global and Mondrian case-level CP achieved 91.8% overall coverage. The case-weighted mean region area changed from 34,764.7 km² (Global) to 35,070.9 km² (Mondrian), an increase of 0.88%. The across-regime coverage range narrowed from approximately 17.4 to 3.1 percentage points.

## Paired test-drifter cluster bootstrap

Differences below are Mondrian minus Global, with 2,000 bootstrap replicates:

| Metric | Estimate | 95% percentile interval |
| --- | ---: | ---: |
| Overall coverage difference | 0.00 percentage points | [-1.66, +1.62] |
| Mean region area difference | +306.18 km² | [-2,299.24, +3,134.41] |
| Across-regime coverage range difference | -14.31 percentage points | [-19.73, -5.10] |
| Kuroshio Core coverage difference | +10.79 percentage points | [+6.21, +16.50] |

The bootstrap estimates uncertainty from held-out test drifters conditional on the fitted model and fixed calibration radii. It does not account for calibration-set sampling variability or exploratory method selection.

## Drifter-max sensitivity analysis

| Regime | Drifter-max radius (km) | Case-level coverage | Drifter-level all-case coverage |
| --- | ---: | ---: | ---: |
| Kuroshio Core | 195.973 | 98.6% | 95.6% (43/45) |
| Shelf | 99.425 | 94.6% | 82.1% (23/28) |
| Transition | 150.880 | 97.0% | 88.4% (38/43) |
| Other Offshore | 132.315 | 99.4% | 95.1% (39/41) |

Overall case-level coverage was 98.0%, with a case-weighted mean area of 67,109.8 km². The 98.0% statistic is **not** the fraction of drifters with all cases covered. Regime-specific drifter-level all-case coverage was below 90% in Shelf and Transition. Drifter counts overlap across regimes and must not be summed as independent unique drifters.

## Whole-trajectory coverage

At a nominal 90% level, requiring all 72 hourly positions to lie within individually calibrated pointwise regions covered 74.1% of test trajectories. A separately calibrated simultaneous trajectory band covered 93.4%, with a 72-hour radius of 166.751 km.

## Statistical limitations

1. Repeated forecast cases from a single drifter are dependent. Case-level split conformal finite-sample validity requires exchangeability assumptions that are not automatically satisfied.
2. Whole-drifter exchangeability and comparable within-drifter case sampling are required for a drifter-max guarantee; neither is established for these observational data.
3. Drifter-max results are sensitive to the number of cases per drifter and may substantially enlarge prediction regions.
4. Regime-aware methods were developed after inspecting global test-set undercoverage. Comparisons are exploratory rather than a fully untouched confirmatory test.
5. The paired bootstrap treats calibration thresholds as fixed; its intervals do not represent full pipeline uncertainty.
6. Circular regions do not capture anisotropic directional forecast errors, and results need not transfer to other oceans, periods, or drifting objects.
