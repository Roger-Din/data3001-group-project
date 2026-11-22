# RQ2 hand-off comparison

The late `RQ2(1).zip` hand-off was compared file-by-file with the existing final-submission package. Most numbered scripts and core data tables were the same as the versions already present. The genuinely new material was the `optimised_fnn_analysis` result family and its figures.

## New evidence added

### Final optimised-FNN Core retraining

On the same 139 Kuroshio-Core test cases from 45 drifters:

| Model version | 72 h RMSE |
|---|---:|
| Frozen global optimised FNN | 85.8617 km |
| Core-trained optimised FNN | 82.1242 km |

The absolute reduction is 3.7376 km, or 4.353%. The drifter-clustered 95% interval for the RMSE reduction is 0.5969 to 6.8504 km, so this particular Core-only refit showed a supported improvement.

The median error moved slightly in the opposite direction (61.22 to 62.17 km), with an interval spanning zero. This is another example of why the conclusion depends on the loss function: regional retraining reduced tail-sensitive RMSE but did not improve the typical-case median.

### Five-model regional refitting

The hand-off also contains a consistent Core-only comparison for optimised KNN, optimised FNN, original Random Forest, Ridge and pooled VAR(1). All regional RMSE values are retained in `RQ2/results/optimised_fnn_analysis/regional_retraining_five_models_summary.csv`. These are exploratory extensions, not replacements for the globally frozen RQ1 model.

### Sensitivity analysis

The Core-versus-Shelf gap remains large across current-speed thresholds of 0.4, 0.5 and 0.6 m/s and transition buffers of 25, 50 and 75 km. At the selected 0.5 m/s threshold and 50 km buffer, Core and Shelf RMSE are 85.86 and 40.83 km respectively. The buffer changes the Transition/Other-Offshore boundary but does not change which cases are Core or Shelf at a fixed current threshold.

## Why an older result looked different

`RQ2/data/kuroshio_all_models/kuroshio_global_vs_regional_all_models.csv` reports 82.53 km for a regional FNN and compares it with an 87.48 km global FNN. That file belongs to the earlier original-model extension pipeline. The new hand-off applies the final optimised-FNN specification and compares it with the frozen optimised global FNN, producing the report's 82.12 km and 4.35% figures.

Both files are preserved and labelled by experiment version. The final report value is sourced from `RQ2/results/optimised_fnn_analysis/`.

## Reproducibility addition

The zip contained result tables but no standalone generator for the optimised-FNN Core refit. `RQ2/scripts/22_reproduce_core_optimised_fnn.py` was therefore added. It uses the frozen 128–64 architecture, L2=0.001, 129 epochs and the three RQ1 ensemble seeds, then performs a drifter-clustered bootstrap.
