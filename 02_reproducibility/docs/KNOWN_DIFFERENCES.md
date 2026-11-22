# Known differences and version notes

## Core-only regional FNN experiment

The main report's descriptive regional comparison is reproducible from the frozen optimised-FNN predictions: Shelf 40.83 km and Kuroshio Core 85.86 km at 72 hours.

The late RQ2 hand-off resolved the previously ambiguous Core-only value. The final optimised-FNN experiment gives 82.12 km after Core-only retraining, a 4.35% improvement over the 85.86 km frozen global optimised FNN. An older all-model extension gives 82.53 km because it belongs to an earlier FNN specification and uses a different global comparator (87.48 km). Both versions are retained; only the final optimised-FNN result is used for the report's 82.12 km statement. See `RQ2_ZIP_COMPARISON.md`.

## RQ3 versions

`RQ3/optimised_fnn/output/` is the canonical final RQ3 output used by the report. An earlier alternative pipeline is retained under `RQ3/results/exploratory_original_pipeline/` and must not be substituted for the canonical output.
