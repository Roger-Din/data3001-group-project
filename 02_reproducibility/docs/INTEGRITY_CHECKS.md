# Integrity checks

`scripts/validate_submission.py` checks:

1. required files and canonical result tables exist;
2. all canonical notebooks are valid JSON and contain no visible Chinese text;
3. processed observation and forecast-case counts match the report;
4. train/calibration/test counts match 3,591/769/757;
5. drifter IDs do not cross partitions;
6. RQ1 FNN and Random Forest headline metrics recompute from case-level files;
7. RQ2 Shelf and Kuroshio Core RMSE recompute by merging the frozen FNN errors with regime labels;
8. final RQ3 global and regime-specific coverage values match the canonical output;
9. RQ3 validation contains no failed checks;
10. file hashes match `MANIFEST_SHA256.txt` when the manifest is present.

The machine-readable result is written to `validation/latest_validation.json`. That transient file is ignored by Git.
