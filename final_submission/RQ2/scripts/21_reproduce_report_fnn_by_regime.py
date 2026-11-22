#!/usr/bin/env python3
"""Reproduce the report's regional metrics from frozen optimised-FNN errors."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ERRORS = ROOT / "output/rq1/zh/optimised_fnn_test_case_metrics.csv"
REGIMES = ROOT / "RQ2/data/forecast_case_regimes.csv"
OUTPUT = ROOT / "RQ2/results/report_optimised_fnn_regime_summary.csv"

errors = pd.read_csv(ERRORS)
regimes = pd.read_csv(REGIMES, usecols=["forecast_case_id", "ID", "partition", "regime"])
merged = errors.merge(regimes, on="forecast_case_id", validate="one_to_one")
if len(merged) != 757 or not merged["partition"].eq("test").all():
    raise ValueError("Expected exactly 757 frozen test cases.")

summary = (
    merged.groupby("regime", observed=True)
    .agg(
        n_cases=("forecast_case_id", "size"),
        n_drifters=("ID", "nunique"),
        median_72h_error_km=("error_72h_km", "median"),
        median_ite_km_h=("ite_km_h", "median"),
    )
    .reset_index()
)
rmse = (
    merged.groupby("regime", observed=True)["error_72h_km"]
    .apply(lambda values: float(np.sqrt(np.mean(np.square(values)))))
    .rename("rmse_72h_km")
    .reset_index()
)
summary = summary.merge(rmse, on="regime", validate="one_to_one")
order = ["Shelf", "Kuroshio Core", "Transition", "Other Offshore"]
summary["regime"] = pd.Categorical(summary["regime"], order, ordered=True)
summary = summary.sort_values("regime").reset_index(drop=True)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT, index=False)
print(summary.to_string(index=False))
print(f"\nSaved {OUTPUT}")
