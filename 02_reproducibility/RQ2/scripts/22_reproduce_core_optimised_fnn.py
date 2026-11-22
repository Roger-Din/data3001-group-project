#!/usr/bin/env python3
"""Retrain the selected FNN on Kuroshio-Core cases and reproduce the RQ2 extension."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RQ2 = ROOT / "RQ2"
OUTPUT = RQ2 / "results" / "optimised_fnn_analysis" / "recomputed"
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


extension = load_module("rq2_extension", RQ2 / "scripts" / "18_regime_conditioned_forecasting.py")
models = load_module("rq2_models", RQ2 / "scripts" / "19_kuroshio_all_models.py")

data, targets, x, _, y, _ = extension.build_xy()
train = np.flatnonzero(data["partition"].eq("training") & data["regime"].eq("Kuroshio Core"))
test = np.flatnonzero(data["partition"].eq("test") & data["regime"].eq("Kuroshio Core"))
if (len(train), len(test)) != (646, 139):
    raise ValueError(f"Unexpected Core split: {len(train)} training and {len(test)} test cases")

# These are the frozen RQ1 selected settings, not a new RQ2 hyperparameter search.
predictions = []
for seed in (2026, 2037, 2048):
    model = models.train_fnn(
        x[train], y[train], hidden=(128, 64), l2=0.001, epochs=129, seed=seed
    )
    predictions.append(models.fnn_predict(model, x[test]))
regional_prediction = np.mean(predictions, axis=0)
_, _, regional = models.evaluate(
    "Regional optimised FNN", regional_prediction, data, targets, test, y
)
regional = regional.rename(columns={
    "final_72h_error_km": "regional_error_72h_km",
    "ite_km_h": "regional_ite_km_h",
})

global_errors = pd.read_csv(ROOT / "output/rq1/zh/optimised_fnn_test_case_metrics.csv")
global_errors = global_errors.rename(columns={
    "error_72h_km": "global_error_72h_km",
    "ite_km_h": "global_ite_km_h",
})
paired = global_errors.merge(
    regional[["forecast_case_id", "ID", "regional_error_72h_km", "regional_ite_km_h"]],
    on="forecast_case_id", validate="one_to_one"
)

def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values))))


rng = np.random.default_rng(3001)
drifters = paired["ID"].unique()
replicates = []
for _ in range(2000):
    sampled = rng.choice(drifters, size=len(drifters), replace=True)
    sample = pd.concat([paired[paired["ID"].eq(d)] for d in sampled], ignore_index=True)
    replicates.append(
        rmse(sample["global_error_72h_km"]) - rmse(sample["regional_error_72h_km"])
    )

global_rmse = rmse(paired["global_error_72h_km"])
regional_rmse = rmse(paired["regional_error_72h_km"])
summary = pd.DataFrame([{
    "global_rmse_km": global_rmse,
    "regional_rmse_km": regional_rmse,
    "global_minus_regional_km": global_rmse - regional_rmse,
    "reduction_pct": 100 * (1 - regional_rmse / global_rmse),
    "ci_low_km": float(np.quantile(replicates, 0.025)),
    "ci_high_km": float(np.quantile(replicates, 0.975)),
    "n_cases": len(paired),
    "n_drifters": paired["ID"].nunique(),
    "ensemble_seeds": "2026;2037;2048",
}])

paired.to_csv(OUTPUT / "core_test_global_vs_regional_optimised_fnn.csv", index=False)
summary.to_csv(OUTPUT / "core_retraining_comparison.csv", index=False)
print(summary.to_string(index=False))
print(f"\nSaved reproducible outputs to {OUTPUT}")
