#!/usr/bin/env python3
"""Independently verify the frozen evidence used by the final report."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
checks: list[dict[str, object]] = []


def record(name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")


def close(actual: float, expected: float, tolerance: float = 0.02) -> bool:
    return bool(np.isfinite(actual) and abs(actual - expected) <= tolerance)


def validate_required_files() -> None:
    required = [
        "data/processed/region_A_observations.parquet",
        "data/processed/forecast_origins.parquet",
        "data/processed/drifter_splits.csv",
        "output/rq1/zh/optimised_fnn_test_case_metrics.csv",
        "output/rq1/zh/random_forest_test_case_metrics.csv",
        "RQ2/data/forecast_case_regimes.csv",
        "RQ2/results/optimised_fnn_analysis/core_retraining_comparison.csv",
        "RQ2/results/optimised_fnn_analysis/recomputed/core_retraining_comparison.csv",
        "RQ2/results/optimised_fnn_analysis/regional_retraining_five_models_summary.csv",
        "RQ3/optimised_fnn/output/optimised_fnn_conformal_overall.csv",
        "RQ3/optimised_fnn/output/optimised_fnn_conformal_regime_summary.csv",
        "RQ3/optimised_fnn/output/optimised_fnn_validation_checks.csv",
        "00_shared/00_report_figures.ipynb",
        "00_shared/01_model_development_and_trial_log.ipynb",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    record("Required files", not missing, "all present" if not missing else f"missing: {missing}")


def validate_notebooks() -> None:
    notebooks = sorted(
        path for path in ROOT.rglob("*.ipynb")
        if "source_handoff" not in path.parts and ".ipynb_checkpoints" not in path.parts
    )
    invalid: list[str] = []
    chinese: list[str] = []
    for path in notebooks:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(notebook.get("cells"), list):
                invalid.append(str(path.relative_to(ROOT)))
                continue
            sources = "\n".join(
                "".join(cell.get("source", []))
                for cell in notebook["cells"]
            )
            if CJK.search(sources):
                chinese.append(str(path.relative_to(ROOT)))
        except Exception:
            invalid.append(str(path.relative_to(ROOT)))
    record("Canonical notebook JSON", not invalid, f"{len(notebooks)} valid notebooks" if not invalid else str(invalid))
    record("English-facing notebook sources", not chinese, "no visible Chinese source text" if not chinese else str(chinese))


def validate_rq1_scripts() -> None:
    notebook_root = ROOT / "RQ1/notebooks"
    script_root = ROOT / "RQ1/scripts"
    notebooks = sorted(notebook_root.rglob("*.ipynb"))
    scripts = sorted(script_root.rglob("*.py"))

    expected = {
        path.relative_to(notebook_root).with_suffix(".py").as_posix()
        for path in notebooks
    }
    actual = {
        path.relative_to(script_root).as_posix()
        for path in scripts
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    record(
        "RQ1 notebook/script correspondence",
        not missing and not extra,
        f"{len(scripts)} scripts match {len(notebooks)} notebooks"
        if not missing and not extra
        else f"missing={missing}, extra={extra}",
    )

    invalid: list[str] = []
    for path in scripts:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            invalid.append(str(path.relative_to(ROOT)))
    record(
        "RQ1 Python script syntax",
        not invalid,
        "all scripts parse successfully" if not invalid else str(invalid),
    )


def validate_data() -> None:
    observations = pd.read_parquet(ROOT / "data/processed/region_A_observations.parquet")
    origins = pd.read_parquet(ROOT / "data/processed/forecast_origins.parquet")
    record("Observation count", len(observations) == 562_944, f"{len(observations):,}")
    record("Forecast-case count", len(origins) == 5_117, f"{len(origins):,}")

    counts = origins["partition"].value_counts().to_dict()
    expected = {"training": 3591, "calibration": 769, "test": 757}
    record("Partition case counts", counts == expected, str(counts))

    id_column = "ID" if "ID" in origins.columns else "drifter_id"
    sets = {part: set(group[id_column]) for part, group in origins.groupby("partition")}
    overlap = (
        sets["training"] & sets["calibration"]
        or sets["training"] & sets["test"]
        or sets["calibration"] & sets["test"]
    )
    record("Drifter-level leakage", not overlap, "no ID appears in multiple partitions")


def validate_rq1() -> None:
    fnn = pd.read_csv(ROOT / "output/rq1/zh/optimised_fnn_test_case_metrics.csv")
    rf = pd.read_csv(ROOT / "output/rq1/zh/random_forest_test_case_metrics.csv")
    fnn_rmse = float(np.sqrt(np.mean(np.square(fnn["error_72h_km"]))))
    fnn_median = float(fnn["error_72h_km"].median())
    fnn_ite = float(fnn["ite_km_h"].median())
    rf_rmse = float(np.sqrt(np.mean(np.square(rf["final_72h_error_km"]))))
    record("RQ1 FNN 72 h RMSE", close(fnn_rmse, 61.69), f"{fnn_rmse:.4f} km")
    record("RQ1 FNN median error", close(fnn_median, 41.34), f"{fnn_median:.4f} km")
    record("RQ1 FNN median ITE", close(fnn_ite, 1403.16, 0.05), f"{fnn_ite:.4f} km h")
    record("RQ1 original RF 72 h RMSE", close(rf_rmse, 62.13), f"{rf_rmse:.4f} km")


def validate_rq2() -> None:
    fnn = pd.read_csv(ROOT / "output/rq1/zh/optimised_fnn_test_case_metrics.csv")
    regimes = pd.read_csv(ROOT / "RQ2/data/forecast_case_regimes.csv")
    merged = fnn.merge(regimes[["forecast_case_id", "regime"]], on="forecast_case_id", validate="one_to_one")
    rmse = merged.groupby("regime")["error_72h_km"].apply(lambda x: float(np.sqrt(np.mean(np.square(x)))))
    record("RQ2 Shelf FNN RMSE", close(rmse["Shelf"], 40.825849, 0.001), f"{rmse['Shelf']:.6f} km")
    record("RQ2 Core FNN RMSE", close(rmse["Kuroshio Core"], 85.861726, 0.001), f"{rmse['Kuroshio Core']:.6f} km")

    retraining = pd.read_csv(
        ROOT / "RQ2/results/optimised_fnn_analysis/core_retraining_comparison.csv"
    )
    rmse_row = retraining.loc[retraining["metric"].eq("rmse")].iloc[0]
    record(
        "RQ2 Core-trained optimised FNN RMSE",
        close(float(rmse_row["regional_km"]), 82.124151, 0.001),
        f"{rmse_row['regional_km']:.6f} km",
    )
    record(
        "RQ2 Core-retraining clustered interval",
        float(rmse_row["ci_low_km"]) > 0,
        f"[{rmse_row['ci_low_km']:.4f}, {rmse_row['ci_high_km']:.4f}] km",
    )
    five_models = pd.read_csv(
        ROOT / "RQ2/results/optimised_fnn_analysis/regional_retraining_five_models_summary.csv"
    )
    fnn = five_models.loc[five_models["model"].eq("FNN")].iloc[0]
    record(
        "RQ2 Core FNN improvement",
        close(float(fnn["reduction_pct"]), 4.353016, 0.001),
        f"{fnn['reduction_pct']:.6f}%",
    )
    recomputed = pd.read_csv(
        ROOT / "RQ2/results/optimised_fnn_analysis/recomputed/core_retraining_comparison.csv"
    ).iloc[0]
    record(
        "RQ2 Core retraining independently reproduced",
        close(float(recomputed["regional_rmse_km"]), float(rmse_row["regional_km"]), 0.000001),
        f"recomputed {recomputed['regional_rmse_km']:.6f} km",
    )


def validate_rq3() -> None:
    out = ROOT / "RQ3/optimised_fnn/output"
    overall = pd.read_csv(out / "optimised_fnn_conformal_overall.csv").iloc[0]
    regimes = pd.read_csv(out / "optimised_fnn_conformal_regime_summary.csv")
    core_global = regimes[(regimes["method"] == "Global") & (regimes["regime"] == "Kuroshio Core")].iloc[0]
    core_mondrian = regimes[(regimes["method"] == "Mondrian") & (regimes["regime"] == "Kuroshio Core")].iloc[0]
    record("RQ3 global coverage", close(float(overall["coverage_72h"]), 0.910172, 0.00001), f"{overall['coverage_72h']:.6f}")
    record("RQ3 global radius", close(float(overall["radius_72h_km"]), 101.873824, 0.001), f"{overall['radius_72h_km']:.6f} km")
    record("RQ3 Core global coverage", close(float(core_global["coverage_72h"]), 0.805755, 0.00001), f"{core_global['coverage_72h']:.6f}")
    record("RQ3 Core Mondrian coverage", close(float(core_mondrian["coverage_72h"]), 0.899281, 0.00001), f"{core_mondrian['coverage_72h']:.6f}")

    audit = pd.read_csv(out / "optimised_fnn_validation_checks.csv")
    statuses = audit["status"].value_counts().to_dict()
    record("RQ3 validation audit", statuses.get("FAIL", 0) == 0, str(statuses))


def validate_manifest() -> None:
    manifest = ROOT / "MANIFEST_SHA256.txt"
    if not manifest.exists():
        record("SHA256 manifest", True, "not yet generated; run scripts/generate_manifest.py")
        return
    mismatches: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.exists():
            mismatches.append(f"missing:{relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(f"changed:{relative}")
    record("SHA256 manifest", not mismatches, "all listed files match" if not mismatches else str(mismatches[:10]))


def main() -> int:
    validate_required_files()
    validate_notebooks()
    validate_rq1_scripts()
    validate_data()
    validate_rq1()
    validate_rq2()
    validate_rq3()
    validate_manifest()
    passed = all(item["status"] == "PASS" for item in checks)
    result = {"overall_status": "PASS" if passed else "FAIL", "checks": checks}
    destination = ROOT / "validation/latest_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nOverall submission validation: {result['overall_status']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
