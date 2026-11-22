#!/usr/bin/env python3
"""Single entry point for verifying or re-running the submitted analyses."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

STAGES = {
    "data": [
        [sys.executable, "scripts/validate_submission.py"],
    ],
    "rq1": [
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/core/01_velocity_persistence_baseline.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/core/02_spatial_mean_flow_baseline.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/core/03_multioutput_ridge_trajectory_model.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/core/04_fnn_trajectory_model.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/core/05_pooled_var1_velocity_model.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/core/06_weighted_knn_trajectory_regression.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/extensions/07_extra_trees_trajectory_model.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/extensions/08_random_forest_trajectory_model.ipynb"],
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "RQ1/notebooks/extensions/11_optimised_fnn.ipynb"],
    ],
    "rq2": [
        [sys.executable, f"RQ2/scripts/{name}"]
        for name in [
            "01_check_data.py", "13_define_ocean_regimes.py",
            "21_reproduce_report_fnn_by_regime.py",
            "22_reproduce_core_optimised_fnn.py",
        ]
    ],
    "rq3": [
        [sys.executable, "RQ3/scripts/RQ3_optimised_fnn_final.py"],
        [sys.executable, "RQ3/scripts/06_validation_and_paired_bootstrap.py"],
        [sys.executable, "RQ3/scripts/make_report_figures.py"],
    ],
}


def run(commands: list[list[str]], dry_run: bool) -> None:
    for command in commands:
        print("$", " ".join(command))
        if not dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["verify", "full"], default="verify")
    parser.add_argument("--stage", choices=["data", "rq1", "rq2", "rq3", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "verify":
        run([[sys.executable, "scripts/validate_submission.py"]], args.dry_run)
        return 0

    selected = list(STAGES) if args.stage == "all" else [args.stage]
    print("Full reproduction may take hours because RQ1 refits grouped-CV models and ensembles.")
    for stage in selected:
        print(f"\n== {stage.upper()} ==")
        run(STAGES[stage], args.dry_run)
    if not args.dry_run:
        run([[sys.executable, "scripts/generate_manifest.py"]], False)
        run([[sys.executable, "scripts/validate_submission.py"]], False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
