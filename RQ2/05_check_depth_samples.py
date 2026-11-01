
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

file_path = (
    ROOT / "RQ2" / "data" /
    "forecast_case_bathymetry.csv"
)

df = pd.read_csv(file_path)

# Only test data
test = df[df["partition"] == "test"].copy()

print("=== TEST SAMPLE DISTRIBUTION ===")

summary = (
    test.groupby("depth_band", observed=True)
    .agg(
        n_cases=("forecast_case_id", "count"),
        n_drifters=("ID", "nunique"),
        median_depth_m=("depth_m", "median")
    )
    .reset_index()
)

summary["cases_per_drifter"] = (
    summary["n_cases"] / summary["n_drifters"]
).round(2)

print(summary.to_string(index=False))

print("\n=== DRIFTERS CROSSING DEPTH BANDS ===")

crossing = (
    test.groupby("ID")["depth_band"]
    .nunique()
)

print(
    "Drifters represented in multiple depth bands:",
    (crossing > 1).sum()
)

print(
    "Total test drifters:",
    test["ID"].nunique()
)
