
from pathlib import Path
import pandas as pd

# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Load processed datasets
origins = pd.read_parquet(DATA_DIR / "forecast_origins.parquet")
splits = pd.read_csv(DATA_DIR / "drifter_splits.csv")
qc = pd.read_csv(DATA_DIR / "qc_summary.csv")

print("=== RQ2 DATA CHECK ===")

print("\nTotal forecast cases:", len(origins))
print("Unique drifters:", origins["ID"].nunique())

print("\nForecast cases by partition:")
print(origins["partition"].value_counts())

print("\nDrifters by partition:")
print(splits["partition"].value_counts())

print("\nAvailable columns:")
print(origins.columns.tolist())

# Check that no drifter belongs to multiple partitions
leakage = origins.groupby("ID")["partition"].nunique().max()
print("\nMaximum partitions per drifter:", leakage)

# Check origin coordinates
print("\nOrigin coordinate ranges:")
print(
    origins[
        ["origin_latitude", "origin_longitude"]
    ].describe()
)

assert len(origins) == 5117, "Unexpected forecast case count"
assert leakage == 1, "Drifter-level data leakage detected"

print("\nSUCCESS: RQ2 data loaded and basic checks passed.")
