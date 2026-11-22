
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
GEBCO_DIR = ROOT / "data" / "external" / "gebco"
OUTPUT_DIR = ROOT / "RQ2" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load forecast cases
origins = pd.read_parquet(
    DATA_DIR / "forecast_origins.parquet"
)

# Load GEBCO
nc_files = sorted(GEBCO_DIR.rglob("*.nc"))
if not nc_files:
    raise FileNotFoundError("GEBCO .nc file not found")

with xr.open_dataset(nc_files[0], engine="scipy") as ds:
    elevation = ds["elevation"]

    # Nearest-neighbor lookup at forecast origin
    lons = xr.DataArray(
        origins["origin_longitude"].to_numpy(),
        dims="case"
    )
    lats = xr.DataArray(
        origins["origin_latitude"].to_numpy(),
        dims="case"
    )

    matched = elevation.sel(
        lon=lons,
        lat=lats,
        method="nearest"
    ).to_numpy()

# Positive depth in meters
depth_m = np.where(matched < 0, -matched, np.nan)

result = origins[
    [
        "forecast_case_id",
        "ID",
        "partition",
        "origin_longitude",
        "origin_latitude"
    ]
].copy()

result["depth_m"] = depth_m

# Descriptive depth categories only
result["depth_band"] = pd.cut(
    result["depth_m"],
    bins=[0, 200, 1000, np.inf],
    labels=[
        "0-200m",
        "200-1000m",
        ">1000m"
    ],
    include_lowest=True
)

# Quality checks
print("Total cases:", len(result))
print("Missing/land depth:", result["depth_m"].isna().sum())

print("\nDepth distribution:")
print(result["depth_m"].describe())

print("\nDepth bands by partition:")
print(pd.crosstab(
    result["partition"],
    result["depth_band"],
    dropna=False
))

assert len(result) == len(origins)
assert result["forecast_case_id"].is_unique

# Save
output_path = OUTPUT_DIR / "forecast_case_bathymetry.csv"
result.to_csv(output_path, index=False)

print("\nSaved:", output_path)
