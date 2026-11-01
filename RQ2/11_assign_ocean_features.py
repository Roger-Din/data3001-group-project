
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

# ============================================================
# 1. Project paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

CLIM_PATH = (
    ROOT / "RQ2" / "data" /
    "glorys_climatology_1994_2022.nc"
)

BATHY_PATH = (
    ROOT / "RQ2" / "data" /
    "forecast_case_bathymetry.csv"
)

OUTPUT_PATH = (
    ROOT / "RQ2" / "data" /
    "forecast_case_ocean_features.csv"
)

# Maximum distance to the nearest valid ocean grid point
MAX_DISTANCE_KM = 15.0

# ============================================================
# 2. Load datasets
# ============================================================

print("=== RQ2 STEP 11: ASSIGN OCEAN FEATURES ===")

cases = pd.read_csv(BATHY_PATH)

with xr.open_dataset(CLIM_PATH) as ds:
    clim = ds.load()

print("\nCases loaded:", len(cases))
print("GLORYS variables:", list(clim.data_vars))

required = [
    "forecast_case_id",
    "origin_latitude",
    "origin_longitude",
    "depth_m",
]

missing = [c for c in required if c not in cases.columns]
if missing:
    raise ValueError(f"Missing case columns: {missing}")

if cases["forecast_case_id"].duplicated().any():
    raise ValueError("Duplicate forecast_case_id detected")

# ============================================================
# 3. Identify climatology variables
# ============================================================

def find_variable(dataset, candidates):
    for name in candidates:
        if name in dataset:
            return name
    raise ValueError(
        f"None of these variables found: {candidates}. "
        f"Available: {list(dataset.data_vars)}"
    )

u_name = find_variable(clim, ["u_mean", "uo_mean", "uo"])
v_name = find_variable(clim, ["v_mean", "vo_mean", "vo"])

u = clim[u_name].squeeze(drop=True)
v = clim[v_name].squeeze(drop=True)

# The climatology should have latitude and longitude only.
if set(u.dims) != {"latitude", "longitude"}:
    raise ValueError(f"Unexpected u dimensions: {u.dims}")

if set(v.dims) != {"latitude", "longitude"}:
    raise ValueError(f"Unexpected v dimensions: {v.dims}")

u = u.transpose("latitude", "longitude")
v = v.transpose("latitude", "longitude")

lat = clim["latitude"].values
lon = clim["longitude"].values

lon_grid, lat_grid = np.meshgrid(lon, lat)

u_grid = u.values
v_grid = v.values

valid = np.isfinite(u_grid) & np.isfinite(v_grid)

print("Valid ocean grid cells:", valid.sum())

# ============================================================
# 4. Build nearest-ocean-grid spatial index
#    Use 3D spherical coordinates for distance matching.
# ============================================================

EARTH_RADIUS_KM = 6371.0088

def latlon_to_xyz(latitude, longitude):
    lat_rad = np.deg2rad(np.asarray(latitude))
    lon_rad = np.deg2rad(np.asarray(longitude))

    return np.column_stack([
        np.cos(lat_rad) * np.cos(lon_rad),
        np.cos(lat_rad) * np.sin(lon_rad),
        np.sin(lat_rad),
    ])

valid_lat = lat_grid[valid]
valid_lon = lon_grid[valid]
valid_u = u_grid[valid]
valid_v = v_grid[valid]

tree = cKDTree(latlon_to_xyz(valid_lat, valid_lon))

case_xyz = latlon_to_xyz(
    cases["origin_latitude"].to_numpy(),
    cases["origin_longitude"].to_numpy()
)

chord_distance, indices = tree.query(case_xyz, k=1)

# Convert unit-sphere chord distance to great-circle km
distance_km = (
    2 * EARTH_RADIUS_KM *
    np.arcsin(np.clip(chord_distance / 2, 0, 1))
)

# ============================================================
# 5. Assign ocean features
# ============================================================

cases["glorys_match_distance_km"] = distance_km
cases["u_clim"] = valid_u[indices]
cases["v_clim"] = valid_v[indices]

cases["current_speed_clim"] = np.hypot(
    cases["u_clim"],
    cases["v_clim"]
)

too_far = distance_km > MAX_DISTANCE_KM

cases.loc[
    too_far,
    ["u_clim", "v_clim", "current_speed_clim"]
] = np.nan

cases["glorys_match_valid"] = ~too_far

# ============================================================
# 6. Quality checks
# ============================================================

print("\n=== MATCHING QUALITY ===")
print("Total cases:", len(cases))
print("Valid matches:", int(cases["glorys_match_valid"].sum()))
print("Unmatched cases:", int(too_far.sum()))

print("\nMatch distance (km):")
print(cases["glorys_match_distance_km"].describe())

print("\n=== CURRENT SPEED DISTRIBUTION (m/s) ===")
print(cases["current_speed_clim"].describe())

if "partition" in cases.columns:
    print("\n=== MATCHES BY PARTITION ===")
    print(
        cases.groupby("partition")["glorys_match_valid"]
        .agg(["count", "sum"])
    )

    print("\n=== TEST CURRENT SPEED ===")
    test = cases[cases["partition"] == "test"]
    print(test["current_speed_clim"].describe())

print("\n=== DEPTH AND CURRENT SPEED ===")

if "depth_band" in cases.columns:
    print(
        cases.groupby("depth_band", observed=True)
        ["current_speed_clim"]
        .agg(["count", "median", "mean"])
    )

# ============================================================
# 7. Save
# ============================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

cases.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
print("=== STEP 11 COMPLETE ===")
