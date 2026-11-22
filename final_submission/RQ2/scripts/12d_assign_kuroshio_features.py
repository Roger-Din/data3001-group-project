
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]

CASE_PATH = (
    ROOT / "RQ2" / "data" /
    "forecast_case_ocean_features.csv"
)

COMP_PATH = (
    ROOT / "RQ2" / "data" /
    "current_components_0p5.nc"
)

OUTPUT_PATH = (
    ROOT / "RQ2" / "data" /
    "forecast_case_kuroshio_features.csv"
)

EARTH_RADIUS_KM = 6371.0088
MAX_MATCH_DISTANCE_KM = 15.0

# Geographically verified candidate Kuroshio components
KUROSHIO_COMPONENTS = {1, 2}

def to_xyz(latitude, longitude):
    lat = np.deg2rad(np.asarray(latitude))
    lon = np.deg2rad(np.asarray(longitude))

    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])

def chord_to_km(chord):
    return (
        2 * EARTH_RADIUS_KM *
        np.arcsin(np.clip(chord / 2, 0, 1))
    )

print("=== RQ2 STEP 12D: ASSIGN KUROSHIO FEATURES ===")

cases = pd.read_csv(CASE_PATH)

with xr.open_dataset(COMP_PATH) as ds:
    components = ds.load()

labels = components["component_id"].values

lat = components["latitude"].values
lon = components["longitude"].values

lon_grid, lat_grid = np.meshgrid(lon, lat)

# ------------------------------------------------------------
# 1. Match cases to their nearest GLORYS grid cell
# ------------------------------------------------------------

with xr.open_dataset(
    ROOT / "RQ2" / "data" /
    "glorys_climatology_1994_2022.nc"
) as clim_ds:
    u = clim_ds["u_mean"].squeeze().values
    v = clim_ds["v_mean"].squeeze().values

valid_grid = np.isfinite(u) & np.isfinite(v)

grid_xyz = to_xyz(
    lat_grid[valid_grid],
    lon_grid[valid_grid]
)

tree = cKDTree(grid_xyz)

case_xyz = to_xyz(
    cases["origin_latitude"].values,
    cases["origin_longitude"].values
)

chord, indices = tree.query(case_xyz)

match_distance = chord_to_km(chord)

grid_labels = labels[valid_grid]

cases["current_component_id"] = grid_labels[indices]
cases["component_match_distance_km"] = match_distance

invalid_match = (
    match_distance > MAX_MATCH_DISTANCE_KM
    ) | (~cases["glorys_match_valid"].astype(bool))

cases.loc[
    invalid_match, "current_component_id"
] = np.nan

# ------------------------------------------------------------
# 2. Define Kuroshio core
# ------------------------------------------------------------

cases["kuroshio_core"] = (
    cases["current_component_id"].isin(
        KUROSHIO_COMPONENTS
    )
)

cases["kuroshio_segment"] = "None"

cases.loc[
    cases["current_component_id"] == 1,
    "kuroshio_segment"
] = "East China Sea"

cases.loc[
    cases["current_component_id"] == 2,
    "kuroshio_segment"
] = "South Japan"

# ------------------------------------------------------------
# 3. Distance to nearest Kuroshio core grid cell
# ------------------------------------------------------------

core_mask = np.isin(
    labels,
    list(KUROSHIO_COMPONENTS)
)

core_xyz = to_xyz(
    lat_grid[core_mask],
    lon_grid[core_mask]
)

core_tree = cKDTree(core_xyz)

core_chord, _ = core_tree.query(case_xyz)

cases["distance_to_kuroshio_km"] = (
    chord_to_km(core_chord)
)

# ------------------------------------------------------------
# 4. Diagnostics
# ------------------------------------------------------------

print("\nTotal cases:", len(cases))

print("\n=== COMPONENT MATCHING ===")
print(
    cases["current_component_id"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\n=== KUROSHIO CORE BY PARTITION ===")
print(
    pd.crosstab(
        cases["partition"],
        cases["kuroshio_core"]
    )
)

print("\n=== TEST SEGMENTS ===")
test = cases[cases["partition"] == "test"]

print(test["kuroshio_segment"].value_counts())

print("\n=== TEST DISTANCE TO KUROSHIO ===")
print(test["distance_to_kuroshio_km"].describe())

print("\n=== TEST DISTANCE QUANTILES ===")
print(
    test["distance_to_kuroshio_km"].quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9]
    )
)

print("\n=== TEST CORE DRIFTERS ===")
print(
    test.groupby("kuroshio_segment")["ID"].nunique()
)

# ------------------------------------------------------------
# 5. Save
# ------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

cases.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
print("=== STEP 12D COMPLETE ===")
