
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

# ============================================================
# 1. Paths and settings
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CLIM_PATH = (
    ROOT / "RQ2" / "data" /
    "glorys_climatology_1994_2022.nc"
)

FIG_DIR = ROOT / "RQ2" / "figures"
DATA_DIR = ROOT / "RQ2" / "data"

FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.5
MIN_COMPONENT_CELLS = 5

# Eight-neighbour connectivity
STRUCTURE = np.ones((3, 3), dtype=int)

# ============================================================
# 2. Load GLORYS climatology
# ============================================================

print("=== RQ2 STEP 12C: IDENTIFY CURRENT COMPONENTS ===")

with xr.open_dataset(CLIM_PATH) as ds:
    clim = ds.load()

u = (
    clim["u_mean"]
    .squeeze(drop=True)
    .transpose("latitude", "longitude")
    .values
)

v = (
    clim["v_mean"]
    .squeeze(drop=True)
    .transpose("latitude", "longitude")
    .values
)

lon = clim["longitude"].values
lat = clim["latitude"].values

lon_grid, lat_grid = np.meshgrid(lon, lat)

speed = np.hypot(u, v)

valid = np.isfinite(speed)
high_current = valid & (speed >= THRESHOLD)

print("Threshold:", THRESHOLD, "m/s")
print("High-current cells:", int(high_current.sum()))

# ============================================================
# 3. Connected-component analysis
# ============================================================

labels, n_components = ndimage.label(
    high_current,
    structure=STRUCTURE
)

print("Number of components:", n_components)

rows = []

for component_id in range(1, n_components + 1):

    mask = labels == component_id
    n_cells = int(mask.sum())

    if n_cells == 0:
        continue

    component_u = u[mask]
    component_v = v[mask]

    mean_u = float(np.mean(component_u))
    mean_v = float(np.mean(component_v))

    mean_direction_deg = (
        np.degrees(np.arctan2(mean_v, mean_u)) + 360
    ) % 360

    rows.append({
        "component_id": component_id,
        "n_cells": n_cells,
        "lon_min": float(lon_grid[mask].min()),
        "lon_max": float(lon_grid[mask].max()),
        "lat_min": float(lat_grid[mask].min()),
        "lat_max": float(lat_grid[mask].max()),
        "centroid_lon": float(lon_grid[mask].mean()),
        "centroid_lat": float(lat_grid[mask].mean()),
        "mean_speed_m_s": float(speed[mask].mean()),
        "max_speed_m_s": float(speed[mask].max()),
        "mean_u_m_s": mean_u,
        "mean_v_m_s": mean_v,
        "mean_direction_deg": mean_direction_deg,
    })

summary = pd.DataFrame(rows)

summary = summary.sort_values(
    "n_cells", ascending=False
).reset_index(drop=True)

# Rank components by size, but retain original component IDs.
summary.insert(
    0,
    "size_rank",
    np.arange(1, len(summary) + 1)
)

summary_path = DATA_DIR / "current_components_0p5.csv"
summary.to_csv(summary_path, index=False)

print("\n=== ALL COMPONENTS ===")
print(
    summary[
        [
            "size_rank",
            "component_id",
            "n_cells",
            "lon_min",
            "lon_max",
            "lat_min",
            "lat_max",
            "mean_speed_m_s",
            "mean_direction_deg",
        ]
    ].round(3).to_string(index=False)
)

# ============================================================
# 4. Save grid labels for reproducibility
# ============================================================

label_ds = xr.Dataset(
    data_vars={
        "component_id": (
            ("latitude", "longitude"),
            labels.astype(np.int16)
        ),
        "high_current_mask": (
            ("latitude", "longitude"),
            high_current.astype(np.int8)
        ),
    },
    coords={
        "latitude": lat,
        "longitude": lon,
    },
    attrs={
        "threshold_m_s": THRESHOLD,
        "connectivity": "8-neighbour",
        "description": (
            "Connected high-current regions in "
            "1994-2022 GLORYS mean vector velocity"
        ),
    }
)

label_path = DATA_DIR / "current_components_0p5.nc"
label_ds.to_netcdf(label_path)

# ============================================================
# 5. Plot numbered components
# ============================================================

fig, ax = plt.subplots(figsize=(12, 9))

background = ax.pcolormesh(
    lon,
    lat,
    speed,
    cmap="Greys",
    vmin=0,
    vmax=1.1,
    shading="auto"
)

# Display each sufficiently large component in a different color
plot_labels = np.ma.masked_where(
    (labels == 0),
    labels
)

ax.pcolormesh(
    lon,
    lat,
    plot_labels,
    cmap="tab20",
    shading="auto",
    alpha=0.75,
    vmin=0,
    vmax=max(n_components, 1)
)

# Annotate component IDs at centroids
for _, row in summary.iterrows():

    if row["n_cells"] < MIN_COMPONENT_CELLS:
        continue

    ax.text(
        row["centroid_lon"],
        row["centroid_lat"],
        str(int(row["component_id"])),
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color="black",
        bbox={
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.9,
            "boxstyle": "round,pad=0.25",
        }
    )

ax.set_xlim(124, 136)
ax.set_ylim(24, 36)

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

ax.set_title(
    "GLORYS High-Current Components (1994–2022)\n"
    "Threshold >= 0.5 m/s, Eight-Neighbour Connectivity"
)

ax.grid(alpha=0.2)

cbar = fig.colorbar(
    background,
    ax=ax,
    shrink=0.8,
    pad=0.02
)

cbar.set_label("Climatological vector speed (m/s)")

plt.tight_layout()

fig_path = FIG_DIR / "current_components_0p5.png"

plt.savefig(
    fig_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# 6. Finish
# ============================================================

print("\n=== SAVED FILES ===")
print(summary_path)
print(label_path)
print(fig_path)

print("\n=== STEP 12C COMPLETE ===")
