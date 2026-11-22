
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm

# 1. Set paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data" / "processed"
GEBCO_DIR = ROOT / "Data" / "external" / "gebco"
OUTPUT_DIR = ROOT / "RQ2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Find GEBCO NetCDF file
nc_files = sorted(GEBCO_DIR.rglob("*.nc"))

if not nc_files:
    raise FileNotFoundError(
        f"No .nc file found in {GEBCO_DIR}"
    )

gebco_path = nc_files[0]
print("Using GEBCO file:", gebco_path.name)

# 3. Read GEBCO bathymetry
ds = xr.open_dataset(gebco_path, engine="scipy")
print("Available variables:", list(ds.data_vars))

if "elevation" not in ds:
    raise KeyError(
        "Expected 'elevation' variable. "
        "Check the GEBCO NetCDF structure."
    )

elevation = ds["elevation"].sel(
    lon=slice(124, 136),
    lat=slice(24, 36)
)

# Convert negative elevation to positive water depth
depth = xr.where(elevation < 0, -elevation, np.nan)

print("Bathymetry loaded.")
print("Grid shape:", depth.shape)

# 4. Load test forecast cases
origins = pd.read_parquet(
    DATA_DIR / "forecast_origins.parquet"
)
test = origins[origins["partition"] == "test"].copy()

print("Test cases:", len(test))

# 5. Plot bathymetry
fig, ax = plt.subplots(figsize=(10, 9))

depth_levels = [
    0, 50, 100, 200, 500,
    1000, 2000, 4000, 8000
]

cmap = plt.get_cmap("Blues", len(depth_levels) - 1)
norm = BoundaryNorm(depth_levels, cmap.N)

mesh = ax.pcolormesh(
    depth["lon"],
    depth["lat"],
    depth,
    cmap=cmap,
    norm=norm,
    shading="auto",
    rasterized=True
)

# Continental shelf boundary (200 m)
ax.contour(
    depth["lon"],
    depth["lat"],
    depth,
    levels=[200],
    colors="red",
    linewidths=1.2
)

# Overlay test forecast origins
ax.scatter(
    test["origin_longitude"],
    test["origin_latitude"],
    s=9,
    color="orange",
    edgecolors="black",
    linewidths=0.15,
    alpha=0.75,
    label=f"Test origins (n={len(test)})"
)

ax.set_xlim(124, 136)
ax.set_ylim(24, 36)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(
    "GEBCO 2026 Bathymetry and Test Forecast Origins"
)

ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.2)
ax.legend(loc="upper right")

cbar = fig.colorbar(mesh, ax=ax, shrink=0.8)
cbar.set_label("Water depth (m)")

plt.tight_layout()

output_path = OUTPUT_DIR / "gebco_test_origins.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Saved figure:", output_path)
plt.show()

ds.close()
