
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

CLIM_PATH = (
    ROOT / "RQ2" / "data" /
    "glorys_climatology_1994_2022.nc"
)

BATHY_DIR = ROOT / "Data" / "external" / "gebco"
ORIGINS_PATH = (
    ROOT / "Data" / "processed" /
    "forecast_origins.parquet"
)

OUTPUT_DIR = ROOT / "RQ2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

gebco_files = sorted(BATHY_DIR.glob("*.nc"))
if not gebco_files:
    raise FileNotFoundError("GEBCO file not found")

with xr.open_dataset(CLIM_PATH, engine="h5netcdf") as ds:
    clim = ds.load()

with xr.open_dataset(gebco_files[0], engine="scipy") as ds:
    bathy = ds["elevation"].sel(
        lon=slice(124, 136),
        lat=slice(24, 36)
    ).load()

origins = pd.read_parquet(ORIGINS_PATH)
test = origins[origins["partition"] == "test"]

fig, ax = plt.subplots(figsize=(11, 9))

# GLORYS long-term mean current speed
mesh = ax.pcolormesh(
    clim.longitude,
    clim.latitude,
    clim["mean_vector_speed"],
    shading="auto",
    cmap="viridis",
    vmin=0,
    vmax=1.1
)

# GEBCO 200m and 1000m isobaths
contours = ax.contour(
    bathy["lon"],
    bathy["lat"],
    bathy,
    levels=[-1000, -200],
    colors=["orange", "red"],
    linewidths=[1.2, 1.5]
)

ax.clabel(contours, inline=True, fontsize=8)

# Mean current directions
step = 6

ax.quiver(
    clim.longitude.values[::step],
    clim.latitude.values[::step],
    clim["u_mean"].values[::step, ::step],
    clim["v_mean"].values[::step, ::step],
    color="white",
    scale=12,
    width=0.002
)

# Test forecast origins
ax.scatter(
    test["origin_longitude"],
    test["origin_latitude"],
    s=9,
    c="magenta",
    alpha=0.65,
    label=f"Test origins (n={len(test)})",
    zorder=5
)

ax.set_xlim(124, 136)
ax.set_ylim(24, 36)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(
    "GLORYS Climatology + GEBCO Bathymetry + Test Origins"
)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Mean surface current speed (m/s)")

ax.legend(loc="upper left")
plt.tight_layout()

output_path = OUTPUT_DIR / "regime_overlay.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print("Saved:", output_path)

plt.show()
