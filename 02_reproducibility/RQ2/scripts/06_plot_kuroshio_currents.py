
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GLORYS_DIR = ROOT / "data" / "external" / "glorys"
OUTPUT_DIR = ROOT / "RQ2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find downloaded GLORYS file
files = sorted(GLORYS_DIR.rglob("*.nc"))
if not files:
    raise FileNotFoundError("No GLORYS .nc file found")

file_path = files[0]
print("Using:", file_path.name)

# Open data (try installed engines)
try:
    ds = xr.open_dataset(file_path, engine="h5netcdf")
except (OSError, ValueError):
    ds = xr.open_dataset(file_path, engine="scipy")

print("Variables:", list(ds.data_vars))
print("Dimensions:", dict(ds.sizes))

# Select first time and shallowest depth
u = ds["uo"].isel(time=0, depth=0)
v = ds["vo"].isel(time=0, depth=0)

# Surface current speed (m/s)
speed = np.sqrt(u**2 + v**2)

fig, ax = plt.subplots(figsize=(10, 8))

mesh = ax.pcolormesh(
    u["longitude"],
    u["latitude"],
    speed,
    shading="auto",
    cmap="viridis",
    vmin=0
)

# Reduce arrow density
step = 4

lon = u["longitude"].values[::step]
lat = u["latitude"].values[::step]

uu = u.values[::step, ::step]
vv = v.values[::step, ::step]

ax.quiver(
    lon,
    lat,
    uu,
    vv,
    color="white",
    scale=12,
    width=0.002
)

ax.set_xlim(124, 136)
ax.set_ylim(24, 36)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("GLORYS Surface Currents - East China Sea")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Surface current speed (m/s)")

plt.tight_layout()

output_path = OUTPUT_DIR / "glorys_surface_currents.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Saved:", output_path)
plt.show()

ds.close()
