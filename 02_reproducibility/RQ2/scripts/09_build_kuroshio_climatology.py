
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GLORYS_DIR = ROOT / "data" / "external" / "glorys"
OUTPUT_DIR = ROOT / "RQ2" / "data"
FIGURE_DIR = ROOT / "RQ2" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Original 1994 file
FILE_1994 = (
    GLORYS_DIR /
    "cmems_mod_glo_phy_my_0.083deg_P1M-m_1792574303833.nc"
)

# Load each annual file explicitly.
# The 2020 Jan-Feb test file is intentionally excluded.
annual_fields = []
all_months = []

for year in range(1994, 2023):
    if year == 1994:
        path = FILE_1994
    else:
        path = GLORYS_DIR / f"glorys_surface_{year}.nc"

    if not path.exists():
        raise FileNotFoundError(f"Missing annual file: {path}")

    with xr.open_dataset(path, engine="h5netcdf") as ds:
        required = {"uo", "vo"}
        if not required.issubset(ds.data_vars):
            raise ValueError(f"Missing variables in {path.name}")

        times = pd.DatetimeIndex(ds.time.values)
        expected = pd.date_range(
            f"{year}-01-01", periods=12, freq="MS"
        )

        if len(times) != 12 or not times.equals(expected):
            raise ValueError(
                f"Incorrect monthly coverage for {year}: {times}"
            )

        all_months.extend(times.tolist())

        # Surface layer
        field = ds[["uo", "vo"]].isel(depth=0).load()

        # Normalize coordinates to the common study grid
        field = field.interp(
            longitude=np.arange(124.125, 136, 1/12),
            latitude=np.arange(24.125, 36, 1/12),
        )

        annual_fields.append(field)

    print(f"Validated {year}: 12 months")

# Check 348 unique months
if len(all_months) != 348:
    raise ValueError("Expected 348 monthly records")

if len(set(all_months)) != 348:
    raise ValueError("Duplicate months detected")

print("\nAll 348 months validated.")

# Concatenate the monthly data
combined = xr.concat(annual_fields, dim="time")

# Calculate long-term mean velocity components
u_mean = combined["uo"].mean(dim="time", skipna=True)
v_mean = combined["vo"].mean(dim="time", skipna=True)

# Magnitude of mean velocity vector
mean_vector_speed = np.hypot(u_mean, v_mean)

result = xr.Dataset(
    {
        "u_mean": u_mean,
        "v_mean": v_mean,
        "mean_vector_speed": mean_vector_speed,
    }
)

result.attrs["description"] = (
    "GLORYS 1994-2022 monthly mean surface-current climatology"
)

output_nc = OUTPUT_DIR / "glorys_climatology_1994_2022.nc"
result.to_netcdf(output_nc, engine="h5netcdf")

print("Saved:", output_nc)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

mesh = ax.pcolormesh(
    result.longitude,
    result.latitude,
    result["mean_vector_speed"],
    shading="auto",
    cmap="viridis"
)

step = 5

ax.quiver(
    result.longitude.values[::step],
    result.latitude.values[::step],
    result["u_mean"].values[::step, ::step],
    result["v_mean"].values[::step, ::step],
    color="white",
    scale=12,
    width=0.002
)

ax.set_xlim(124, 136)
ax.set_ylim(24, 36)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title("GLORYS Mean Surface Currents (1994–2022)")

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Magnitude of mean current (m/s)")

plt.tight_layout()

output_png = FIGURE_DIR / "glorys_climatology_1994_2022.png"
plt.savefig(output_png, dpi=300, bbox_inches="tight")

print("Saved:", output_png)
plt.show()
