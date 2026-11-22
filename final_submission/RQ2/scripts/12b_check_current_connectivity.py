
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]

CLIM_PATH = (
    ROOT / "RQ2" / "data" /
    "glorys_climatology_1994_2022.nc"
)

OUTPUT_DIR = ROOT / "RQ2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with xr.open_dataset(CLIM_PATH) as ds:
    clim = ds.load()

u = clim["u_mean"].squeeze().transpose(
    "latitude", "longitude"
).values

v = clim["v_mean"].squeeze().transpose(
    "latitude", "longitude"
).values

speed = np.hypot(u, v)

lon = clim["longitude"].values
lat = clim["latitude"].values

thresholds = [0.4, 0.5, 0.6]

# Eight-neighbour connectivity
structure = np.ones((3, 3), dtype=int)

fig, axes = plt.subplots(
    1, 3, figsize=(18, 6),
    sharex=True, sharey=True
)

for ax, threshold in zip(axes, thresholds):

    mask = np.isfinite(speed) & (speed >= threshold)

    labels, n_components = ndimage.label(
        mask, structure=structure
    )

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest_label = np.argmax(sizes)
    largest_size = sizes[largest_label]

    largest_mask = labels == largest_label

    print(f"\nThreshold: {threshold} m/s")
    print("Connected components:", n_components)
    print("Largest component cells:", largest_size)
    print("All high-current cells:", mask.sum())
    print(
        "Largest component fraction:",
        round(largest_size / max(mask.sum(), 1), 3)
    )

    ax.pcolormesh(
        lon, lat, speed,
        cmap="Greys",
        vmin=0, vmax=1.1,
        shading="auto"
    )

    ax.contourf(
        lon, lat, mask.astype(int),
        levels=[0.5, 1.5],
        colors=["orange"],
        alpha=0.45
    )

    ax.contourf(
        lon, lat, largest_mask.astype(int),
        levels=[0.5, 1.5],
        colors=["red"],
        alpha=0.75
    )

    ax.set_xlim(124, 136)
    ax.set_ylim(24, 36)
    ax.set_title(
        f"Threshold >= {threshold} m/s\n"
        f"Largest component: {largest_size} cells"
    )
    ax.set_xlabel("Longitude (°E)")
    ax.grid(alpha=0.2)

axes[0].set_ylabel("Latitude (°N)")

fig.suptitle(
    "GLORYS Climatological High-Current Connectivity",
    fontsize=15
)

plt.tight_layout()

output = OUTPUT_DIR / "current_connectivity.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", output)
print("=== STEP 12B COMPLETE ===")
