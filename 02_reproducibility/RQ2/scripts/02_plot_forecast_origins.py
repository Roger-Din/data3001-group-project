
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "RQ2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load forecast cases
origins = pd.read_parquet(
    DATA_DIR / "forecast_origins.parquet"
)

# Define colors
colors = {
    "training": "#377eb8",
    "calibration": "#ff9f1c",
    "test": "#2ca58d"
}

fig, ax = plt.subplots(figsize=(9, 8))

# Plot each partition separately
for partition in ["training", "calibration", "test"]:
    subset = origins[origins["partition"] == partition]

    ax.scatter(
        subset["origin_longitude"],
        subset["origin_latitude"],
        s=8,
        alpha=0.5,
        color=colors[partition],
        label=f"{partition.capitalize()} (n={len(subset)})"
    )

# Study region
ax.set_xlim(125, 135)
ax.set_ylim(25, 35)
ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")
ax.set_title(
    "Forecast Origin Distribution - East China Sea Region A"
)

ax.set_aspect("equal", adjustable="box")
ax.grid(alpha=0.25)
ax.legend(loc="best")

plt.tight_layout()

# Save figure
output_path = OUTPUT_DIR / "forecast_origins_map.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print("Figure saved to:", output_path)

plt.show()
