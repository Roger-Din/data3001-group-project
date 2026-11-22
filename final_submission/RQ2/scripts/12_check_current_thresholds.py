
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "RQ2" / "data" /
    "forecast_case_ocean_features.csv"
)

df = pd.read_csv(DATA_PATH)

test = df[df["partition"] == "test"].copy()

print("=== RQ2 STEP 12: CURRENT THRESHOLD CHECK ===")

print("\nTest cases:", len(test))
print("Unique test drifters:", test["ID"].nunique())

# Examine candidate thresholds, without defining regimes yet.
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

rows = []

for threshold in thresholds:
    subset = test[
        (test["depth_m"] > 200)
        & (test["current_speed_clim"] >= threshold)
    ]

    rows.append({
        "threshold_m_s": threshold,
        "n_cases": len(subset),
        "n_drifters": subset["ID"].nunique(),
        "percent_test": round(
            len(subset) / len(test) * 100, 2
        ),
        "median_depth_m": (
            subset["depth_m"].median()
            if len(subset) else np.nan
        )
    })

summary = pd.DataFrame(rows)

print("\n=== CANDIDATE HIGH-CURRENT SAMPLES ===")
print(summary.to_string(index=False))

print("\n=== TEST SPEED QUANTILES ===")
print(
    test["current_speed_clim"].quantile(
        [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    )
)

print("\n=== HIGH-CURRENT CASES BY DEPTH BAND ===")

for threshold in [0.3, 0.5, 0.7]:
    print(f"\nThreshold >= {threshold} m/s")

    subset = test[
        test["current_speed_clim"] >= threshold
    ]

    print(subset["depth_band"].value_counts())

print("\n=== STEP 12A COMPLETE ===")
