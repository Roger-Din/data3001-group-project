
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 1. Paths and configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "RQ2" / "data" /
    "forecast_case_kuroshio_features.csv"
)

DATA_DIR = ROOT / "RQ2" / "data"
FIG_DIR = ROOT / "RQ2" / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = DATA_DIR / "forecast_case_regimes.csv"
SUMMARY_PATH = DATA_DIR / "regime_summary.csv"
DRIFTER_PATH = DATA_DIR / "regime_test_drifters.csv"
FIG_PATH = FIG_DIR / "test_ocean_regimes.png"

TRANSITION_DISTANCE_KM = 50.0

REGIME_ORDER = [
    "Shelf",
    "Kuroshio Core",
    "Transition",
    "Other Offshore",
    "Unclassified",
]

REGIME_COLORS = {
    "Shelf": "#2A9D8F",
    "Kuroshio Core": "#E76F51",
    "Transition": "#E9C46A",
    "Other Offshore": "#457B9D",
    "Unclassified": "#999999",
}

# ============================================================
# 2. Load and validate
# ============================================================

print("=== RQ2 STEP 13: DEFINE OCEAN REGIMES ===")

df = pd.read_csv(INPUT_PATH)

required_columns = [
    "forecast_case_id",
    "ID",
    "partition",
    "origin_longitude",
    "origin_latitude",
    "depth_m",
    "current_speed_clim",
    "glorys_match_valid",
    "current_component_id",
    "distance_to_kuroshio_km",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

if df["forecast_case_id"].duplicated().any():
    raise ValueError(
        "Duplicate forecast_case_id detected."
    )

if len(df) != 5117:
    raise ValueError(
        f"Expected 5117 cases, found {len(df)}"
    )

# ============================================================
# 3. Define four regimes
# ============================================================

df["regime"] = "Unclassified"

# A case is valid only if its physical features exist.
valid = (
    df["depth_m"].notna()
    & (df["depth_m"] >= 0)
    & df["current_speed_clim"].notna()
    & df["distance_to_kuroshio_km"].notna()
    & df["glorys_match_valid"].astype(bool)
)

# Priority 1: Kuroshio Core
is_core = (
    valid
    & df["current_component_id"].isin([1, 2])
)

df.loc[is_core, "regime"] = "Kuroshio Core"

# Priority 2: Shelf
is_shelf = (
    valid
    & ~is_core
    & (df["depth_m"] <= 200)
)

df.loc[is_shelf, "regime"] = "Shelf"

# Priority 3: Transition
is_transition = (
    valid
    & ~is_core
    & ~is_shelf
    & (
        (
            (df["depth_m"] > 200)
            & (df["depth_m"] <= 1000)
        )
        | (
            df["distance_to_kuroshio_km"]
            <= TRANSITION_DISTANCE_KM
        )
    )
)

df.loc[is_transition, "regime"] = "Transition"

# Priority 4: Other Offshore
is_offshore = (
    valid
    & ~is_core
    & ~is_shelf
    & ~is_transition
)

df.loc[is_offshore, "regime"] = "Other Offshore"

df["regime"] = pd.Categorical(
    df["regime"],
    categories=REGIME_ORDER,
    ordered=True,
)

# ============================================================
# 4. Check classification completeness
# ============================================================

print("\n=== ALL CASES ===")
print("Total:", len(df))

print(
    df["regime"]
    .value_counts()
    .reindex(REGIME_ORDER, fill_value=0)
)

assert df["regime"].notna().all()

# ============================================================
# 5. Partition summary
# ============================================================

summary = (
    df.groupby(
        ["partition", "regime"],
        observed=False
    )
    .agg(
        n_cases=("forecast_case_id", "count"),
        n_drifters=("ID", "nunique"),
        median_depth_m=("depth_m", "median"),
        median_current_speed_m_s=(
            "current_speed_clim", "median"
        ),
        median_distance_to_core_km=(
            "distance_to_kuroshio_km", "median"
        ),
    )
    .reset_index()
)

summary.to_csv(SUMMARY_PATH, index=False)

print("\n=== CASE COUNTS BY PARTITION ===")

counts = pd.crosstab(
    df["partition"],
    df["regime"]
).reindex(columns=REGIME_ORDER, fill_value=0)

print(counts.to_string())

print("\n=== DRIFTER COUNTS BY PARTITION ===")

drifter_counts = (
    df.groupby(
        ["partition", "regime"],
        observed=False
    )["ID"]
    .nunique()
    .unstack(fill_value=0)
    .reindex(columns=REGIME_ORDER, fill_value=0)
)

print(drifter_counts.to_string())

# ============================================================
# 6. Test-set diagnostics
# ============================================================

test = df[df["partition"] == "test"].copy()

if len(test) != 757:
    raise ValueError(
        f"Expected 757 test cases, found {len(test)}"
    )

print("\n=== TEST REGIME DISTRIBUTION ===")

test_counts = (
    test["regime"]
    .value_counts()
    .reindex(REGIME_ORDER, fill_value=0)
)

test_percent = (
    test_counts / len(test) * 100
).round(2)

test_summary = pd.DataFrame({
    "n_cases": test_counts,
    "percent": test_percent,
})

print(test_summary.to_string())

print("\n=== TEST DRIFTER OVERLAP ===")

drifter_regime_counts = (
    test.groupby("ID", observed=True)["regime"]
    .nunique()
)

print(
    "Total unique test drifters:",
    test["ID"].nunique()
)

print(
    "Drifters in multiple regimes:",
    int((drifter_regime_counts > 1).sum())
)

# Save drifter-by-regime case counts
drifter_table = pd.crosstab(
    test["ID"],
    test["regime"]
).reindex(columns=REGIME_ORDER, fill_value=0)

drifter_table.to_csv(DRIFTER_PATH)

print("\n=== TEST DRIFTER TABLE ===")
print("Shape:", drifter_table.shape)

# ============================================================
# 7. Plot test cases by regime
# ============================================================

fig, ax = plt.subplots(figsize=(11, 9))

for regime in REGIME_ORDER:

    subset = test[
        test["regime"] == regime
    ]

    if subset.empty:
        continue

    ax.scatter(
        subset["origin_longitude"],
        subset["origin_latitude"],
        s=18,
        alpha=0.75,
        c=REGIME_COLORS[regime],
        label=f"{regime} (n={len(subset)})",
        edgecolors="none",
    )

ax.set_xlim(124, 136)
ax.set_ylim(24, 36)

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

ax.set_title(
    "RQ2 Oceanographic Regimes\n"
    "Test Forecast Origins (n=757)"
)

ax.legend(
    loc="best",
    fontsize=10,
    framealpha=0.9,
)

ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(
    FIG_PATH,
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

# ============================================================
# 8. Save and finish
# ============================================================

df.to_csv(OUTPUT_PATH, index=False)

print("\n=== SAVED FILES ===")
print(OUTPUT_PATH)
print(SUMMARY_PATH)
print(DRIFTER_PATH)
print(FIG_PATH)

print("\n=== STEP 13 COMPLETE ===")
