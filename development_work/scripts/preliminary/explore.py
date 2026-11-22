# %%
# ============================================================
# DATA3001 — Project IV
# Region Selection and Data Screening Pipeline
#
# Purpose
# -------
# Compare candidate study regions using the same reproducible
# diagnostics:
#
# 1. Basic data availability
# 2. Spatial observation density
# 3. Temporal gaps
# 4. Continuous trajectory segments
# 5. Temporal coverage
# 6. Era-stratified segment continuity
# 7. Forecast evaluation-origin budget
# 8. Per-drifter sample concentration
#
# IMPORTANT
# ---------
# The 144-hour spacing used below is currently a WORKING
# ASSUMPTION, not a measured decorrelation timescale.
#
# Later, after velocity data (ve/vn) are obtained, estimate
# the Lagrangian integral timescale T_L and replace this value.
# ============================================================


# %%
# ============================================================
# 0. Imports
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
# ============================================================
# 1. CONFIGURATION
#
# WHEN TESTING NEW REGIONS:
# EDIT ONLY THIS SECTION.
# ============================================================

BASE = os.path.join("data", "regions")

REGIONS = {

    "A": {
        "name": "East China Sea / Kuroshio upstream",
        "path": f"{BASE}/region_A.csv",
        "box": (125, 135, 25, 35),
    },

    "B": {
        "name": "Kuroshio Extension",
        "path": f"{BASE}/region_B.csv",
        "box": (135, 145, 30, 40),
    },

    "C": {
        "name": "Subtropical Countercurrent region",
        "path": f"{BASE}/region_C.csv",
        "box": (135, 145, 20, 30),
    },

    "B_prime": {
        "name": "Revised Kuroshio Extension",
        "path": f"{BASE}/region_B_prime.csv",
        "box": (142, 152, 30, 40),
    },
}

# Forecast design assumptions
LOOKBACK_H = 24
HORIZON_H = 72

# Preliminary independence spacing.
# Replace later using measured Lagrangian decorrelation time.
DECORR_H = 144


ERA_BINS = [
    (1990, 1999),
    (2000, 2004),
    (2005, 2009),
    (2010, 2014),
    (2015, 2019),
    (2020, 2025),
]


# %%
# ============================================================
# 2. Helper: load and validate ERDDAP data
# ============================================================

def load_region(path):
    """
    Load one ERDDAP drifter CSV.

    ERDDAP CSV contains:
        row 0 = column names
        row 1 = units

    Therefore skiprows=[1].
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}"
        )

    d = pd.read_csv(path, skiprows=[1])

    required = [
        "ID",
        "time",
        "latitude",
        "longitude",
    ]

    missing = [
        c for c in required
        if c not in d.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(d.columns)}"
        )

    d["time"] = pd.to_datetime(
        d["time"],
        utc=True,
        errors="coerce"
    )

    # Remove invalid rows
    d = d.dropna(
        subset=[
            "ID",
            "time",
            "latitude",
            "longitude",
        ]
    ).copy()

    # Sort consistently
    d = (
        d.sort_values(["ID", "time"])
         .reset_index(drop=True)
    )

    return d


# %%
# ============================================================
# 3. Helper: construct continuous hourly segments
# ============================================================

def get_segments(d):
    """
    Split each drifter trajectory whenever consecutive
    observations are not exactly one hour apart.

    25 hourly observations correspond to 24 hours duration.
    """

    x = d.sort_values(
        ["ID", "time"]
    ).copy()

    gap_seconds = (
        x.groupby("ID")["time"]
         .diff()
         .dt.total_seconds()
    )

    x["gap_hours"] = (
        gap_seconds / 3600
    )

    # New segment:
    # first observation OR gap != 3600 seconds
    x["new_segment"] = (
        gap_seconds.isna()
        | (gap_seconds.round() != 3600)
    )

    x["segment"] = (
        x.groupby("ID")["new_segment"]
         .cumsum()
    )

    seg = (
        x.groupby(["ID", "segment"])
         .agg(
             start=("time", "min"),
             end=("time", "max"),
             n_obs=("time", "size"),
             start_lat=("latitude", "first"),
             start_lon=("longitude", "first"),
             end_lat=("latitude", "last"),
             end_lon=("longitude", "last"),
         )
         .reset_index()
    )

    seg["duration_hours"] = (
        seg["n_obs"] - 1
    )

    seg["start_year"] = (
        seg["start"].dt.year
    )

    return x, seg


# %%
# ============================================================
# 4. Helper: era labels
# ============================================================

def era_label(year):

    for lo, hi in ERA_BINS:

        if lo <= year <= hi:
            return f"{lo}-{hi}"

    return "other"


# %%
# ============================================================
# 5. Helper: basic data summary
# ============================================================

def basic_summary(d):

    drifter_summary = (
        d.groupby("ID")
         .agg(
             observations=("time", "size"),
             start=("time", "min"),
             end=("time", "max"),
         )
    )

    drifter_summary["duration_days"] = (
        (
            drifter_summary["end"]
            - drifter_summary["start"]
        )
        .dt.total_seconds()
        / 86400
    )

    result = {
        "observations": len(d),
        "unique_drifters": d["ID"].nunique(),
        "start_time": d["time"].min(),
        "end_time": d["time"].max(),
        "median_drifter_duration_days":
            drifter_summary["duration_days"].median(),
        "mean_drifter_duration_days":
            drifter_summary["duration_days"].mean(),
        "drifters_ge_1d":
            int((drifter_summary["duration_days"] >= 1).sum()),
        "drifters_ge_2d":
            int((drifter_summary["duration_days"] >= 2).sum()),
        "drifters_ge_7d":
            int((drifter_summary["duration_days"] >= 7).sum()),
        "drifters_ge_12d":
            int((drifter_summary["duration_days"] >= 12).sum()),
    }

    return result, drifter_summary


# %%
# ============================================================
# 6. Helper: gap diagnostics
# ============================================================

def gap_summary(x):

    gaps = x["gap_hours"].dropna()

    result = {
        "n_gaps": len(gaps),

        "median_gap_h":
            gaps.median(),

        "mean_gap_h":
            gaps.mean(),

        "gap_eq_1h":
            int((gaps == 1).sum()),

        "gap_gt_1h":
            int((gaps > 1).sum()),

        "gap_gt_2h":
            int((gaps > 2).sum()),

        "gap_gt_6h":
            int((gaps > 6).sum()),

        "gap_gt_24h":
            int((gaps > 24).sum()),

        "pct_exact_hourly":
            100 * (gaps == 1).mean()
            if len(gaps) else np.nan,
    }

    return result


# %%
# ============================================================
# 7. Helper: continuous segment summary
# ============================================================

def segment_summary(seg):

    duration = seg["duration_hours"]

    result = {
        "total_segments":
            len(seg),

        "median_segment_h":
            duration.median(),

        "mean_segment_h":
            duration.mean(),

        "p90_segment_h":
            duration.quantile(0.90),

        "max_segment_h":
            duration.max(),

        "segments_ge_24h":
            int((duration >= 24).sum()),

        "segments_ge_48h":
            int((duration >= 48).sum()),

        "segments_ge_72h":
            int((duration >= 72).sum()),

        "segments_ge_168h":
            int((duration >= 168).sum()),

        "segments_ge_288h":
            int((duration >= 288).sum()),
    }

    return result


# %%
# ============================================================
# 8. Helper: temporal coverage
# ============================================================

def year_summary(d):

    y = d.copy()

    y["year"] = y["time"].dt.year

    result = (
        y.groupby("year")
         .agg(
             observations=("ID", "size"),
             unique_drifters=("ID", "nunique"),
         )
         .reset_index()
    )

    return result


# %%
# ============================================================
# 9. Helper: era-stratified continuity
# ============================================================

def era_summary(seg):

    s = seg.copy()

    s["era"] = (
        s["start_year"]
        .map(era_label)
    )

    result = (
        s.groupby("era")
         .agg(
             n_segments=(
                 "duration_hours",
                 "size"
             ),

             median_h=(
                 "duration_hours",
                 "median"
             ),

             mean_h=(
                 "duration_hours",
                 "mean"
             ),

             p90_h=(
                 "duration_hours",
                 lambda x:
                 x.quantile(0.90)
             ),

             n_ge_168h=(
                 "duration_hours",
                 lambda x:
                 int((x >= 168).sum())
             ),
         )
    )

    return result


# %%
# ============================================================
# 10. Helper: forecast evaluation-origin budget
# ============================================================

def evaluation_budget(
    seg,
    lookback_h=LOOKBACK_H,
    horizon_h=HORIZON_H,
    decorr_h=DECORR_H,
):
    """
    Estimate the number of approximately independent
    forecast evaluation origins.

    Forecast origin requires:

        lookback + horizon

    hours of continuous observations.

    Evaluation origins from the same segment are separated
    by DECORR_H.

    IMPORTANT:
    DECORR_H is currently preliminary.
    """

    need = (
        lookback_h
        + horizon_h
    )

    usable = (
        seg[
            seg["duration_hours"] >= need
        ]
        .copy()
    )

    usable["n_eval"] = (
        1
        + (
            (
                usable["duration_hours"]
                - need
            )
            // decorr_h
        )
    ).astype(int)

    result = {
        "total_segments":
            len(seg),

        "segments_ge_96h":
            len(usable),

        "independent_samples":
            int(usable["n_eval"].sum()),

        "median_samples_per_usable_segment":
            (
                usable["n_eval"].median()
                if len(usable)
                else 0
            ),

        "mean_samples_per_usable_segment":
            (
                usable["n_eval"].mean()
                if len(usable)
                else 0
            ),

        "max_samples_from_one_segment":
            (
                usable["n_eval"].max()
                if len(usable)
                else 0
            ),
    }

    return result, usable


# %%
# ============================================================
# 11. Helper: per-drifter sample concentration
# ============================================================

def concentration_summary(usable):

    if len(usable) == 0:

        return {
            "usable_drifters": 0,
            "total_origins": 0,
            "origins_per_drifter_mean": 0,
            "origins_per_drifter_median": 0,
            "top10pct_share": np.nan,
            "top20pct_share": np.nan,
            "largest_drifter_share": np.nan,
            "drifters_ge_10_origins": 0,
            "max_origins_one_drifter": 0,
        }

    per_drifter = (
        usable.groupby("ID")["n_eval"]
              .sum()
              .sort_values(
                  ascending=False
              )
    )

    total = per_drifter.sum()

    n_drifters = len(
        per_drifter
    )

    top10_n = max(
        1,
        int(
            np.ceil(
                n_drifters * 0.10
            )
        )
    )

    top20_n = max(
        1,
        int(
            np.ceil(
                n_drifters * 0.20
            )
        )
    )

    result = {

        "usable_drifters":
            n_drifters,

        "total_origins":
            int(total),

        "origins_per_drifter_mean":
            per_drifter.mean(),

        "origins_per_drifter_median":
            per_drifter.median(),

        "top10pct_share":
            (
                100
                * per_drifter.head(
                    top10_n
                ).sum()
                / total
            ),

        "top20pct_share":
            (
                100
                * per_drifter.head(
                    top20_n
                ).sum()
                / total
            ),

        "largest_drifter_share":
            (
                100
                * per_drifter.iloc[0]
                / total
            ),

        "drifters_ge_10_origins":
            int(
                (
                    per_drifter >= 10
                ).sum()
            ),

        "max_origins_one_drifter":
            int(
                per_drifter.iloc[0]
            ),
    }

    return result


# %%
# ============================================================
# 12. Helper: density map
# ============================================================

def plot_density(d, code, info):

    lon_min, lon_max, lat_min, lat_max = (
        info["box"]
    )

    plt.figure(
        figsize=(8, 7)
    )

    plt.hist2d(
        d["longitude"],
        d["latitude"],
        bins=50,
        cmap="viridis",
    )

    plt.colorbar(
        label="Number of observations"
    )

    plt.xlabel(
        "Longitude"
    )

    plt.ylabel(
        "Latitude"
    )

    plt.title(
        f"Candidate {code}: Observation Density\n"
        f"{info['name']}"
    )

    plt.xlim(
        lon_min,
        lon_max
    )

    plt.ylim(
        lat_min,
        lat_max
    )

    plt.tight_layout()

    plt.show()


# %%
# ============================================================
# 13. Helper: yearly drifter plot
# ============================================================

def plot_yearly_drifters(
    ys,
    code,
    info
):

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        ys["year"],
        ys["unique_drifters"]
    )

    plt.xlabel(
        "Year"
    )

    plt.ylabel(
        "Number of unique drifters"
    )

    plt.title(
        f"Candidate {code}: Unique Drifters by Year\n"
        f"{info['name']}"
    )

    plt.xticks(
        rotation=45
    )

    plt.tight_layout()

    plt.show()


# %%
# ============================================================
# 14. RUN PIPELINE
# ============================================================

all_data = {}
all_segments = {}

basic_results = {}
gap_results = {}
segment_results = {}
year_results = {}
era_results = {}
budget_results = {}
concentration_results = {}


for code, info in REGIONS.items():

    print("\n")
    print("=" * 72)
    print(
        f"Candidate {code}: {info['name']}"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    d = load_region(
        info["path"]
    )

    x, seg = get_segments(
        d
    )

    all_data[code] = d
    all_segments[code] = seg

    # --------------------------------------------------------
    # Basic summary
    # --------------------------------------------------------

    basic, drifter_summary = (
        basic_summary(d)
    )

    basic_results[code] = basic

    print(
        "\n--- 1. BASIC DATA AVAILABILITY ---"
    )

    for k, v in basic.items():
        print(
            f"{k:<35}: {v}"
        )

    # --------------------------------------------------------
    # Density
    # --------------------------------------------------------

    print(
        "\n--- 2. SPATIAL DENSITY ---"
    )

    plot_density(
        d,
        code,
        info
    )

    # --------------------------------------------------------
    # Gaps
    # --------------------------------------------------------

    gaps = gap_summary(
        x
    )

    gap_results[code] = gaps

    print(
        "\n--- 3. TEMPORAL GAPS ---"
    )

    for k, v in gaps.items():

        if isinstance(
            v,
            (float, np.floating)
        ):
            print(
                f"{k:<35}: {v:.2f}"
            )
        else:
            print(
                f"{k:<35}: {v}"
            )

    # --------------------------------------------------------
    # Segments
    # --------------------------------------------------------

    ss = segment_summary(
        seg
    )

    segment_results[code] = ss

    print(
        "\n--- 4. CONTINUOUS SEGMENTS ---"
    )

    for k, v in ss.items():

        if isinstance(
            v,
            (float, np.floating)
        ):
            print(
                f"{k:<35}: {v:.2f}"
            )
        else:
            print(
                f"{k:<35}: {v}"
            )

    # --------------------------------------------------------
    # Year coverage
    # --------------------------------------------------------

    ys = year_summary(
        d
    )

    year_results[code] = ys

    print(
        "\n--- 5. TEMPORAL COVERAGE ---"
    )

    print(
        ys.to_string(
            index=False
        )
    )

    plot_yearly_drifters(
        ys,
        code,
        info
    )

    # --------------------------------------------------------
    # Era
    # --------------------------------------------------------

    es = era_summary(
        seg
    )

    era_results[code] = es

    print(
        "\n--- 6. ERA-STRATIFIED CONTINUITY ---"
    )

    print(
        es.to_string()
    )

    # --------------------------------------------------------
    # Evaluation budget
    # --------------------------------------------------------

    budget, usable = (
        evaluation_budget(seg)
    )

    budget_results[code] = budget

    print(
        "\n--- 7. EVALUATION-ORIGIN BUDGET ---"
    )

    print(
        f"Lookback = {LOOKBACK_H} h"
    )

    print(
        f"Horizon  = {HORIZON_H} h"
    )

    print(
        f"Preliminary spacing = {DECORR_H} h"
    )

    for k, v in budget.items():

        if isinstance(
            v,
            (float, np.floating)
        ):
            print(
                f"{k:<40}: {v:.2f}"
            )
        else:
            print(
                f"{k:<40}: {v}"
            )

    # --------------------------------------------------------
    # Concentration
    # --------------------------------------------------------

    concentration = (
        concentration_summary(
            usable
        )
    )

    concentration_results[
        code
    ] = concentration

    print(
        "\n--- 8. PER-DRIFTER SAMPLE CONCENTRATION ---"
    )

    for k, v in concentration.items():

        if isinstance(
            v,
            (float, np.floating)
        ):
            print(
                f"{k:<40}: {v:.2f}"
            )
        else:
            print(
                f"{k:<40}: {v}"
            )


# %%
# ============================================================
# 15. SIDE-BY-SIDE COMPARISON TABLES
# ============================================================

basic_table = (
    pd.DataFrame(
        basic_results
    ).T
)

gap_table = (
    pd.DataFrame(
        gap_results
    ).T
)

segment_table = (
    pd.DataFrame(
        segment_results
    ).T
)

budget_table = (
    pd.DataFrame(
        budget_results
    ).T
)

concentration_table = (
    pd.DataFrame(
        concentration_results
    ).T
)


print("\n")
print("=" * 72)
print("TABLE 1 — BASIC DATA AVAILABILITY")
print("=" * 72)
print(
    basic_table.to_string()
)


print("\n")
print("=" * 72)
print("TABLE 2 — TEMPORAL GAPS")
print("=" * 72)
print(
    gap_table.to_string()
)


print("\n")
print("=" * 72)
print("TABLE 3 — CONTINUOUS SEGMENTS")
print("=" * 72)
print(
    segment_table.to_string()
)


print("\n")
print("=" * 72)
print("TABLE 4 — EVALUATION-ORIGIN BUDGET")
print("=" * 72)
print(
    budget_table.to_string()
)


print("\n")
print("=" * 72)
print("TABLE 5 — SAMPLE CONCENTRATION")
print("=" * 72)
print(
    concentration_table.to_string()
)


# %%
# ============================================================
# 16. COMPACT REGION-SELECTION TABLE
# ============================================================

decision_table = pd.DataFrame(
    index=list(REGIONS.keys())
)


decision_table["observations"] = [
    basic_results[r][
        "observations"
    ]
    for r in decision_table.index
]


decision_table["unique_drifters"] = [
    basic_results[r][
        "unique_drifters"
    ]
    for r in decision_table.index
]


decision_table["pct_exact_hourly"] = [
    gap_results[r][
        "pct_exact_hourly"
    ]
    for r in decision_table.index
]


decision_table["continuous_segments"] = [
    segment_results[r][
        "total_segments"
    ]
    for r in decision_table.index
]


decision_table["segments_ge_48h"] = [
    segment_results[r][
        "segments_ge_48h"
    ]
    for r in decision_table.index
]


decision_table["segments_ge_168h"] = [
    segment_results[r][
        "segments_ge_168h"
    ]
    for r in decision_table.index
]


decision_table["segments_ge_288h"] = [
    segment_results[r][
        "segments_ge_288h"
    ]
    for r in decision_table.index
]


decision_table["usable_drifters"] = [
    concentration_results[r][
        "usable_drifters"
    ]
    for r in decision_table.index
]


decision_table["evaluation_origins"] = [
    budget_results[r][
        "independent_samples"
    ]
    for r in decision_table.index
]


decision_table["origins_per_drifter"] = [
    concentration_results[r][
        "origins_per_drifter_mean"
    ]
    for r in decision_table.index
]


decision_table["top10pct_share"] = [
    concentration_results[r][
        "top10pct_share"
    ]
    for r in decision_table.index
]


print("\n")
print("=" * 72)
print("FINAL REGION-SELECTION DIAGNOSTIC TABLE")
print("=" * 72)

print(
    decision_table.round(2).to_string()
)


# %%
# ============================================================
# 17. ERA TABLES
# ============================================================

print("\n")
print("=" * 72)
print("ERA-STRATIFIED RESULTS")
print("=" * 72)

for code in REGIONS:

    print("\n")
    print(
        f"Candidate {code}: "
        f"{REGIONS[code]['name']}"
    )

    print("-" * 72)

    print(
        era_results[
            code
        ].to_string()
    )


# %%
# ============================================================
# 18. FINAL NOTES
# ============================================================

print("\n")
print("=" * 72)
print("IMPORTANT INTERPRETATION NOTES")
print("=" * 72)

print(
    """
1. Raw observations are NOT independent samples.

2. Continuous segments are more informative than raw row counts
   for trajectory forecasting.

3. A forecast origin currently requires:
       24 h lookback + 72 h forecast horizon = 96 h continuous data.

4. The 144 h spacing between evaluation origins is preliminary.
   It must later be replaced by a value informed by the measured
   Lagrangian velocity decorrelation / integral timescale.

5. 'Independent_samples' therefore means approximately spaced
   evaluation origins under the current working assumption,
   NOT formally independent statistical observations.

6. Per-drifter concentration is reported because many evaluation
   origins may come from the same drifter.

7. Train / calibration / test splitting must NOT be performed
   randomly at the row or forecast-origin level.

8. Later stages should use drifter-level or correlation-cluster-level
   splitting to reduce information leakage.

9. Additional GDP variables should later be obtained where available:
       - ve
       - vn
       - drogue status / drogue-loss information
       - location type
       - uncertainty / error variables

10. Final region selection should combine:
       (a) data adequacy,
       (b) effective independent sampling,
       (c) spatial/temporal coverage,
       (d) oceanographic interpretability,
       (e) relevance to the forecasting and uncertainty task.
"""
)