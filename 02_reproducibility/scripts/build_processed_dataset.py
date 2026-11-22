#!/usr/bin/env python3
"""Build reproducible processed datasets for the DATA3001 Region A project.

The script keeps RegionA.csv unchanged and writes five analysis-ready files:

    data/processed/region_A_observations.parquet
    data/processed/region_A_segments.parquet
    data/processed/drifter_splits.csv
    data/processed/forecast_origins.parquet
    data/processed/qc_summary.csv

Run from the repository root with:

    .venv/bin/python build_processed_dataset.py

Quality-control flags are retained for review. High-speed, land-mask, Argos,
and drogue-lost records are not automatically discarded.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


LON_MIN, LON_MAX = 125.0, 135.0
LAT_MIN, LAT_MAX = 25.0, 35.0
INPUT_STEPS = 24
TARGET_STEPS = 72
WINDOW_STEPS = INPUT_STEPS + TARGET_STEPS
DEFAULT_SEED = 3001
DEFAULT_MAX_ACF_LAG = 168


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed Region A drifter datasets"
    )
    parser.add_argument("--data", type=Path, default=Path("RegionA.csv"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed")
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-acf-lag",
        type=int,
        default=DEFAULT_MAX_ACF_LAG,
        help="Maximum hourly lag used for the training-only velocity ACF",
    )
    parser.add_argument(
        "--origin-spacing-hours",
        type=int,
        default=None,
        help=(
            "Override automatic origin spacing. The default is the larger of "
            "96 hours and the estimated Lagrangian integral timescale."
        ),
    )
    parser.add_argument(
        "--skip-land-mask",
        action="store_true",
        help="Skip the optional global-land-mask check to reduce memory use",
    )
    return parser.parse_args()


def read_erddap_csv(path: Path) -> pd.DataFrame:
    """Read an ERDDAP CSV while detecting its optional units row."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    preview = pd.read_csv(path, nrows=1, low_memory=False)
    has_units_row = (
        "latitude" in preview.columns
        and not preview.empty
        and pd.isna(
            pd.to_numeric(preview.loc[0, "latitude"], errors="coerce")
        )
    )
    return pd.read_csv(
        path,
        skiprows=[1] if has_units_row else None,
        low_memory=False,
    )


def clean_and_flag_observations(
    raw: pd.DataFrame, use_land_mask: bool
) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    """Standardise types, remove unusable key records, and add QC flags."""
    required = {"ID", "time", "latitude", "longitude", "ve", "vn"}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = raw.copy()
    df["source_row"] = np.arange(len(df), dtype=np.int64)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

    numeric_columns = [
        "ID",
        "latitude",
        "longitude",
        "sst",
        "flg_sst",
        "ve",
        "vn",
        "err_lat",
        "err_lon",
        "err_ve",
        "err_vn",
        "gap",
        "deploy_lat",
        "deploy_lon",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    core = ["ID", "time", "latitude", "longitude", "ve", "vn"]
    df["flag_core_missing"] = df[core].isna().any(axis=1)
    df["flag_invalid_coordinate"] = ~(
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )
    df["flag_outside_region"] = ~(
        df["longitude"].between(LON_MIN, LON_MAX)
        & df["latitude"].between(LAT_MIN, LAT_MAX)
    )

    exact_duplicates = int(df.duplicated(subset=raw.columns.tolist()).sum())
    duplicate_id_time = int(df.duplicated(["ID", "time"]).sum())
    duplicate_id_time_any = df.duplicated(["ID", "time"], keep=False)
    df["flag_duplicate_id_time"] = duplicate_id_time_any

    # Exact duplicates and repeated ID-time records cannot define two distinct
    # hourly states. Keep the first deterministically and report every removal.
    df = df.drop_duplicates(subset=raw.columns.tolist(), keep="first")
    df = df.drop_duplicates(["ID", "time"], keep="first")

    unusable = (
        df["flag_core_missing"]
        | df["flag_invalid_coordinate"]
        | df["flag_outside_region"]
    )
    unusable_rows = int(unusable.sum())
    df = df.loc[~unusable].copy()

    # IDs are complete at this point and are stored as integers.
    df["ID"] = df["ID"].astype("int64")
    df = df.sort_values(["ID", "time"]).reset_index(drop=True)
    df["speed"] = np.hypot(df["ve"], df["vn"])
    df["flag_high_speed"] = df["speed"] > 2.0
    df["flag_extreme_speed"] = df["speed"] > 3.0
    df["flag_missing_position_error"] = df[
        [c for c in ["err_lat", "err_lon"] if c in df.columns]
    ].isna().any(axis=1)
    df["flag_missing_velocity_error"] = df[
        [c for c in ["err_ve", "err_vn"] if c in df.columns]
    ].isna().any(axis=1)

    if "drogue_lost_date" in df.columns:
        df["drogue_lost_time"] = pd.to_datetime(
            df["drogue_lost_date"], utc=True, errors="coerce"
        )
        known_loss = df["drogue_lost_time"].notna()
        df["drogue_status"] = np.select(
            [
                known_loss & (df["time"] < df["drogue_lost_time"]),
                known_loss & (df["time"] >= df["drogue_lost_time"]),
            ],
            ["attached", "lost"],
            default="unknown",
        )
    else:
        df["drogue_lost_time"] = pd.NaT
        df["drogue_status"] = "unknown"

    land_mask_status = "skipped"
    df["flag_land_mask"] = False
    if use_land_mask:
        try:
            from global_land_mask import globe

            df["flag_land_mask"] = globe.is_land(
                df["latitude"].to_numpy(),
                df["longitude"].to_numpy(),
            )
            land_mask_status = "computed"
        except ImportError:
            land_mask_status = "unavailable"

    df["flag_any_review"] = df[
        [
            "flag_high_speed",
            "flag_extreme_speed",
            "flag_missing_position_error",
            "flag_missing_velocity_error",
            "flag_land_mask",
        ]
    ].any(axis=1)

    metrics: dict[str, int | float | str] = {
        "raw_rows": len(raw),
        "exact_duplicate_rows_removed": exact_duplicates,
        "duplicate_ID_time_rows_removed": duplicate_id_time,
        "unusable_core_or_coordinate_rows_removed": unusable_rows,
        "processed_observation_rows": len(df),
        "unique_drifters": df["ID"].nunique(),
        "core_missing_after_processing": int(df[core].isna().any(axis=1).sum()),
        "outside_region_after_processing": int(df["flag_outside_region"].sum()),
        "speed_above_2_m_per_s": int(df["flag_high_speed"].sum()),
        "speed_above_3_m_per_s": int(df["flag_extreme_speed"].sum()),
        "land_mask_status": land_mask_status,
        "land_mask_flagged_observations": int(df["flag_land_mask"].sum()),
        "observations_missing_position_error": int(
            df["flag_missing_position_error"].sum()
        ),
        "observations_missing_velocity_error": int(
            df["flag_missing_velocity_error"].sum()
        ),
    }
    return df, metrics


def add_continuous_segments(
    observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign approximately hourly continuous segments and summarise them."""
    df = observations.sort_values(["ID", "time"]).copy()
    df["dt_hours"] = (
        df.groupby("ID", sort=False)["time"]
        .diff()
        .dt.total_seconds()
        .div(3600)
    )
    df["flag_time_gap"] = (
        df["dt_hours"].notna() & ~df["dt_hours"].between(0.5, 1.5)
    )
    df["new_segment"] = df["dt_hours"].isna() | df["flag_time_gap"]
    df["segment_id"] = (
        df.groupby("ID", sort=False)["new_segment"].cumsum().astype("int64")
    )
    df["segment_step"] = (
        df.groupby(["ID", "segment_id"], sort=False).cumcount().astype("int64")
    )

    segments = (
        df.groupby(["ID", "segment_id"], sort=False)
        .agg(
            start_time=("time", "min"),
            end_time=("time", "max"),
            n_observations=("time", "size"),
            n_high_speed_flags=("flag_high_speed", "sum"),
            n_extreme_speed_flags=("flag_extreme_speed", "sum"),
            n_land_mask_flags=("flag_land_mask", "sum"),
            n_review_flags=("flag_any_review", "sum"),
            GPS_observations=(
                "location_type",
                lambda s: int((s.astype(str).str.upper() == "GPS").sum()),
            ),
            Argos_observations=(
                "location_type",
                lambda s: int((s.astype(str).str.upper() == "ARGOS").sum()),
            ),
        )
        .reset_index()
    )
    segments["elapsed_hours"] = (
        segments["end_time"] - segments["start_time"]
    ).dt.total_seconds().div(3600)
    segments["eligible_24_to_72"] = (
        segments["n_observations"] >= WINDOW_STEPS
    )
    segments["potential_overlapping_windows"] = (
        segments["n_observations"] - WINDOW_STEPS + 1
    ).clip(lower=0)
    return df, segments


def make_drifter_splits(
    observations: pd.DataFrame,
    segments: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    """Assign each complete drifter ID to exactly one partition."""
    ids = np.sort(observations["ID"].unique()).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_train = int(0.70 * len(ids))
    n_calibration = int(0.15 * len(ids))

    assignments = pd.DataFrame(
        {
            "ID": ids,
            "partition": (
                ["training"] * n_train
                + ["calibration"] * n_calibration
                + ["test"] * (len(ids) - n_train - n_calibration)
            ),
        }
    )

    observation_summary = (
        observations.groupby("ID")
        .agg(
            n_observations=("time", "size"),
            first_time=("time", "min"),
            last_time=("time", "max"),
        )
        .reset_index()
    )
    segment_summary = (
        segments.groupby("ID")
        .agg(
            n_segments=("segment_id", "size"),
            n_eligible_segments=("eligible_24_to_72", "sum"),
        )
        .reset_index()
    )
    splits = assignments.merge(
        observation_summary, on="ID", how="left", validate="one_to_one"
    ).merge(segment_summary, on="ID", how="left", validate="one_to_one")

    if splits["ID"].duplicated().any():
        raise AssertionError("A drifter ID appears more than once in splits")
    if set(splits["ID"]) != set(observations["ID"].unique()):
        raise AssertionError("Split table does not cover every drifter")
    return splits.sort_values("ID").reset_index(drop=True)


def estimate_lagrangian_timescale(
    observations: pd.DataFrame,
    splits: pd.DataFrame,
    max_lag: int,
) -> tuple[float, int, np.ndarray]:
    """Estimate a pooled training-only velocity autocorrelation timescale.

    Velocity components are demeaned within each continuous segment. The
    vector autocorrelation is pooled across eligible training segments and
    integrated to the first non-positive lag.
    """
    training_ids = set(
        splits.loc[splits["partition"] == "training", "ID"].tolist()
    )
    train = observations.loc[observations["ID"].isin(training_ids)].copy()
    eligible_keys = (
        train.groupby(["ID", "segment_id"])["time"]
        .size()
        .loc[lambda s: s >= WINDOW_STEPS]
        .index
    )
    if len(eligible_keys) == 0:
        raise ValueError("No eligible training segments are available for ACF")

    numerators = np.zeros(max_lag + 1, dtype=float)
    left_energy = np.zeros(max_lag + 1, dtype=float)
    right_energy = np.zeros(max_lag + 1, dtype=float)

    for key, group in train.groupby(["ID", "segment_id"], sort=False):
        if key not in eligible_keys:
            continue
        velocity = group[["ve", "vn"]].to_numpy(dtype=float)
        velocity = velocity - np.nanmean(velocity, axis=0, keepdims=True)
        if not np.isfinite(velocity).all():
            continue
        local_max = min(max_lag, len(velocity) - 1)
        for lag in range(local_max + 1):
            if lag == 0:
                left = right = velocity
            else:
                left = velocity[:-lag]
                right = velocity[lag:]
            numerators[lag] += np.sum(left * right)
            left_energy[lag] += np.sum(left * left)
            right_energy[lag] += np.sum(right * right)

    denominator = np.sqrt(left_energy * right_energy)
    rho = np.divide(
        numerators,
        denominator,
        out=np.full_like(numerators, np.nan),
        where=denominator > 0,
    )
    rho[0] = 1.0

    non_positive = np.flatnonzero(rho[1:] <= 0)
    cutoff = int(non_positive[0] + 1) if len(non_positive) else max_lag
    integral_timescale = float(
        np.trapezoid(rho[: cutoff + 1], dx=1.0)
    )

    below_e = np.flatnonzero(rho[1:] <= np.exp(-1))
    decorrelation_lag = int(below_e[0] + 1) if len(below_e) else max_lag
    return integral_timescale, decorrelation_lag, rho


def build_forecast_origins(
    observations: pd.DataFrame,
    segments: pd.DataFrame,
    splits: pd.DataFrame,
    spacing_hours: int,
) -> pd.DataFrame:
    """Create non-overlapping 24-input/72-target forecast-case indices."""
    eligible_keys = set(
        map(
            tuple,
            segments.loc[
                segments["eligible_24_to_72"], ["ID", "segment_id"]
            ].to_numpy(),
        )
    )
    partition_map = splits.set_index("ID")["partition"].to_dict()
    records: list[dict[str, object]] = []
    case_number = 0

    for key, group in observations.groupby(["ID", "segment_id"], sort=False):
        if key not in eligible_keys:
            continue
        group = group.sort_values("time").reset_index(drop=True)
        max_start = len(group) - WINDOW_STEPS
        for history_start_step in range(0, max_start + 1, spacing_hours):
            origin_step = history_start_step + INPUT_STEPS - 1
            target_start_step = origin_step + 1
            target_end_step = history_start_step + WINDOW_STEPS - 1
            window = group.iloc[history_start_step : target_end_step + 1]
            origin = group.iloc[origin_step]
            case_number += 1
            records.append(
                {
                    "forecast_case_id": case_number,
                    "ID": int(key[0]),
                    "segment_id": int(key[1]),
                    "partition": partition_map[int(key[0])],
                    "history_start_step": int(history_start_step),
                    "origin_step": int(origin_step),
                    "target_start_step": int(target_start_step),
                    "target_end_step": int(target_end_step),
                    "history_start_time": group.iloc[history_start_step]["time"],
                    "origin_time": origin["time"],
                    "target_start_time": group.iloc[target_start_step]["time"],
                    "target_end_time": group.iloc[target_end_step]["time"],
                    "origin_latitude": float(origin["latitude"]),
                    "origin_longitude": float(origin["longitude"]),
                    "origin_ve": float(origin["ve"]),
                    "origin_vn": float(origin["vn"]),
                    "location_type_at_origin": origin.get("location_type", pd.NA),
                    "drogue_status_at_origin": origin["drogue_status"],
                    "n_high_speed_flags_in_window": int(
                        window["flag_high_speed"].sum()
                    ),
                    "n_extreme_speed_flags_in_window": int(
                        window["flag_extreme_speed"].sum()
                    ),
                    "n_land_mask_flags_in_window": int(
                        window["flag_land_mask"].sum()
                    ),
                }
            )

    origins = pd.DataFrame.from_records(records)
    if origins.empty:
        raise ValueError("No forecast origins were created")
    key_columns = ["ID", "segment_id", "origin_time"]
    if origins.duplicated(key_columns).any():
        raise AssertionError("Forecast-origin composite key is not unique")
    return origins


def build_qc_summary(
    observations: pd.DataFrame,
    segments: pd.DataFrame,
    splits: pd.DataFrame,
    origins: pd.DataFrame,
    base_metrics: dict[str, int | float | str],
    lag_1_autocorrelation: float,
    integral_timescale: float,
    decorrelation_lag: int,
    spacing_hours: int,
) -> pd.DataFrame:
    """Create a compact audit trail for cleaning and sample construction."""
    rows: list[dict[str, object]] = []

    actions = {
        "exact_duplicate_rows_removed": "Remove exact duplicates",
        "duplicate_ID_time_rows_removed": "Keep first record after audit",
        "unusable_core_or_coordinate_rows_removed": "Exclude from forecasting",
        "speed_above_2_m_per_s": "Retain and review",
        "speed_above_3_m_per_s": "Retain and review",
        "land_mask_flagged_observations": "Retain pending coastal review",
        "observations_missing_position_error": "Retain with missingness flag",
        "observations_missing_velocity_error": "Retain with missingness flag",
    }
    for metric, value in base_metrics.items():
        rows.append(
            {
                "category": "quality_control",
                "metric": metric,
                "value": value,
                "action_or_interpretation": actions.get(metric, "Report"),
            }
        )

    derived = {
        "continuous_segments": len(segments),
        "eligible_segments_at_least_96_observations": int(
            segments["eligible_24_to_72"].sum()
        ),
        "potential_overlapping_windows": int(
            segments["potential_overlapping_windows"].sum()
        ),
        "training_drifters": int((splits["partition"] == "training").sum()),
        "calibration_drifters": int(
            (splits["partition"] == "calibration").sum()
        ),
        "test_drifters": int((splits["partition"] == "test").sum()),
        "lag_1_velocity_autocorrelation": round(lag_1_autocorrelation, 6),
        "lag_at_or_below_one_over_e_hours": decorrelation_lag,
        "lagrangian_integral_timescale_hours": round(integral_timescale, 3),
        "selected_forecast_origin_spacing_hours": spacing_hours,
        "selected_forecast_origins": len(origins),
        "training_forecast_origins": int(
            (origins["partition"] == "training").sum()
        ),
        "calibration_forecast_origins": int(
            (origins["partition"] == "calibration").sum()
        ),
        "test_forecast_origins": int(
            (origins["partition"] == "test").sum()
        ),
    }
    for metric, value in derived.items():
        rows.append(
            {
                "category": "sample_construction",
                "metric": metric,
                "value": value,
                "action_or_interpretation": "Report",
            }
        )
    return pd.DataFrame(rows)


def validate_outputs(
    observations: pd.DataFrame,
    segments: pd.DataFrame,
    splits: pd.DataFrame,
    origins: pd.DataFrame,
) -> None:
    """Enforce the intended primary/foreign-key relationships."""
    if observations.duplicated(["ID", "time"]).any():
        raise AssertionError("Observations key (ID, time) is not unique")
    if segments.duplicated(["ID", "segment_id"]).any():
        raise AssertionError("Segments key (ID, segment_id) is not unique")
    if splits["ID"].duplicated().any():
        raise AssertionError("Splits key ID is not unique")
    if origins.duplicated(["ID", "segment_id", "origin_time"]).any():
        raise AssertionError("Forecast-origin composite key is not unique")

    segment_keys = pd.MultiIndex.from_frame(segments[["ID", "segment_id"]])
    observation_keys = pd.MultiIndex.from_frame(
        observations[["ID", "segment_id"]].drop_duplicates()
    )
    origin_keys = pd.MultiIndex.from_frame(
        origins[["ID", "segment_id"]].drop_duplicates()
    )
    if not observation_keys.isin(segment_keys).all():
        raise AssertionError("An observation references a missing segment")
    if not origin_keys.isin(segment_keys).all():
        raise AssertionError("A forecast origin references a missing segment")
    if not set(observations["ID"]).issubset(set(splits["ID"])):
        raise AssertionError("An observation references a missing split ID")
    if not set(origins["ID"]).issubset(set(splits["ID"])):
        raise AssertionError("A forecast origin references a missing split ID")

    origin_partition = origins[["ID", "partition"]].drop_duplicates()
    expected_partition = splits[["ID", "partition"]]
    checked = origin_partition.merge(
        expected_partition,
        on="ID",
        how="left",
        suffixes=("_origin", "_split"),
        validate="one_to_one",
    )
    if not (
        checked["partition_origin"] == checked["partition_split"]
    ).all():
        raise AssertionError("Origin and drifter split partitions disagree")


def main() -> None:
    args = parse_args()
    data_path = args.data.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading:", data_path)
    raw = read_erddap_csv(data_path)
    observations, base_metrics = clean_and_flag_observations(
        raw, use_land_mask=not args.skip_land_mask
    )
    observations, segments = add_continuous_segments(observations)
    splits = make_drifter_splits(observations, segments, seed=args.seed)

    print("Estimating training-only velocity autocorrelation...")
    integral_timescale, decorrelation_lag, rho = (
        estimate_lagrangian_timescale(
            observations,
            splits,
            max_lag=args.max_acf_lag,
        )
    )
    automatic_spacing = max(WINDOW_STEPS, math.ceil(integral_timescale))
    spacing_hours = (
        args.origin_spacing_hours
        if args.origin_spacing_hours is not None
        else automatic_spacing
    )
    if spacing_hours < WINDOW_STEPS:
        print(
            "Warning: origin spacing is below 96 hours, so complete forecast "
            "windows may overlap."
        )

    origins = build_forecast_origins(
        observations,
        segments,
        splits,
        spacing_hours=spacing_hours,
    )
    qc_summary = build_qc_summary(
        observations,
        segments,
        splits,
        origins,
        base_metrics,
        float(rho[1]),
        integral_timescale,
        decorrelation_lag,
        spacing_hours,
    )
    validate_outputs(observations, segments, splits, origins)

    observation_path = output_dir / "region_A_observations.parquet"
    segment_path = output_dir / "region_A_segments.parquet"
    split_path = output_dir / "drifter_splits.csv"
    origin_path = output_dir / "forecast_origins.parquet"
    qc_path = output_dir / "qc_summary.csv"

    observations.to_parquet(observation_path, index=False)
    segments.to_parquet(segment_path, index=False)
    splits.to_csv(split_path, index=False)
    origins.to_parquet(origin_path, index=False)
    qc_summary.to_csv(qc_path, index=False)

    print("\nProcessing completed successfully.")
    print(f"Observations: {len(observations):,}")
    print(f"Segments: {len(segments):,}")
    print(
        "Eligible segments:",
        f"{int(segments['eligible_24_to_72'].sum()):,}",
    )
    print(
        "Estimated Lagrangian integral timescale:",
        f"{integral_timescale:.2f} hours",
    )
    print("Selected origin spacing:", f"{spacing_hours} hours")
    print("Forecast origins:", f"{len(origins):,}")
    print("\nWritten files:")
    for path in [
        observation_path,
        segment_path,
        split_path,
        origin_path,
        qc_path,
    ]:
        print(" -", path)


if __name__ == "__main__":
    main()
