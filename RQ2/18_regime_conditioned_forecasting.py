"""RQ2 extension: global, regional, current-conditioned and residual forecasts.

This experiment deliberately keeps the learning algorithm fixed (multi-output
Ridge) and changes only the modelling strategy.  That makes the comparison
answer two focused questions:

1. Does fitting one model per physical regime improve forecasts relative to a
   single global model?
2. Does supplying the local GLORYS climatological current, or predicting only
   the residual from that current, improve forecasts?

All hyperparameters are selected inside the training partition with grouped
five-fold cross-validation by drifter.  The calibration partition is unused
and the fixed test partition is evaluated once at the end.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RQ2_DATA = ROOT / "RQ2" / "data"
OUTPUT_DIR = RQ2_DATA / "rq2_extension"
FIGURE_DIR = ROOT / "RQ2" / "figures" / "rq2_extension"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6_371.0
HISTORY_HOURS = 24
HORIZON_HOURS = 72
N_FOLDS = 5
ALPHAS = np.array([0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0])
N_BOOT = 2_000
RNG_SEED = 3001

REGIMES = ["Shelf", "Kuroshio Core", "Transition", "Other Offshore"]
STRATEGIES = [
    "Global original",
    "Regional",
    "Current-conditioned",
    "Current-residual",
]
COLORS = {
    "Global original": "#6C757D",
    "Regional": "#E76F51",
    "Current-conditioned": "#2A9D8F",
    "Current-residual": "#457B9D",
}


def latlon_to_local_km(origin_lat, origin_lon, latitude, longitude):
    lat0 = np.deg2rad(np.asarray(origin_lat, dtype=float))
    lon0 = np.deg2rad(np.asarray(origin_lon, dtype=float))
    lat1 = np.deg2rad(np.asarray(latitude, dtype=float))
    lon1 = np.deg2rad(np.asarray(longitude, dtype=float))
    dlon = (lon1 - lon0 + np.pi) % (2 * np.pi) - np.pi
    dlat = lat1 - lat0
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    )
    central_angle = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    bearing = np.arctan2(
        np.sin(dlon) * np.cos(lat1),
        np.cos(lat0) * np.sin(lat1)
        - np.sin(lat0) * np.cos(lat1) * np.cos(dlon),
    )
    distance_km = EARTH_RADIUS_KM * central_angle
    return distance_km * np.sin(bearing), distance_km * np.cos(bearing)


def local_km_to_latlon(origin_lat, origin_lon, east_km, north_km):
    lat0 = np.deg2rad(np.asarray(origin_lat, dtype=float))
    lon0 = np.deg2rad(np.asarray(origin_lon, dtype=float))
    east_km = np.asarray(east_km, dtype=float)
    north_km = np.asarray(north_km, dtype=float)
    distance_km = np.hypot(east_km, north_km)
    bearing = np.arctan2(east_km, north_km)
    angular_distance = distance_km / EARTH_RADIUS_KM
    latitude = np.arcsin(
        np.sin(lat0) * np.cos(angular_distance)
        + np.cos(lat0) * np.sin(angular_distance) * np.cos(bearing)
    )
    longitude = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(angular_distance) * np.cos(lat0),
        np.cos(angular_distance) - np.sin(lat0) * np.sin(latitude),
    )
    return np.rad2deg(latitude), (np.rad2deg(longitude) + 180) % 360 - 180


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def fit_ridge(x_train, y_train, alpha):
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    y_mean = y_train.mean(axis=0)
    x_standard = (x_train - x_mean) / x_scale
    gram = x_standard.T @ x_standard
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(gram.shape[0]),
        x_standard.T @ (y_train - y_mean),
    )
    return {
        "alpha": float(alpha),
        "x_mean": x_mean,
        "x_scale": x_scale,
        "y_mean": y_mean,
        "coefficients": coefficients,
    }


def predict_ridge(model, x_new):
    return (
        (x_new - model["x_mean"]) / model["x_scale"]
    ) @ model["coefficients"] + model["y_mean"]


def balanced_group_folds(indices, groups, n_folds=N_FOLDS):
    counts = pd.Series(groups[indices]).value_counts().sort_values(ascending=False)
    if len(counts) < n_folds:
        raise ValueError(f"Only {len(counts)} drifters available for {n_folds} folds")
    fold_groups = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    for group, count in counts.items():
        fold = int(np.argmin(fold_sizes))
        fold_groups[fold].append(group)
        fold_sizes[fold] += int(count)
    return fold_groups


def displacement_rmse(y_pred, y_true):
    pred = y_pred.reshape(-1, HORIZON_HOURS, 2)
    true = y_true.reshape(-1, HORIZON_HOURS, 2)
    distances = np.linalg.norm(pred - true, axis=2)
    return float(np.sqrt(np.mean(np.square(distances))))


def select_alpha(x, y, indices, groups, baseline=None, label="model"):
    fold_groups = balanced_group_folds(indices, groups)
    rows = []
    for alpha in ALPHAS:
        for fold_number, validation_groups in enumerate(fold_groups, start=1):
            is_validation = np.isin(groups[indices], validation_groups)
            validation_indices = indices[is_validation]
            fitting_indices = indices[~is_validation]
            y_fit = y[fitting_indices]
            if baseline is not None:
                y_fit = y_fit - baseline[fitting_indices]
            model = fit_ridge(x[fitting_indices], y_fit, alpha)
            prediction = predict_ridge(model, x[validation_indices])
            if baseline is not None:
                prediction = prediction + baseline[validation_indices]
            rows.append({
                "model": label,
                "alpha": float(alpha),
                "fold": fold_number,
                "n_validation_cases": len(validation_indices),
                "validation_trajectory_rmse_km": displacement_rmse(
                    prediction, y[validation_indices]
                ),
            })
    fold_results = pd.DataFrame(rows)
    summary = (
        fold_results.groupby(["model", "alpha"], as_index=False)
        .agg(
            mean_cv_rmse_km=("validation_trajectory_rmse_km", "mean"),
            std_cv_rmse_km=("validation_trajectory_rmse_km", "std"),
        )
    )
    best_alpha = float(
        summary.loc[summary["mean_cv_rmse_km"].idxmin(), "alpha"]
    )
    return best_alpha, fold_results, summary


def build_xy():
    origins = (
        pd.read_parquet(PROCESSED / "forecast_origins.parquet")
        .sort_values("forecast_case_id")
        .reset_index(drop=True)
    )
    observations = pd.read_parquet(
        PROCESSED / "region_A_observations.parquet",
        columns=[
            "ID", "segment_id", "segment_step", "time", "latitude",
            "longitude", "ve", "vn",
        ],
    )
    n_cases = len(origins)

    history = origins[[
        "forecast_case_id", "ID", "segment_id", "history_start_step",
        "origin_latitude", "origin_longitude",
    ]].loc[origins.index.repeat(HISTORY_HOURS)].reset_index(drop=True)
    history["history_index"] = np.tile(np.arange(HISTORY_HOURS), n_cases)
    history["segment_step"] = history["history_start_step"] + history["history_index"]
    history = history.merge(
        observations,
        on=["ID", "segment_id", "segment_step"],
        how="left",
        validate="many_to_one",
    ).sort_values(["forecast_case_id", "history_index"])
    history["relative_east_km"], history["relative_north_km"] = latlon_to_local_km(
        history["origin_latitude"], history["origin_longitude"],
        history["latitude"], history["longitude"],
    )
    history_values = history[[
        "relative_east_km", "relative_north_km", "ve", "vn",
    ]].to_numpy().reshape(n_cases, HISTORY_HOURS, 4)
    x_original = np.column_stack([
        origins["origin_latitude"].to_numpy(),
        origins["origin_longitude"].to_numpy(),
        history_values.reshape(n_cases, -1),
    ])

    targets = origins[[
        "forecast_case_id", "ID", "segment_id", "target_start_step",
        "origin_latitude", "origin_longitude",
    ]].loc[origins.index.repeat(HORIZON_HOURS)].reset_index(drop=True)
    targets["lead_time_hours"] = np.tile(
        np.arange(1, HORIZON_HOURS + 1), n_cases
    )
    targets["segment_step"] = (
        targets["target_start_step"] + targets["lead_time_hours"] - 1
    )
    targets = targets.merge(
        observations[[
            "ID", "segment_id", "segment_step", "time", "latitude", "longitude",
        ]],
        on=["ID", "segment_id", "segment_step"],
        how="left",
        validate="many_to_one",
    ).sort_values(["forecast_case_id", "lead_time_hours"])
    targets["target_east_km"], targets["target_north_km"] = latlon_to_local_km(
        targets["origin_latitude"], targets["origin_longitude"],
        targets["latitude"], targets["longitude"],
    )
    y = targets[["target_east_km", "target_north_km"]].to_numpy().reshape(
        n_cases, HORIZON_HOURS * 2
    )

    regimes = pd.read_csv(RQ2_DATA / "forecast_case_regimes.csv")
    data = origins.merge(
        regimes[[
            "forecast_case_id", "regime", "depth_m", "u_clim", "v_clim",
            "current_speed_clim", "distance_to_kuroshio_km",
        ]],
        on="forecast_case_id",
        how="left",
        validate="one_to_one",
    )
    if not data["forecast_case_id"].equals(origins["forecast_case_id"]):
        raise ValueError("Case ordering changed while joining RQ2 features")

    # Only the local mean-current vector and magnitude are appended here.  This
    # isolates the hypothesis that knowing local climatological flow helps.
    current_features = data[["u_clim", "v_clim", "current_speed_clim"]].copy()
    training_mask = data["partition"].eq("training")
    for column in current_features:
        fill_value = current_features.loc[training_mask, column].median()
        current_features[column] = current_features[column].fillna(fill_value)
    x_conditioned = np.column_stack([x_original, current_features.to_numpy()])

    # A simple origin-current baseline: the climatological vector observed at
    # the forecast origin is integrated for 1..72 hours.  m/s * 3.6 = km/hour.
    lead_hours = np.arange(1, HORIZON_HOURS + 1, dtype=float)
    baseline = np.empty((n_cases, HORIZON_HOURS, 2), dtype=float)
    baseline[:, :, 0] = current_features["u_clim"].to_numpy()[:, None] * 3.6 * lead_hours
    baseline[:, :, 1] = current_features["v_clim"].to_numpy()[:, None] * 3.6 * lead_hours
    baseline = baseline.reshape(n_cases, HORIZON_HOURS * 2)

    checks = {
        "cases": n_cases == 5_117,
        "x_original": x_original.shape == (5_117, 98),
        "x_conditioned": x_conditioned.shape == (5_117, 101),
        "y": y.shape == (5_117, 144),
        "finite_x": np.isfinite(x_original).all() and np.isfinite(x_conditioned).all(),
        "finite_y": np.isfinite(y).all() and np.isfinite(baseline).all(),
    }
    if not all(checks.values()):
        raise ValueError(f"Data checks failed: {checks}")
    return data, targets.reset_index(drop=True), x_original, x_conditioned, y, baseline


def evaluate_predictions(data, targets, test_indices, prediction_map, y):
    test_cases = data.loc[test_indices].reset_index(drop=True)
    true_targets = (
        targets[targets["forecast_case_id"].isin(test_cases["forecast_case_id"])]
        .sort_values(["forecast_case_id", "lead_time_hours"])
        .reset_index(drop=True)
    )
    records = []
    for strategy, prediction in prediction_map.items():
        pred = prediction.reshape(len(test_indices), HORIZON_HOURS, 2)
        origin_lat = np.repeat(test_cases["origin_latitude"].to_numpy(), HORIZON_HOURS)
        origin_lon = np.repeat(test_cases["origin_longitude"].to_numpy(), HORIZON_HOURS)
        pred_lat, pred_lon = local_km_to_latlon(
            origin_lat,
            origin_lon,
            pred[:, :, 0].reshape(-1),
            pred[:, :, 1].reshape(-1),
        )
        error = haversine_km(
            pred_lat,
            pred_lon,
            true_targets["latitude"].to_numpy(),
            true_targets["longitude"].to_numpy(),
        )
        frame = pd.DataFrame({
            "strategy": strategy,
            "forecast_case_id": np.repeat(
                test_cases["forecast_case_id"].to_numpy(), HORIZON_HOURS
            ),
            "ID": np.repeat(test_cases["ID"].to_numpy(), HORIZON_HOURS),
            "regime": np.repeat(test_cases["regime"].astype(str).to_numpy(), HORIZON_HOURS),
            "lead_time_hours": np.tile(
                np.arange(1, HORIZON_HOURS + 1), len(test_indices)
            ),
            "separation_error_km": error,
        })
        records.append(frame)
    evaluation = pd.concat(records, ignore_index=True)

    final = evaluation[evaluation["lead_time_hours"].eq(72)].copy()
    observed_final = y[test_indices].reshape(-1, HORIZON_HOURS, 2)[:, -1, :]
    observed_displacement = np.linalg.norm(observed_final, axis=1)
    displacement_map = dict(zip(test_cases["forecast_case_id"], observed_displacement))
    final["observed_72h_displacement_km"] = final["forecast_case_id"].map(displacement_map)
    final["normalised_final_error"] = (
        final["separation_error_km"]
        / final["observed_72h_displacement_km"].clip(lower=1.0)
    )

    case_rows = []
    for (strategy, case_id), group in evaluation.groupby(
        ["strategy", "forecast_case_id"], sort=False
    ):
        group = group.sort_values("lead_time_hours")
        errors = group["separation_error_km"].to_numpy()
        hours = group["lead_time_hours"].to_numpy()
        row = group.iloc[0]
        case_rows.append({
            "strategy": strategy,
            "forecast_case_id": case_id,
            "ID": row["ID"],
            "regime": row["regime"],
            "final_72h_error_km": errors[-1],
            "mean_trajectory_error_km": errors.mean(),
            "ite_km_h": np.trapezoid(np.r_[0.0, errors], x=np.r_[0, hours]),
            "observed_72h_displacement_km": displacement_map[case_id],
            "normalised_final_error": errors[-1] / max(displacement_map[case_id], 1.0),
        })
    cases = pd.DataFrame(case_rows)

    summary = (
        cases.groupby(["regime", "strategy"], as_index=False)
        .agg(
            n_cases=("forecast_case_id", "nunique"),
            n_drifters=("ID", "nunique"),
            median_72h_error_km=("final_72h_error_km", "median"),
            rmse_72h_km=(
                "final_72h_error_km",
                lambda z: np.sqrt(np.mean(np.square(z))),
            ),
            median_ite_km_h=("ite_km_h", "median"),
            median_normalised_error=("normalised_final_error", "median"),
        )
    )
    baseline_rmse = (
        summary[summary["strategy"].eq("Global original")]
        .set_index("regime")["rmse_72h_km"]
    )
    summary["rmse_improvement_vs_global_pct"] = summary.apply(
        lambda row: 100 * (
            1 - row["rmse_72h_km"] / baseline_rmse.loc[row["regime"]]
        ),
        axis=1,
    )
    return evaluation, cases, summary


def clustered_bootstrap(cases):
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    global_cases = cases[cases["strategy"].eq("Global original")][[
        "forecast_case_id", "ID", "regime", "final_72h_error_km",
    ]].rename(columns={"final_72h_error_km": "global_error"})
    for regime in REGIMES:
        global_regime = global_cases[global_cases["regime"].eq(regime)]
        for strategy in STRATEGIES[1:]:
            candidate = cases[
                cases["strategy"].eq(strategy) & cases["regime"].eq(regime)
            ][["forecast_case_id", "final_72h_error_km"]].rename(
                columns={"final_72h_error_km": "candidate_error"}
            )
            paired = global_regime.merge(
                candidate, on="forecast_case_id", how="inner", validate="one_to_one"
            )
            drifters = paired["ID"].unique()
            boot = np.empty(N_BOOT)
            for b in range(N_BOOT):
                sampled = rng.choice(drifters, size=len(drifters), replace=True)
                chunks = [paired[paired["ID"].eq(d)] for d in sampled]
                sample = pd.concat(chunks, ignore_index=True)
                global_rmse = np.sqrt(np.mean(np.square(sample["global_error"])))
                candidate_rmse = np.sqrt(np.mean(np.square(sample["candidate_error"])))
                boot[b] = 100 * (1 - candidate_rmse / global_rmse)
            observed_global = np.sqrt(np.mean(np.square(paired["global_error"])))
            observed_candidate = np.sqrt(np.mean(np.square(paired["candidate_error"])))
            observed = 100 * (1 - observed_candidate / observed_global)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append({
                "regime": regime,
                "strategy": strategy,
                "n_cases": len(paired),
                "n_drifters": len(drifters),
                "rmse_improvement_vs_global_pct": observed,
                "ci_lower_pct": lo,
                "ci_upper_pct": hi,
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


def make_figures(summary, bootstrap):
    plot = summary[summary["regime"].isin(REGIMES)].copy()
    x = np.arange(len(REGIMES))
    width = 0.19
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strategy in enumerate(STRATEGIES):
        values = (
            plot[plot["strategy"].eq(strategy)]
            .set_index("regime")
            .reindex(REGIMES)["rmse_72h_km"]
        )
        ax.bar(
            x + (i - 1.5) * width,
            values,
            width,
            label=strategy,
            color=COLORS[strategy],
        )
    ax.set_xticks(x, REGIMES)
    ax.set_ylabel("72-h separation RMSE (km)")
    ax.set_title("Global, regional and climatological-current modelling strategies")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "strategy_rmse_by_regime.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(REGIMES), figsize=(15, 4.6), sharey=True)
    for ax, regime in zip(axes, REGIMES):
        part = bootstrap[bootstrap["regime"].eq(regime)].set_index("strategy")
        strategies = STRATEGIES[1:]
        y = np.arange(len(strategies))
        values = part.reindex(strategies)["rmse_improvement_vs_global_pct"].to_numpy()
        lo = part.reindex(strategies)["ci_lower_pct"].to_numpy()
        hi = part.reindex(strategies)["ci_upper_pct"].to_numpy()
        ax.errorbar(
            values,
            y,
            xerr=np.vstack([values - lo, hi - values]),
            fmt="o",
            capsize=4,
            color="#1D3557",
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(strategies)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(regime)
        ax.grid(axis="x", alpha=0.2)
    fig.supxlabel("72-h RMSE improvement over original global Ridge (%)")
    fig.suptitle("Positive values favour the alternative strategy; 95% drifter-clustered CIs")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "strategy_improvement_cluster_ci.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== RQ2 EXTENSION: REGIONAL AND CURRENT-CONDITIONED FORECASTS ===")
    data, targets, x_original, x_conditioned, y, current_baseline = build_xy()
    groups = data["ID"].to_numpy()
    train_indices = np.flatnonzero(data["partition"].eq("training").to_numpy())
    calibration_indices = np.flatnonzero(data["partition"].eq("calibration").to_numpy())
    test_indices = np.flatnonzero(data["partition"].eq("test").to_numpy())
    assert len(train_indices) == 3_591
    assert len(calibration_indices) == 769
    assert len(test_indices) == 757

    all_cv_folds = []
    all_cv_summaries = []
    selected_rows = []

    print("\nSelecting global original Ridge...")
    alpha_global, folds, cv = select_alpha(
        x_original, y, train_indices, groups, label="Global original"
    )
    all_cv_folds.append(folds)
    all_cv_summaries.append(cv)
    global_model = fit_ridge(
        x_original[train_indices], y[train_indices], alpha_global
    )
    global_prediction = predict_ridge(global_model, x_original[test_indices])
    selected_rows.append({
        "strategy": "Global original", "regime": "All",
        "selected_alpha": alpha_global, "n_training_cases": len(train_indices),
        "n_training_drifters": data.loc[train_indices, "ID"].nunique(),
    })

    print("Selecting current-conditioned global Ridge...")
    alpha_conditioned, folds, cv = select_alpha(
        x_conditioned, y, train_indices, groups, label="Current-conditioned"
    )
    all_cv_folds.append(folds)
    all_cv_summaries.append(cv)
    conditioned_model = fit_ridge(
        x_conditioned[train_indices], y[train_indices], alpha_conditioned
    )
    conditioned_prediction = predict_ridge(
        conditioned_model, x_conditioned[test_indices]
    )
    selected_rows.append({
        "strategy": "Current-conditioned", "regime": "All",
        "selected_alpha": alpha_conditioned, "n_training_cases": len(train_indices),
        "n_training_drifters": data.loc[train_indices, "ID"].nunique(),
    })

    print("Selecting current-residual global Ridge...")
    alpha_residual, folds, cv = select_alpha(
        x_conditioned,
        y,
        train_indices,
        groups,
        baseline=current_baseline,
        label="Current-residual",
    )
    all_cv_folds.append(folds)
    all_cv_summaries.append(cv)
    residual_model = fit_ridge(
        x_conditioned[train_indices],
        y[train_indices] - current_baseline[train_indices],
        alpha_residual,
    )
    residual_prediction = (
        predict_ridge(residual_model, x_conditioned[test_indices])
        + current_baseline[test_indices]
    )
    selected_rows.append({
        "strategy": "Current-residual", "regime": "All",
        "selected_alpha": alpha_residual, "n_training_cases": len(train_indices),
        "n_training_drifters": data.loc[train_indices, "ID"].nunique(),
    })

    print("Selecting one regional Ridge per regime...")
    regional_prediction = np.full_like(global_prediction, np.nan)
    for regime in REGIMES:
        region_train = np.flatnonzero(
            data["partition"].eq("training").to_numpy()
            & data["regime"].eq(regime).to_numpy()
        )
        region_test_positions = np.flatnonzero(
            data.loc[test_indices, "regime"].eq(regime).to_numpy()
        )
        region_test = test_indices[region_test_positions]
        alpha, folds, cv = select_alpha(
            x_original,
            y,
            region_train,
            groups,
            label=f"Regional: {regime}",
        )
        all_cv_folds.append(folds)
        all_cv_summaries.append(cv)
        model = fit_ridge(x_original[region_train], y[region_train], alpha)
        regional_prediction[region_test_positions] = predict_ridge(
            model, x_original[region_test]
        )
        selected_rows.append({
            "strategy": "Regional", "regime": regime,
            "selected_alpha": alpha, "n_training_cases": len(region_train),
            "n_training_drifters": data.loc[region_train, "ID"].nunique(),
        })
        print(
            f"  {regime}: alpha={alpha:g}, train={len(region_train)}, "
            f"test={len(region_test)}"
        )
    if not np.isfinite(regional_prediction).all():
        # The current test set has no unclassified cases.  This fallback keeps
        # the experiment safe if future labels contain an unseen category.
        missing = ~np.isfinite(regional_prediction).all(axis=1)
        regional_prediction[missing] = global_prediction[missing]

    prediction_map = {
        "Global original": global_prediction,
        "Regional": regional_prediction,
        "Current-conditioned": conditioned_prediction,
        "Current-residual": residual_prediction,
    }
    evaluation, cases, summary = evaluate_predictions(
        data, targets, test_indices, prediction_map, y
    )
    bootstrap = clustered_bootstrap(cases)

    pd.concat(all_cv_folds, ignore_index=True).to_csv(
        OUTPUT_DIR / "cv_fold_results.csv", index=False
    )
    pd.concat(all_cv_summaries, ignore_index=True).to_csv(
        OUTPUT_DIR / "cv_summary.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        OUTPUT_DIR / "selected_models.csv", index=False
    )
    evaluation.to_csv(OUTPUT_DIR / "test_lead_errors.csv", index=False)
    cases.to_csv(OUTPUT_DIR / "test_case_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "strategy_summary_by_regime.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "strategy_cluster_bootstrap.csv", index=False)
    make_figures(summary, bootstrap)

    print("\n=== 72-HOUR RESULTS ===")
    print(
        summary[[
            "regime", "strategy", "n_cases", "rmse_72h_km",
            "median_72h_error_km", "median_normalised_error",
            "rmse_improvement_vs_global_pct",
        ]].round(3).to_string(index=False)
    )
    print("\n=== CLUSTERED BOOTSTRAP ===")
    print(bootstrap.round(3).to_string(index=False))
    print("\nSaved data:", OUTPUT_DIR)
    print("Saved figures:", FIGURE_DIR)


if __name__ == "__main__":
    main()
