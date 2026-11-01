"""RQ2: six global models evaluated by regime at every lead time.

This script reconstructs the predictions of the six fitted RQ1 methods on the
same 757 held-out cases, joins the RQ2 regime at the forecast origin, and saves
case-level hourly errors plus median/RMSE growth curves for the four regimes.
It does not retrain models by regime; that is a separate extension experiment.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RQ2 = ROOT / "RQ2"
RQ1_OUTPUT = Path("/Users/roger/Documents/data3001-project-main/output/rq1/en")
OUT = RQ2 / "data" / "regional_lead_time"
FIG = RQ2 / "figures" / "regional_lead_time"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "rq2_extension", RQ2 / "18_regime_conditioned_forecasting.py"
)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

H = 72
EARTH_RADIUS_M = 6_371_000.0
REGIMES = ["Shelf", "Kuroshio Core", "Transition", "Other Offshore"]
MODELS = ["Persistence", "Mean Flow", "VAR(1)", "Ridge", "KNN", "FNN"]
COLORS = {
    "Persistence": "#636B72", "Mean Flow": "#9AA0A6", "VAR(1)": "#E68600",
    "Ridge": "#3267E3", "KNN": "#1B9E77", "FNN": "#D81B60",
}
LINESTYLES = {"Persistence": (0, (7, 5)), "Mean Flow": (0, (2, 3))}


def error_frame(model, test_cases, targets, pred_lat, pred_lon):
    truth = (
        targets[targets["forecast_case_id"].isin(test_cases["forecast_case_id"])]
        .sort_values(["forecast_case_id", "lead_time_hours"])
        .reset_index(drop=True)
    )
    error = ext.haversine_km(
        pred_lat, pred_lon, truth["latitude"].to_numpy(), truth["longitude"].to_numpy()
    )
    return pd.DataFrame({
        "model": model,
        "forecast_case_id": np.repeat(test_cases["forecast_case_id"].to_numpy(), H),
        "ID": np.repeat(test_cases["ID"].to_numpy(), H),
        "regime": np.repeat(test_cases["regime"].astype(str).to_numpy(), H),
        "lead_time_hours": np.tile(np.arange(1, H + 1), len(test_cases)),
        "separation_error_km": error,
    })


def displacement_error_frame(model, prediction, test_cases, targets):
    pred = prediction.reshape(len(test_cases), H, 2)
    origin_lat = np.repeat(test_cases["origin_latitude"].to_numpy(), H)
    origin_lon = np.repeat(test_cases["origin_longitude"].to_numpy(), H)
    pred_lat, pred_lon = ext.local_km_to_latlon(
        origin_lat, origin_lon, pred[:, :, 0].reshape(-1), pred[:, :, 1].reshape(-1)
    )
    return error_frame(model, test_cases, targets, pred_lat, pred_lon)


def predict_persistence(test_cases):
    expanded = test_cases.loc[test_cases.index.repeat(H)].reset_index(drop=True)
    lead = np.tile(np.arange(1, H + 1), len(test_cases))
    lat0 = np.deg2rad(expanded["origin_latitude"].to_numpy())
    lon0 = np.deg2rad(expanded["origin_longitude"].to_numpy())
    ve = expanded["origin_ve"].to_numpy()
    vn = expanded["origin_vn"].to_numpy()
    pred_lat = lat0 + vn * 3600 * lead / EARTH_RADIUS_M
    lat_before = lat0 + vn * 3600 * (lead - 1) / EARTH_RADIUS_M
    delta_lon = ve * 3600 / (EARTH_RADIUS_M * np.cos(lat_before))
    cumulative_lon = pd.Series(delta_lon).groupby(expanded["forecast_case_id"]).cumsum().to_numpy()
    pred_lon = lon0 + cumulative_lon
    return np.rad2deg(pred_lat), (np.rad2deg(pred_lon) + 180) % 360 - 180


def build_mean_flow():
    observations = pd.read_parquet(
        ROOT / "data" / "processed" / "region_A_observations.parquet",
        columns=["ID", "latitude", "longitude", "ve", "vn"],
    )
    splits = pd.read_csv(ROOT / "data" / "processed" / "drifter_splits.csv")
    training_ids = set(splits.loc[splits["partition"].eq("training"), "ID"])
    flow = observations[observations["ID"].isin(training_ids)].dropna().copy()

    def grid(frame, size):
        frame = frame.copy()
        frame["gx"] = np.floor((frame["longitude"] - 125) / size).astype(int)
        frame["gy"] = np.floor((frame["latitude"] - 25) / size).astype(int)
        result = frame.groupby(["gx", "gy"]).agg(
            mean_ve=("ve", "mean"), mean_vn=("vn", "mean"), n=("ve", "size")
        ).reset_index()
        return result[result["n"].ge(24)]

    fine, coarse = grid(flow, .25), grid(flow, 1.0)
    fine_lookup = {(int(r.gx), int(r.gy)): (r.mean_ve, r.mean_vn) for r in fine.itertuples()}
    coarse_lookup = {(int(r.gx), int(r.gy)): (r.mean_ve, r.mean_vn) for r in coarse.itertuples()}
    regional = (float(flow["ve"].mean()), float(flow["vn"].mean()))
    return fine_lookup, coarse_lookup, regional


def predict_mean_flow(test_cases):
    fine, coarse, regional = build_mean_flow()
    lat = test_cases["origin_latitude"].to_numpy(float).copy()
    lon = test_cases["origin_longitude"].to_numpy(float).copy()
    all_lat, all_lon = [], []
    for _ in range(H):
        ve, vn = np.empty(len(lat)), np.empty(len(lat))
        for i, (la, lo) in enumerate(zip(lat, lon)):
            key_f = (int(np.floor((lo - 125) / .25)), int(np.floor((la - 25) / .25)))
            key_c = (int(np.floor(lo - 125)), int(np.floor(la - 25)))
            ve[i], vn[i] = fine.get(key_f, coarse.get(key_c, regional))
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        lat = np.rad2deg(lat_r + vn * 3600 / EARTH_RADIUS_M)
        lon = (np.rad2deg(lon_r + ve * 3600 / (EARTH_RADIUS_M * np.cos(lat_r))) + 180) % 360 - 180
        all_lat.append(lat.copy()); all_lon.append(lon.copy())
    return np.column_stack(all_lat).reshape(-1), np.column_stack(all_lon).reshape(-1)


def predict_ridge(x_test):
    z = np.load(RQ1_OUTPUT / "ridge_model_parameters.npz")
    return ((x_test - z["x_mean"]) / z["x_scale"]) @ z["coefficients"] + z["y_mean"]


def predict_fnn(x_test):
    z = np.load(RQ1_OUTPUT / "fnn_model_parameters.npz")
    current = (x_test - z["x_mean"]) / z["x_scale"]
    current = np.maximum(current @ z["W1"] + z["b1"], 0)
    current = np.maximum(current @ z["W2"] + z["b2"], 0)
    return (current @ z["W3"] + z["b3"]) * z["y_scale"] + z["y_mean"]


def predict_knn(x, y, train_indices, test_indices):
    z = np.load(RQ1_OUTPUT / "knn_model_parameters.npz")
    stored_train = z["training_indices"].astype(int)
    if not np.array_equal(stored_train, train_indices):
        raise ValueError("Stored KNN training indices do not match the current dataset")
    xtr = (x[train_indices] - z["x_mean"]) / z["x_scale"]
    xte = (x[test_indices] - z["x_mean"]) / z["x_scale"]
    k, power = int(z["k"]), float(z["distance_power"])
    train_sq = np.sum(xtr ** 2, axis=1)
    predictions = []
    for start in range(0, len(xte), 256):
        q = xte[start:start + 256]
        sq = (np.sum(q ** 2, axis=1, keepdims=True) + train_sq - 2 * q @ xtr.T) / xtr.shape[1]
        sq = np.maximum(sq, 0)
        idx = np.argpartition(sq, k - 1, axis=1)[:, :k]
        dist = np.sqrt(np.take_along_axis(sq, idx, axis=1))
        weights = 1 / (dist + 1e-8) ** power
        weights /= weights.sum(axis=1, keepdims=True)
        predictions.append(np.einsum("nk,nko->no", weights, y[train_indices][idx], optimize=True))
    return np.vstack(predictions)


def predict_var(test_cases):
    coefficients = np.load(RQ1_OUTPUT / "var1_model_parameters.npz")["coefficients"]
    velocity = test_cases[["origin_ve", "origin_vn"]].to_numpy(float)
    cumulative = np.zeros_like(velocity)
    output = np.zeros((len(test_cases), H, 2))
    for lead in range(H):
        velocity = np.column_stack([np.ones(len(velocity)), velocity]) @ coefficients
        cumulative += velocity * 3.6
        output[:, lead] = cumulative
    return output.reshape(len(test_cases), H * 2)


def summarise(errors):
    return errors.groupby(["regime", "model", "lead_time_hours"], as_index=False).agg(
        n_cases=("forecast_case_id", "nunique"),
        n_drifters=("ID", "nunique"),
        median_error_km=("separation_error_km", "median"),
        rmse_km=("separation_error_km", lambda x: np.sqrt(np.mean(np.square(x)))),
        q25_error_km=("separation_error_km", lambda x: x.quantile(.25)),
        q75_error_km=("separation_error_km", lambda x: x.quantile(.75)),
    )


def plot_panels(summary, metric, ylabel, filename):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for ax, regime in zip(axes.flat, REGIMES):
        panel = summary[summary["regime"].eq(regime)]
        for model in MODELS:
            line = panel[panel["model"].eq(model)].sort_values("lead_time_hours")
            ax.plot(line["lead_time_hours"], line[metric], label=model,
                    color=COLORS[model], linestyle=LINESTYLES.get(model, "-"),
                    linewidth=2.4 if model in ["FNN", "KNN"] else 2.0)
        ax.set_title(regime, fontweight="bold")
        ax.grid(alpha=.2); ax.set_xlim(1, 72)
        ax.set_xticks([1, 12, 24, 36, 48, 60, 72])
    axes[1, 0].set_xlabel("Lead time (h)"); axes[1, 1].set_xlabel("Lead time (h)")
    axes[0, 0].set_ylabel(ylabel); axes[1, 0].set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(.5, .99))
    fig.suptitle(f"RQ2: {ylabel} versus lead time by ocean regime", y=1.04,
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(FIG / filename, dpi=220, bbox_inches="tight")
    fig.savefig(FIG / filename.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    data, targets, x, _, y, _ = ext.build_xy()
    train_indices = np.flatnonzero(data["partition"].eq("training"))
    test_indices = np.flatnonzero(data["partition"].eq("test"))
    test_cases = data.loc[test_indices].reset_index(drop=True)
    frames = []

    lat, lon = predict_persistence(test_cases)
    frames.append(error_frame("Persistence", test_cases, targets, lat, lon))
    lat, lon = predict_mean_flow(test_cases)
    frames.append(error_frame("Mean Flow", test_cases, targets, lat, lon))
    frames.append(displacement_error_frame("VAR(1)", predict_var(test_cases), test_cases, targets))
    frames.append(displacement_error_frame("Ridge", predict_ridge(x[test_indices]), test_cases, targets))
    frames.append(displacement_error_frame("KNN", predict_knn(x, y, train_indices, test_indices), test_cases, targets))
    frames.append(displacement_error_frame("FNN", predict_fnn(x[test_indices]), test_cases, targets))

    errors = pd.concat(frames, ignore_index=True)
    summary = summarise(errors)
    checkpoints = summary[summary["lead_time_hours"].isin([24, 48, 72])].copy()
    errors.to_csv(OUT / "six_model_regional_lead_errors.csv", index=False)
    summary.to_csv(OUT / "six_model_regional_lead_summary.csv", index=False)
    checkpoints.to_csv(OUT / "six_model_regional_24_48_72.csv", index=False)

    # Confirm the reconstructed 72-h metrics reproduce the existing RQ2 table.
    existing = pd.read_csv(RQ2 / "data" / "rq2_regime_model_summary.csv")
    current = summary[summary["lead_time_hours"].eq(72)].rename(columns={
        "median_error_km": "median_rebuilt", "rmse_km": "rmse_rebuilt"
    })
    check = current.merge(existing, on=["regime", "model"], validate="one_to_one")
    check["median_abs_difference_km"] = (check["median_rebuilt"] - check["median_72h_km"]).abs()
    check["rmse_abs_difference_km"] = (check["rmse_rebuilt"] - check["rmse_72h_km"]).abs()
    check.to_csv(OUT / "reconstruction_validation.csv", index=False)
    if check[["median_abs_difference_km", "rmse_abs_difference_km"]].to_numpy().max() > .05:
        raise ValueError("Reconstructed predictions do not reproduce the saved RQ2 metrics")

    plot_panels(summary, "median_error_km", "Median separation error (km)",
                "regional_median_error_vs_lead_time.png")
    plot_panels(summary, "rmse_km", "RMSE (km)", "regional_rmse_vs_lead_time.png")
    print("Saved:", OUT)
    print("Figures:", FIG)
    print("Maximum reconstruction difference (km):",
          check[["median_abs_difference_km", "rmse_abs_difference_km"]].to_numpy().max())


if __name__ == "__main__":
    main()
