"""Retrain every learned RQ1 model using Kuroshio-Core training data only.

The global train/calibration/test drifter split is kept fixed.  Only cases
whose forecast origin is labelled ``Kuroshio Core`` are used for fitting and
the same 139 held-out Kuroshio test cases are used for every comparison.

Learned models: multi-output Ridge, FNN, weighted KNN, pooled VAR(1).
Velocity persistence and the existing spatial mean-flow model are reported as
references; persistence has no fitting stage, while the poster mean-flow
baseline was fitted globally and is therefore not relabelled as regional.
"""

from __future__ import annotations

import copy
import importlib.util
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RQ2 = ROOT / "RQ2"
OUT = RQ2 / "data" / "kuroshio_all_models"
FIG = RQ2 / "figures" / "kuroshio_all_models"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "rq2_extension", RQ2 / "18_regime_conditioned_forecasting.py"
)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

REGIME = "Kuroshio Core"
H = 72
SEED = 3001
K_VALUES = [5, 10, 20, 50, 100]
POWERS = [1.0, 2.0]
FNN_CONFIGS = [
    {"name": "64", "hidden_sizes": (64,), "l2": 1e-4},
    {"name": "128-64", "hidden_sizes": (128, 64), "l2": 1e-4},
    {"name": "128-64 stronger L2", "hidden_sizes": (128, 64), "l2": 1e-3},
]
MAX_EPOCHS = 180
PATIENCE = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3


def standardise(train, query):
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return (train - mean) / scale, (query - mean) / scale, mean, scale


def nearest(x_train, x_query, max_k):
    train_sq = np.sum(x_train * x_train, axis=1)
    out_i, out_d = [], []
    for start in range(0, len(x_query), 256):
        q = x_query[start:start + 256]
        sq = (np.sum(q * q, axis=1, keepdims=True) + train_sq[None, :]
              - 2 * q @ x_train.T) / x_train.shape[1]
        sq = np.maximum(sq, 0)
        idx = np.argpartition(sq, kth=max_k - 1, axis=1)[:, :max_k]
        dist = np.take_along_axis(sq, idx, axis=1)
        order = np.argsort(dist, axis=1)
        out_i.append(np.take_along_axis(idx, order, axis=1))
        out_d.append(np.sqrt(np.take_along_axis(dist, order, axis=1)))
    return np.vstack(out_i), np.vstack(out_d)


def knn_predict(y_train, indices, distances, k, power):
    i = indices[:, :k]
    d = distances[:, :k]
    w = 1 / np.power(d + 1e-8, power)
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nko->no", w, y_train[i], optimize=True)


def select_knn(x, y, indices, groups):
    rows = []
    folds = ext.balanced_group_folds(indices, groups)
    max_k = min(max(K_VALUES), min(
        np.sum(~np.isin(groups[indices], validation_groups))
        for validation_groups in folds
    ))
    candidates = [k for k in K_VALUES if k <= max_k]
    for fold, validation_groups in enumerate(folds, 1):
        valid_mask = np.isin(groups[indices], validation_groups)
        va, tr = indices[valid_mask], indices[~valid_mask]
        xtr, xva, _, _ = standardise(x[tr], x[va])
        nn_i, nn_d = nearest(xtr, xva, max(candidates))
        for p in POWERS:
            for k in candidates:
                pred = knn_predict(y[tr], nn_i, nn_d, k, p)
                rows.append({
                    "fold": fold, "k": k, "distance_power": p,
                    "validation_rmse_km": ext.displacement_rmse(pred, y[va]),
                })
    folds_df = pd.DataFrame(rows)
    summary = folds_df.groupby(["k", "distance_power"], as_index=False).agg(
        mean_cv_rmse_km=("validation_rmse_km", "mean"),
        std_cv_rmse_km=("validation_rmse_km", "std"),
    )
    best = summary.loc[summary["mean_cv_rmse_km"].idxmin()]
    return int(best["k"]), float(best["distance_power"]), folds_df, summary


def relu(z):
    return np.maximum(z, 0)


def initialise(layer_sizes, rng):
    return [
        {"W": rng.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out)),
         "b": np.zeros(fan_out)}
        for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def forward(parameters, x):
    activations, pre = [x], []
    current = x
    for number, layer in enumerate(parameters):
        z = current @ layer["W"] + layer["b"]
        pre.append(z)
        current = relu(z) if number < len(parameters) - 1 else z
        activations.append(current)
    return current, activations, pre


def loss_gradients(parameters, x, y, l2):
    pred, activations, pre = forward(parameters, x)
    residual = pred - y
    loss = np.mean(residual ** 2) + l2 * sum(
        np.sum(layer["W"] ** 2) for layer in parameters
    )
    delta = 2 * residual / residual.size
    gradients = [None] * len(parameters)
    for number in range(len(parameters) - 1, -1, -1):
        gradients[number] = {
            "W": activations[number].T @ delta + 2 * l2 * parameters[number]["W"],
            "b": delta.sum(axis=0),
        }
        if number:
            delta = (delta @ parameters[number]["W"].T) * (pre[number - 1] > 0)
    return loss, gradients


def train_fnn(x_train, y_train, hidden, l2, epochs, seed,
              x_valid=None, y_valid=None, patience=None):
    x_mean, x_scale = x_train.mean(0), x_train.std(0)
    x_scale[x_scale < 1e-12] = 1
    y_mean, y_scale = y_train.mean(0), y_train.std(0)
    y_scale[y_scale < 1e-12] = 1
    xtr, ytr = (x_train - x_mean) / x_scale, (y_train - y_mean) / y_scale
    xva = None if x_valid is None else (x_valid - x_mean) / x_scale
    rng = np.random.default_rng(seed)
    params = initialise((x_train.shape[1], *hidden, y_train.shape[1]), rng)
    m = [{"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])} for p in params]
    v = copy.deepcopy(m)
    step = 0
    best, best_params, best_epoch, stale = np.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(xtr))
        for start in range(0, len(xtr), BATCH_SIZE):
            batch = permutation[start:start + BATCH_SIZE]
            _, grads = loss_gradients(params, xtr[batch], ytr[batch], l2)
            step += 1
            for layer_no, (layer, grad) in enumerate(zip(params, grads)):
                for key in ("W", "b"):
                    m[layer_no][key] = .9 * m[layer_no][key] + .1 * grad[key]
                    v[layer_no][key] = .999 * v[layer_no][key] + .001 * grad[key] ** 2
                    mh = m[layer_no][key] / (1 - .9 ** step)
                    vh = v[layer_no][key] / (1 - .999 ** step)
                    layer[key] -= LEARNING_RATE * mh / (np.sqrt(vh) + 1e-8)
        if xva is not None:
            pred = forward(params, xva)[0] * y_scale + y_mean
            score = ext.displacement_rmse(pred, y_valid)
            if score < best - 1e-5:
                best, best_params, best_epoch, stale = score, copy.deepcopy(params), epoch, 0
            else:
                stale += 1
            if stale >= patience:
                break
    return {
        "parameters": best_params if xva is not None else params,
        "x_mean": x_mean, "x_scale": x_scale,
        "y_mean": y_mean, "y_scale": y_scale,
        "best_epoch": best_epoch if xva is not None else epochs,
        "best_score": best,
    }


def fnn_predict(model, x):
    z = (x - model["x_mean"]) / model["x_scale"]
    return forward(model["parameters"], z)[0] * model["y_scale"] + model["y_mean"]


def select_fnn(x, y, indices, groups):
    rows = []
    folds = ext.balanced_group_folds(indices, groups)
    for config_no, config in enumerate(FNN_CONFIGS):
        for fold, validation_groups in enumerate(folds, 1):
            valid_mask = np.isin(groups[indices], validation_groups)
            va, tr = indices[valid_mask], indices[~valid_mask]
            model = train_fnn(
                x[tr], y[tr], config["hidden_sizes"], config["l2"],
                MAX_EPOCHS, SEED + 100 * config_no + fold,
                x_valid=x[va], y_valid=y[va], patience=PATIENCE,
            )
            rows.append({
                "config": config["name"], "fold": fold,
                "hidden_sizes": str(config["hidden_sizes"]), "l2": config["l2"],
                "best_epoch": model["best_epoch"],
                "validation_rmse_km": model["best_score"],
            })
    folds_df = pd.DataFrame(rows)
    summary = folds_df.groupby(["config", "hidden_sizes", "l2"], as_index=False).agg(
        mean_cv_rmse_km=("validation_rmse_km", "mean"),
        std_cv_rmse_km=("validation_rmse_km", "std"),
        median_best_epoch=("best_epoch", "median"),
    )
    best = summary.loc[summary["mean_cv_rmse_km"].idxmin()]
    config = next(c for c in FNN_CONFIGS if c["name"] == best["config"])
    return config, int(round(best["median_best_epoch"])), folds_df, summary


def fit_var1(transitions):
    design = np.column_stack([np.ones(len(transitions)), transitions[["ve", "vn"]]])
    target = transitions[["next_ve", "next_vn"]].to_numpy()
    return np.linalg.lstsq(design, target, rcond=None)[0]


def predict_velocity(coefficients, velocity):
    return np.column_stack([np.ones(len(velocity)), velocity]) @ coefficients


def var_displacements(coefficients, initial_velocity):
    velocity = initial_velocity.copy()
    out = np.zeros((len(velocity), H, 2))
    cumulative = np.zeros_like(velocity)
    for lead in range(H):
        velocity = predict_velocity(coefficients, velocity)
        cumulative += velocity * 3.6
        out[:, lead] = cumulative
    return out.reshape(len(velocity), H * 2)


def kuroshio_history_transitions(data, indices):
    observations = pd.read_parquet(
        ROOT / "data" / "processed" / "region_A_observations.parquet",
        columns=["ID", "segment_id", "segment_step", "ve", "vn"],
    )
    cases = data.loc[indices, ["ID", "segment_id", "history_start_step"]]
    windows = cases.loc[cases.index.repeat(23)].reset_index(drop=True)
    windows["offset"] = np.tile(np.arange(23), len(cases))
    windows["segment_step"] = windows["history_start_step"] + windows["offset"]
    current = windows.merge(observations, on=["ID", "segment_id", "segment_step"],
                            how="left", validate="many_to_one")
    following = observations.rename(columns={
        "segment_step": "next_step", "ve": "next_ve", "vn": "next_vn"
    })
    current["next_step"] = current["segment_step"] + 1
    transitions = current.merge(
        following[["ID", "segment_id", "next_step", "next_ve", "next_vn"]],
        on=["ID", "segment_id", "next_step"], how="left", validate="many_to_one",
    )
    return transitions.dropna(subset=["ve", "vn", "next_ve", "next_vn"]).drop_duplicates(
        ["ID", "segment_id", "segment_step"]
    )


def evaluate(name, prediction, data, targets, test_indices, y):
    test_cases = data.loc[test_indices].reset_index(drop=True)
    truth = (
        targets[targets["forecast_case_id"].isin(test_cases["forecast_case_id"])]
        .sort_values(["forecast_case_id", "lead_time_hours"])
        .reset_index(drop=True)
    )
    pred = prediction.reshape(len(test_indices), H, 2)
    origin_lat = np.repeat(test_cases["origin_latitude"].to_numpy(), H)
    origin_lon = np.repeat(test_cases["origin_longitude"].to_numpy(), H)
    pred_lat, pred_lon = ext.local_km_to_latlon(
        origin_lat, origin_lon, pred[:, :, 0].reshape(-1), pred[:, :, 1].reshape(-1)
    )
    errors = ext.haversine_km(
        pred_lat, pred_lon, truth["latitude"].to_numpy(), truth["longitude"].to_numpy()
    )
    evaluation = pd.DataFrame({
        "model": name,
        "forecast_case_id": np.repeat(test_cases["forecast_case_id"].to_numpy(), H),
        "ID": np.repeat(test_cases["ID"].to_numpy(), H),
        "regime": REGIME,
        "lead_time_hours": np.tile(np.arange(1, H + 1), len(test_indices)),
        "separation_error_km": errors,
    })
    observed_final = y[test_indices].reshape(-1, H, 2)[:, -1]
    observed_distance = np.linalg.norm(observed_final, axis=1)
    distance_map = dict(zip(test_cases["forecast_case_id"], observed_distance))
    case_rows = []
    for case_id, group in evaluation.groupby("forecast_case_id", sort=False):
        group = group.sort_values("lead_time_hours")
        e = group["separation_error_km"].to_numpy()
        hours = group["lead_time_hours"].to_numpy()
        first = group.iloc[0]
        case_rows.append({
            "model": name, "forecast_case_id": case_id, "ID": first["ID"],
            "regime": REGIME, "final_72h_error_km": e[-1],
            "mean_trajectory_error_km": e.mean(),
            "ite_km_h": np.trapezoid(np.r_[0.0, e], x=np.r_[0, hours]),
            "normalised_final_error": e[-1] / max(distance_map[case_id], 1.0),
        })
    cases = pd.DataFrame(case_rows)
    row = {
        "model": name, "n_cases": cases["forecast_case_id"].nunique(),
        "n_drifters": cases["ID"].nunique(),
        "median_72h_error_km": cases["final_72h_error_km"].median(),
        "rmse_72h_km": np.sqrt(np.mean(cases["final_72h_error_km"] ** 2)),
        "median_ite_km_h": cases["ite_km_h"].median(),
        "median_normalised_error": cases["normalised_final_error"].median(),
    }
    return row, evaluation, cases


def main():
    start = time.time()
    data, targets, x, _, y, _ = ext.build_xy()
    groups = data["ID"].to_numpy()
    train = np.flatnonzero(data["partition"].eq("training") & data["regime"].eq(REGIME))
    test = np.flatnonzero(data["partition"].eq("test") & data["regime"].eq(REGIME))
    assert len(train) == 646 and len(test) == 139

    rows, eval_frames, case_frames, selected = [], [], [], []

    print("1/4 Ridge: grouped CV")
    alpha, ridge_folds, ridge_cv = ext.select_alpha(x, y, train, groups, label="Regional Ridge")
    ridge_model = ext.fit_ridge(x[train], y[train], alpha)
    ridge_pred = ext.predict_ridge(ridge_model, x[test])
    row, ev, ca = evaluate("Regional Ridge", ridge_pred, data, targets, test, y)
    rows.append(row); eval_frames.append(ev); case_frames.append(ca)
    selected.append({"model": "Regional Ridge", "hyperparameters": f"alpha={alpha:g}"})

    print("2/4 Weighted KNN: grouped CV")
    k, power, knn_folds, knn_cv = select_knn(x, y, train, groups)
    xtr, xte, _, _ = standardise(x[train], x[test])
    nn_i, nn_d = nearest(xtr, xte, k)
    knn_pred = knn_predict(y[train], nn_i, nn_d, k, power)
    row, ev, ca = evaluate("Regional Weighted KNN", knn_pred, data, targets, test, y)
    rows.append(row); eval_frames.append(ev); case_frames.append(ca)
    selected.append({"model": "Regional Weighted KNN", "hyperparameters": f"K={k}, p={power:g}"})

    print("3/4 FNN: grouped CV (this is the slowest step)")
    config, epochs, fnn_folds, fnn_cv = select_fnn(x, y, train, groups)
    fnn_model = train_fnn(x[train], y[train], config["hidden_sizes"], config["l2"],
                          epochs, SEED + 999)
    fnn_pred = fnn_predict(fnn_model, x[test])
    row, ev, ca = evaluate("Regional FNN", fnn_pred, data, targets, test, y)
    rows.append(row); eval_frames.append(ev); case_frames.append(ca)
    selected.append({
        "model": "Regional FNN",
        "hyperparameters": f"hidden={config['hidden_sizes']}, L2={config['l2']:g}, epochs={epochs}",
    })

    print("4/4 VAR(1): Kuroshio history-window transitions")
    transitions = kuroshio_history_transitions(data, train)
    coefficients = fit_var1(transitions)
    var_pred = var_displacements(coefficients, data.loc[test, ["origin_ve", "origin_vn"]].to_numpy())
    row, ev, ca = evaluate("Regional VAR(1)", var_pred, data, targets, test, y)
    rows.append(row); eval_frames.append(ev); case_frames.append(ca)
    selected.append({
        "model": "Regional VAR(1)",
        "hyperparameters": f"no tuned hyperparameter; {len(transitions)} unique one-hour transitions",
    })

    regional = pd.DataFrame(rows)
    keep = ["model", "n_cases", "n_drifters", "median_72h_error_km", "rmse_72h_km",
            "median_ite_km_h", "median_normalised_error"]
    regional = regional[keep]

    global_table = pd.read_csv(RQ2 / "data" / "rq2_regime_model_summary.csv")
    global_table = global_table[global_table["regime"].eq(REGIME)].copy()
    global_table = global_table.rename(columns={
        "model": "global_model", "median_72h_km": "global_median_72h_error_km",
        "rmse_72h_km": "global_rmse_72h_km",
    })
    mapping = {
        "Regional Ridge": "Ridge", "Regional Weighted KNN": "KNN",
        "Regional FNN": "FNN", "Regional VAR(1)": "VAR(1)",
    }
    regional["global_model"] = regional["model"].map(mapping)
    comparison = regional.merge(
        global_table[["global_model", "global_median_72h_error_km", "global_rmse_72h_km"]],
        on="global_model", how="left", validate="one_to_one",
    )
    comparison["rmse_change_km"] = comparison["rmse_72h_km"] - comparison["global_rmse_72h_km"]
    comparison["rmse_improvement_pct"] = 100 * (
        1 - comparison["rmse_72h_km"] / comparison["global_rmse_72h_km"]
    )
    comparison["median_improvement_pct"] = 100 * (
        1 - comparison["median_72h_error_km"] / comparison["global_median_72h_error_km"]
    )
    comparison = comparison.sort_values("rmse_72h_km")

    comparison.to_csv(OUT / "kuroshio_global_vs_regional_all_models.csv", index=False)
    learned_view = comparison[[
        "global_model", "global_rmse_72h_km", "rmse_72h_km",
        "rmse_improvement_pct", "global_median_72h_error_km",
        "median_72h_error_km",
    ]].rename(columns={
        "global_model": "model", "rmse_72h_km": "kuroshio_trained_rmse_72h_km",
        "median_72h_error_km": "kuroshio_trained_median_72h_error_km",
    })
    learned_view["regional_refit"] = "yes"
    learned_view["note"] = "Retrained using Kuroshio-Core training cases only"
    baseline_view = global_table[
        global_table["global_model"].isin(["Persistence", "Mean Flow"])
    ][["global_model", "global_rmse_72h_km", "global_median_72h_error_km"]].rename(
        columns={"global_model": "model"}
    )
    baseline_view["kuroshio_trained_rmse_72h_km"] = np.nan
    baseline_view["rmse_improvement_pct"] = np.nan
    baseline_view["kuroshio_trained_median_72h_error_km"] = np.nan
    baseline_view["regional_refit"] = "no"
    baseline_view["note"] = [
        "No trainable parameters; unchanged physical baseline"
        if model == "Persistence" else
        "Poster mean-flow baseline fitted globally; not relabelled as regional"
        for model in baseline_view["model"]
    ]
    all_models = pd.concat([learned_view, baseline_view], ignore_index=True)
    all_models.to_csv(OUT / "kuroshio_all_six_models_summary.csv", index=False)
    pd.DataFrame(selected).to_csv(OUT / "selected_hyperparameters.csv", index=False)
    pd.concat(eval_frames, ignore_index=True).to_csv(OUT / "regional_test_lead_errors.csv", index=False)
    pd.concat(case_frames, ignore_index=True).to_csv(OUT / "regional_test_case_metrics.csv", index=False)
    ridge_folds.to_csv(OUT / "ridge_cv_folds.csv", index=False)
    ridge_cv.to_csv(OUT / "ridge_cv_summary.csv", index=False)
    knn_folds.to_csv(OUT / "knn_cv_folds.csv", index=False)
    knn_cv.to_csv(OUT / "knn_cv_summary.csv", index=False)
    fnn_folds.to_csv(OUT / "fnn_cv_folds.csv", index=False)
    fnn_cv.to_csv(OUT / "fnn_cv_summary.csv", index=False)

    labels = [m.replace("Regional ", "") for m in comparison["model"]]
    xloc = np.arange(len(comparison))
    width = .35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    panels = [
        ("global_rmse_72h_km", "rmse_72h_km", "72-h endpoint RMSE (km)",
         "RMSE: sensitive to large failures"),
        ("global_median_72h_error_km", "median_72h_error_km",
         "72-h median endpoint error (km)", "Median: typical test case"),
    ]
    for ax, (global_col, regional_col, ylabel, title) in zip(axes, panels):
        ax.bar(xloc - width/2, comparison[global_col], width,
               label="Global training", color="#9AA0A6")
        ax.bar(xloc + width/2, comparison[regional_col], width,
               label="Kuroshio-only training", color="#159FD3")
        for i, value in enumerate(comparison[regional_col]):
            ax.text(i + width/2, value + 1.2, f"{value:.1f}",
                    ha="center", fontsize=9)
        ax.set_xticks(xloc, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Kuroshio Core: global versus Kuroshio-only training", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "kuroshio_global_vs_regional_all_models.png", dpi=220)
    plt.close(fig)

    print("\nRESULTS")
    print(comparison[["model", "global_rmse_72h_km", "rmse_72h_km",
                      "rmse_improvement_pct", "median_72h_error_km"]].round(3).to_string(index=False))
    print(f"\nCompleted in {time.time()-start:.1f} seconds")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
