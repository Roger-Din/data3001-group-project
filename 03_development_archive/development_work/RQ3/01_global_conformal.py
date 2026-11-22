
# %%
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "data" / "processed").exists()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Cannot find data/processed. Run from the DATA3001 project root or a subdirectory.")

PROCESSED = PROJECT_ROOT / "data" / "processed"
RQ1_OUTPUT = PROJECT_ROOT / "output" / "rq1" / "en"
RQ2_DATA = PROJECT_ROOT / "RQ2" / "data"
OUTPUT_DIR = PROJECT_ROOT / "RQ3" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_KM = 6371.0
HISTORY_HOURS = 24
HORIZON_HOURS = 72
RANDOM_SEED = 3001
ALPHAS = [0.20, 0.10, 0.05]
MAIN_ALPHA = 0.10

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

print("Project root:", PROJECT_ROOT)
print("RQ3 output directory:", OUTPUT_DIR)

# |---|---:|---|---|---|
#

# %%
origins = (
    pd.read_parquet(PROCESSED / "forecast_origins.parquet")
    .sort_values("forecast_case_id")
    .reset_index(drop=True)
)
split_summary = (
    origins.groupby("partition", sort=False)
    .agg(usable_drifters=("ID", "nunique"), forecast_cases=("forecast_case_id", "size"))
    .reindex(["training", "calibration", "test"])
)
split_summary.loc["Total"] = split_summary.sum()
display(split_summary)

assert split_summary.loc["training", "forecast_cases"] == 3591
assert split_summary.loc["calibration", "forecast_cases"] == 769
assert split_summary.loc["test", "forecast_cases"] == 757

# $$s_{i,t}=d_{\mathrm{Haversine}}(\widehat{\mathbf{x}}_{i,t},\mathbf{x}_{i,t}).$$
#
#
#
#
# $$k=\left\lceil(n_{cal}+1)(1-\alpha)\right\rceil,\qquad
# q_t=s_{(k),t}^{cal}.$$
#
#
# $$C_t(X)=\{z:d_{\mathrm{Haversine}}(z,\widehat{x}_t)\le q_t\}.$$
#
#
# $$P\{x_t\in C_t(X)\}\ge 1-\alpha.$$
#


# %%
def latlon_to_local_km(origin_lat, origin_lon, latitude, longitude):
    lat0 = np.deg2rad(np.asarray(origin_lat, dtype=float))
    lon0 = np.deg2rad(np.asarray(origin_lon, dtype=float))
    lat1 = np.deg2rad(np.asarray(latitude, dtype=float))
    lon1 = np.deg2rad(np.asarray(longitude, dtype=float))
    dlon = (lon1 - lon0 + np.pi) % (2 * np.pi) - np.pi
    dlat = lat1 - lat0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    central_angle = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    bearing = np.arctan2(
        np.sin(dlon) * np.cos(lat1),
        np.cos(lat0) * np.sin(lat1) - np.sin(lat0) * np.cos(lat1) * np.cos(dlon),
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
    return np.rad2deg(latitude), (np.rad2deg(longitude) + 180.0) % 360.0 - 180.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# %%
observations = pd.read_parquet(
    PROCESSED / "region_A_observations.parquet",
    columns=["ID", "segment_id", "segment_step", "latitude", "longitude", "ve", "vn"],
)
n_cases = len(origins)

history = origins[[
    "forecast_case_id", "ID", "segment_id", "history_start_step",
    "origin_latitude", "origin_longitude",
]].loc[origins.index.repeat(HISTORY_HOURS)].reset_index(drop=True)
history["history_index"] = np.tile(np.arange(HISTORY_HOURS), n_cases)
history["segment_step"] = history["history_start_step"] + history["history_index"]
history = history.merge(
    observations, on=["ID", "segment_id", "segment_step"],
    how="left", validate="many_to_one",
).sort_values(["forecast_case_id", "history_index"]).reset_index(drop=True)

history["relative_east_km"], history["relative_north_km"] = latlon_to_local_km(
    history["origin_latitude"], history["origin_longitude"],
    history["latitude"], history["longitude"],
)
history_values = history[[
    "relative_east_km", "relative_north_km", "ve", "vn",
]].to_numpy().reshape(n_cases, HISTORY_HOURS, 4)

X = np.column_stack([
    origins["origin_latitude"].to_numpy(),
    origins["origin_longitude"].to_numpy(),
    history_values.reshape(n_cases, -1),
]).astype(float)

targets = origins[[
    "forecast_case_id", "ID", "segment_id", "target_start_step",
    "origin_latitude", "origin_longitude",
]].loc[origins.index.repeat(HORIZON_HOURS)].reset_index(drop=True)
targets["lead_time_hours"] = np.tile(np.arange(1, HORIZON_HOURS + 1), n_cases)
targets["segment_step"] = targets["target_start_step"] + targets["lead_time_hours"] - 1
targets = targets.merge(
    observations[["ID", "segment_id", "segment_step", "latitude", "longitude"]],
    on=["ID", "segment_id", "segment_step"], how="left", validate="many_to_one",
).sort_values(["forecast_case_id", "lead_time_hours"]).reset_index(drop=True)

checks = pd.Series({
    "cases = 5,117": n_cases == 5117,
    "X shape = 5,117 × 98": X.shape == (5117, 98),
    "Each case has 72 observed future positions": targets.groupby("forecast_case_id").size().eq(72).all(),
    "X contains only finite values": np.isfinite(X).all(),
}, name="Passed")
display(checks.to_frame())
assert checks.all()


# %%
archive = np.load(RQ1_OUTPUT / "fnn_model_parameters.npz")

def predict_saved_fnn(X_new):
    z = (X_new - archive["x_mean"]) / archive["x_scale"]
    z = np.maximum(z @ archive["W1"] + archive["b1"], 0.0)
    z = np.maximum(z @ archive["W2"] + archive["b2"], 0.0)
    y_standard = z @ archive["W3"] + archive["b3"]
    return y_standard * archive["y_scale"] + archive["y_mean"]

predicted_local = predict_saved_fnn(X).reshape(n_cases, HORIZON_HOURS, 2)

origin_lat_2d = np.repeat(origins["origin_latitude"].to_numpy()[:, None], HORIZON_HOURS, axis=1)
origin_lon_2d = np.repeat(origins["origin_longitude"].to_numpy()[:, None], HORIZON_HOURS, axis=1)
pred_lat, pred_lon = local_km_to_latlon(
    origin_lat_2d, origin_lon_2d,
    predicted_local[:, :, 0], predicted_local[:, :, 1],
)
true_lat = targets["latitude"].to_numpy().reshape(n_cases, HORIZON_HOURS)
true_lon = targets["longitude"].to_numpy().reshape(n_cases, HORIZON_HOURS)
errors_km = haversine_km(pred_lat, pred_lon, true_lat, true_lon)

cal_idx = np.flatnonzero(origins["partition"].eq("calibration").to_numpy())
test_idx = np.flatnonzero(origins["partition"].eq("test").to_numpy())

saved_test = pd.read_csv(RQ1_OUTPUT / "fnn_test_case_metrics.csv")
# Verify predictions by case ID rather than assuming identical CSV row ordering.
check_table = origins.loc[test_idx, ["forecast_case_id"]].copy()
check_table["rq3_error_km"] = errors_km[test_idx, -1]
check_table = check_table.merge(saved_test[["forecast_case_id", "final_72h_error_km"]], on="forecast_case_id", validate="one_to_one")
assert len(check_table) == len(test_idx)
max_difference = np.max(np.abs(check_table["rq3_error_km"] - check_table["final_72h_error_km"]))
print(f"Maximum difference from saved RQ1 72 h test error:{max_difference:.3e} km")
assert max_difference < 1e-8

# $$k=\lceil770\times0.9\rceil=693.$$
#

# %%
def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return np.partition(scores, k - 1)[k - 1]


quantile_rows = []
for alpha in ALPHAS:
    for lead_index in range(HORIZON_HOURS):
        q = conformal_quantile(errors_km[cal_idx, lead_index], alpha)
        quantile_rows.append({
            "alpha": alpha,
            "nominal_coverage": 1 - alpha,
            "lead_time_hours": lead_index + 1,
            "radius_km": q,
        })
quantiles = pd.DataFrame(quantile_rows)

display(
    quantiles[quantiles["lead_time_hours"].isin([24, 48, 72])]
    .pivot(index="lead_time_hours", columns="nominal_coverage", values="radius_km")
    .round(2)
)


# %%
evaluation_rows = []
for alpha in ALPHAS:
    q = (
        quantiles.loc[quantiles["alpha"].eq(alpha)]
        .sort_values("lead_time_hours")["radius_km"].to_numpy()
    )
    covered = errors_km[test_idx] <= q[None, :]
    area = 2 * np.pi * EARTH_RADIUS_KM**2 * (1 - np.cos(q / EARTH_RADIUS_KM))
    for lead_index in range(HORIZON_HOURS):
        evaluation_rows.append({
            "alpha": alpha,
            "nominal_coverage": 1 - alpha,
            "lead_time_hours": lead_index + 1,
            "empirical_coverage": covered[:, lead_index].mean(),
            "radius_km": q[lead_index],
            "area_km2": area[lead_index],
            "n_test_cases": len(test_idx),
        })
coverage_by_lead = pd.DataFrame(evaluation_rows)

key_hours = coverage_by_lead[coverage_by_lead["lead_time_hours"].isin([24, 48, 72])].copy()
key_hours["nominal_coverage"] = key_hours["nominal_coverage"].map(lambda x: f"{x:.0%}")
display(key_hours[[
    "nominal_coverage", "lead_time_hours", "empirical_coverage", "radius_km", "area_km2",
]].round({"empirical_coverage": 3, "radius_km": 2, "area_km2": 0}))

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
colors = {0.8: "#2ca25f", 0.9: "#3182bd", 0.95: "#de2d26"}
for nominal, group in coverage_by_lead.groupby("nominal_coverage"):
    label = f"{nominal:.0%} nominal"
    axes[0].plot(group["lead_time_hours"], group["empirical_coverage"],
                 lw=2.2, color=colors[nominal], label=label)
    axes[0].axhline(nominal, color=colors[nominal], ls="--", alpha=0.45)
    axes[1].plot(group["lead_time_hours"], group["radius_km"],
                 lw=2.2, color=colors[nominal], label=label)

axes[0].set(title="Test coverage versus lead time", xlabel="Lead time (h)", ylabel="Empirical coverage", ylim=(0.72, 1.01))
axes[1].set(title="Conformal radius versus lead time", xlabel="Lead time (h)", ylabel="Radius (km)")
for ax in axes:
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
fig.suptitle("Lead-time-specific circular prediction regions", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "rq3_coverage_and_radius_by_lead.png", dpi=200, bbox_inches="tight")
plt.close()



# %%
main_q = (
    quantiles.loc[quantiles["alpha"].eq(MAIN_ALPHA)]
    .sort_values("lead_time_hours")["radius_km"].to_numpy()
)
main_covered = errors_km[test_idx] <= main_q[None, :]
test_meta = origins.loc[test_idx, ["forecast_case_id", "ID"]].reset_index(drop=True)
test_drifters = test_meta["ID"].unique()
case_rows_by_drifter = {
    drifter: np.flatnonzero(test_meta["ID"].eq(drifter).to_numpy())
    for drifter in test_drifters
}

rng = np.random.default_rng(RANDOM_SEED)
bootstrap_hours = [24, 48, 72]
bootstrap_values = {hour: [] for hour in bootstrap_hours}
for _ in range(2000):
    sampled_drifters = rng.choice(test_drifters, size=len(test_drifters), replace=True)
    sampled_rows = np.concatenate([case_rows_by_drifter[d] for d in sampled_drifters])
    for hour in bootstrap_hours:
        bootstrap_values[hour].append(main_covered[sampled_rows, hour - 1].mean())

bootstrap_summary = pd.DataFrame([
    {
        "lead_time_hours": hour,
        "coverage": main_covered[:, hour - 1].mean(),
        "cluster_bootstrap_lower_95": np.quantile(values, 0.025),
        "cluster_bootstrap_upper_95": np.quantile(values, 0.975),
        "test_drifters": len(test_drifters),
        "test_cases": len(test_idx),
    }
    for hour, values in bootstrap_values.items()
])
display(bootstrap_summary.round(3))


# %%
regimes = pd.read_csv(RQ2_DATA / "forecast_case_regimes.csv")[["forecast_case_id", "regime"]]
test_regime_meta = (
    origins.loc[test_idx, ["forecast_case_id", "ID"]]
    .reset_index(drop=True)
    .merge(regimes, on="forecast_case_id", how="left", validate="one_to_one")
)
assert test_regime_meta["regime"].notna().all()

regime_rows = []
for regime, row_index in test_regime_meta.groupby("regime", sort=False).groups.items():
    row_index = np.asarray(list(row_index), dtype=int)
    for hour in [24, 48, 72]:
        regime_rows.append({
            "regime": regime,
            "lead_time_hours": hour,
            "n_cases": len(row_index),
            "n_drifters": test_regime_meta.loc[row_index, "ID"].nunique(),
            "empirical_coverage": main_covered[row_index, hour - 1].mean(),
            "global_radius_km": main_q[hour - 1],
            "median_error_km": np.median(errors_km[test_idx[row_index], hour - 1]),
            "rmse_km": np.sqrt(np.mean(errors_km[test_idx[row_index], hour - 1] ** 2)),
        })
regime_coverage = pd.DataFrame(regime_rows)
display(regime_coverage.round(3))

# %%
plot_data = regime_coverage[regime_coverage["lead_time_hours"].eq(72)].copy()
fig, ax = plt.subplots(figsize=(9, 4.8))
bars = ax.bar(plot_data["regime"], plot_data["empirical_coverage"], color=["#74c476", "#e34a33", "#fd8d3c", "#6baed6"])
ax.axhline(0.90, color="black", ls="--", lw=1.5, label="90% nominal")
ax.set(title="90% global conformal region: 72 h coverage by regime", ylabel="Empirical coverage", ylim=(0, 1.02))
ax.legend(frameon=False)
for bar, value in zip(bars, plot_data["empirical_coverage"]):
    ax.text(bar.get_x() + bar.get_width()/2, value + 0.025, f"{value:.1%}", ha="center", fontweight="bold")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "rq3_72h_coverage_by_regime.png", dpi=200, bbox_inches="tight")
plt.close()


# $$S_i=\max_{t=1,\ldots,72}\frac{e_{i,t}}{m_t}.$$
#

# %%
lead_scale = np.maximum(np.median(errors_km[cal_idx], axis=0), 1e-6)
simultaneous_scores = np.max(errors_km[cal_idx] / lead_scale[None, :], axis=1)
simultaneous_multiplier = conformal_quantile(simultaneous_scores, MAIN_ALPHA)
simultaneous_radius = simultaneous_multiplier * lead_scale
simultaneously_covered = np.all(errors_km[test_idx] <= simultaneous_radius[None, :], axis=1)

pointwise_all_hours = np.all(main_covered, axis=1)
simultaneous_summary = pd.DataFrame({
    "method": ["Pointwise 90% radii; all 72 positions required", "Calibrated 90% simultaneous trajectory band"],
    "whole_trajectory_coverage": [pointwise_all_hours.mean(), simultaneously_covered.mean()],
    "radius_24h_km": [main_q[23], simultaneous_radius[23]],
    "radius_48h_km": [main_q[47], simultaneous_radius[47]],
    "radius_72h_km": [main_q[71], simultaneous_radius[71]],
})
display(simultaneous_summary.round(3))



# %%
test_errors_72 = errors_km[test_idx, 71]
representative_row = np.argmin(np.abs(test_errors_72 - np.median(test_errors_72)))
case_index = test_idx[representative_row]
case_id = int(origins.loc[case_index, "forecast_case_id"])

fig, ax = plt.subplots(figsize=(7.5, 6.2))
ax.plot(true_lon[case_index], true_lat[case_index], color="#238b45", lw=2.2, label="Observed trajectory")
ax.plot(pred_lon[case_index], pred_lat[case_index], color="#cb181d", lw=2.0, ls="--", label="FNN point forecast")
ax.scatter(origins.loc[case_index, "origin_longitude"], origins.loc[case_index, "origin_latitude"],
           marker="*", s=140, color="black", zorder=5, label="Forecast origin")

for hour, color in [(24, "#9ecae1"), (48, "#4292c6"), (72, "#08519c")]:
    j = hour - 1
    theta = np.linspace(0, 2*np.pi, 240)
    radius = main_q[j]
    lat_center = pred_lat[case_index, j]
    lon_center = pred_lon[case_index, j]
    circle_lat = lat_center + (radius / 111.0) * np.sin(theta)
    circle_lon = lon_center + (radius / (111.0 * np.cos(np.deg2rad(lat_center)))) * np.cos(theta)
    ax.plot(circle_lon, circle_lat, color=color, lw=1.5, label=f"{hour} h: 90% radius {radius:.1f} km")
    ax.scatter(true_lon[case_index, j], true_lat[case_index, j], s=35, color=color, zorder=5)

ax.set(
    title=f"Representative test case {case_id}",
    xlabel="Longitude (°E)", ylabel="Latitude (°N)",
)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "rq3_representative_case_regions.png", dpi=200, bbox_inches="tight")
plt.close()

# - uncertainty validity：coverage（RQ3）；


# %%
main_key = coverage_by_lead[
    coverage_by_lead["alpha"].eq(MAIN_ALPHA)
    & coverage_by_lead["lead_time_hours"].isin([24, 48, 72])
].copy()

summary_lines = [
    "### RQ3 Results Summary",
    "",
    "We used 769 held-out calibration cases to calibrate a training-only FNN using split conformal prediction.",
]
for row in main_key.itertuples():
    summary_lines.append(
        f"- {row.lead_time_hours} h: 90% radius {row.radius_km:.2f} km; "
        f"empirical coverage across 757 held-out test cases: {row.empirical_coverage:.1%} ."
    )
kuro_72 = regime_coverage.query("regime == 'Kuroshio Core' and lead_time_hours == 72").iloc[0]
summary_lines.extend([
    f"- Overall 72 h test coverage meets the target, but Kuroshio Core coverage is only {kuro_72.empirical_coverage:.1%} revealing regime-specific undercoverage despite acceptable marginal coverage .",
    f"- The calibrated simultaneous trajectory band covers entire 72 h test trajectories at a rate of {simultaneously_covered.mean():.1%} .",
    "- Conclusion: split conformal yields useful overall prediction regions, but heterogeneous ocean regimes may require regime-aware or locally adaptive calibration .",
])
display(Markdown("\n".join(summary_lines)))


# %%
quantiles.to_csv(OUTPUT_DIR / "rq3_conformal_quantiles.csv", index=False)
coverage_by_lead.to_csv(OUTPUT_DIR / "rq3_coverage_by_lead.csv", index=False)
bootstrap_summary.to_csv(OUTPUT_DIR / "rq3_cluster_bootstrap_coverage.csv", index=False)
regime_coverage.to_csv(OUTPUT_DIR / "rq3_coverage_by_regime.csv", index=False)
simultaneous_summary.to_csv(OUTPUT_DIR / "rq3_simultaneous_trajectory_summary.csv", index=False)

saved = sorted(p.name for p in OUTPUT_DIR.iterdir())
print("Saved files:")
for name in saved:
    print(" -", name)

# **Methods**
#
# We applied split conformal prediction to the fixed FNN trajectory forecasts. The FNN was fitted using
# training drifters only. For each lead time, Haversine separation errors from 769 held-out calibration cases
# were used as nonconformity scores. A finite-sample corrected empirical quantile defined a circular prediction
# region centred on each FNN point forecast. Empirical coverage and region size were evaluated once on 757
# held-out test cases. Coverage was additionally stratified by ocean regime, and uncertainty in the reported
# coverage was assessed by resampling complete test drifters.
#
# **Results**
#
# The lead-time-specific conformal regions achieved approximately nominal marginal coverage on the full test
# set, while their radii increased with forecast horizon. However, regime-stratified evaluation revealed
# undercoverage in the Kuroshio Core and conservative coverage in lower-error regimes. Thus, global split
# conformal calibration provided useful overall uncertainty quantification but did not guarantee equal
# reliability across heterogeneous ocean environments.
#
#
# Overall conformal coverage was reliable, but uncertainty was not equally calibrated across ocean regimes,
# with the strongest undercoverage occurring in the dynamically energetic Kuroshio Core.
