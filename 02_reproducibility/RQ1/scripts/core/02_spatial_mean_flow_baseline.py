#!/usr/bin/env python
# coding: utf-8

# # RQ1 Step 2: Spatial mean-flow baseline
# 
# This notebook estimates a multiresolution spatial mean-flow field from training drifters only and recursively advects the fixed forecast cases for 72 hours. It is evaluated on exactly the same held-out cases as velocity persistence.
# 

# ## 1. Method
# 
# Training observations are binned into 0.25-degree cells and their eastward and northward velocities are averaged. Sparse fine cells fall back to a 1-degree cell and then to the regional training mean. The flow is looked up again after every predicted movement, so trajectories may bend across cells.
# 

# In[1]:


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython
from IPython.display import display

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic('matplotlib', 'inline')

PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data' / 'processed').exists()),
    None,
)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Could not find data/processed; open the notebook from the project directory or a notebook subdirectory.')

PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'output' / 'rq1' / 'en'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6_371_000.0
HORIZON_HOURS = 72
FINE_GRID_DEG = 0.25
COARSE_GRID_DEG = 1.0
MIN_CELL_OBSERVATIONS = 24
GRID_ORIGIN_LON = 125.0
GRID_ORIGIN_LAT = 25.0

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print('Project directory: ', PROJECT_ROOT.resolve())
print('Output directory: ', OUTPUT_DIR.resolve())


# ## 2. Read fixed cases, drifter partitions and observations
# 
# The saved split identifies training, calibration and test drifters. Only training IDs may contribute observations to the flow field; fixed forecast origins retain the same partition used by all other models.
# 

# In[2]:


origins = pd.read_parquet(PROCESSED / 'forecast_origins.parquet')
splits = pd.read_csv(PROCESSED / 'drifter_splits.csv')
observations = pd.read_parquet(
    PROCESSED / 'region_A_observations.parquet',
    columns=['ID', 'segment_id', 'segment_step', 'time', 'latitude', 'longitude', 've', 'vn'],
)

training_ids = set(splits.loc[splits['partition'].eq('training'), 'ID'])
flow_training = observations[observations['ID'].isin(training_ids)].copy()
flow_training = flow_training.dropna(subset=['latitude', 'longitude', 've', 'vn'])

dataset_summary = pd.DataFrame({
    'Item': [
        'All forecast cases', 'Training cases', 'Calibration cases', 'Test cases',
        'Training drifters', 'Training observations used for flow estimation',
    ],
    'Count': [
        len(origins), origins['partition'].eq('training').sum(),
        origins['partition'].eq('calibration').sum(), origins['partition'].eq('test').sum(),
        len(training_ids), len(flow_training),
    ],
})
display(dataset_summary)

assert len(origins) == 5117
assert origins['partition'].eq('test').sum() == 757
assert set(flow_training['ID']).issubset(training_ids)
assert not set(splits.loc[splits['partition'].ne('training'), 'ID']).intersection(flow_training['ID'])


# ## 3. Build a training-only multiscale mean-flow field
# 
# Fine and coarse cells require at least 24 observations. The regional mean is the final fallback. Counts are retained so that sparse estimates and fallback use remain auditable.
# 

# In[3]:


def add_grid_indices(frame, grid_deg):
    result = frame.copy()
    result['grid_x'] = np.floor((result['longitude'] - GRID_ORIGIN_LON) / grid_deg).astype(int)
    result['grid_y'] = np.floor((result['latitude'] - GRID_ORIGIN_LAT) / grid_deg).astype(int)
    return result


def build_flow_grid(frame, grid_deg, minimum_count):
    indexed = add_grid_indices(frame, grid_deg)
    grid = (
        indexed.groupby(['grid_x', 'grid_y'])
        .agg(
            mean_ve=('ve', 'mean'),
            mean_vn=('vn', 'mean'),
            n_observations=('ve', 'size'),
            n_drifters=('ID', 'nunique'),
        )
        .reset_index()
    )
    grid = grid[grid['n_observations'].ge(minimum_count)].copy()
    grid['center_longitude'] = GRID_ORIGIN_LON + (grid['grid_x'] + 0.5) * grid_deg
    grid['center_latitude'] = GRID_ORIGIN_LAT + (grid['grid_y'] + 0.5) * grid_deg
    grid['mean_speed'] = np.hypot(grid['mean_ve'], grid['mean_vn'])
    return grid


fine_flow = build_flow_grid(flow_training, FINE_GRID_DEG, MIN_CELL_OBSERVATIONS)
coarse_flow = build_flow_grid(flow_training, COARSE_GRID_DEG, MIN_CELL_OBSERVATIONS)
regional_mean_ve = float(flow_training['ve'].mean())
regional_mean_vn = float(flow_training['vn'].mean())

fine_lookup = {
    (int(r.grid_x), int(r.grid_y)): (float(r.mean_ve), float(r.mean_vn))
    for r in fine_flow.itertuples()
}
coarse_lookup = {
    (int(r.grid_x), int(r.grid_y)): (float(r.mean_ve), float(r.mean_vn))
    for r in coarse_flow.itertuples()
}

flow_summary = pd.DataFrame({
    'Flow-field level': ['0.25-degree fine grid', '1-degree coarse grid', 'Regional mean'],
    'Available cells': [len(fine_flow), len(coarse_flow), 1],
    'Minimum observations': [MIN_CELL_OBSERVATIONS, MIN_CELL_OBSERVATIONS, len(flow_training)],
})
display(flow_summary)
print(f'Regional mean velocity: ve={regional_mean_ve:.4f} m/s, vn={regional_mean_vn:.4f} m/s')
assert len(fine_lookup) > 0 and len(coarse_lookup) > 0


# In[4]:


fig, ax = plt.subplots(figsize=(11, 5.2))
q = ax.quiver(
    fine_flow['center_longitude'], fine_flow['center_latitude'],
    fine_flow['mean_ve'], fine_flow['mean_vn'], fine_flow['mean_speed'],
    cmap='viridis', angles='xy', scale_units='xy', scale=1.8, width=0.0028,
)
fig.colorbar(q, ax=ax, label='Mean speed (m/s)')
ax.set(
    xlabel='Longitude (degrees E)', ylabel='Latitude (degrees N)',
    title='0.25-degree spatial mean-flow field estimated from training drifters only',
)
ax.set_xlim(125, 135)
ax.set_ylim(25, 35)
ax.grid(alpha=0.2)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'spatial_mean_flow_field.png', dpi=200, bbox_inches='tight')
plt.show()


# ## 4. Recursively forecast 72 future positions
# 
# At each hour, the model looks up the flow at the current predicted position, converts velocity to a one-hour displacement and updates latitude and longitude. This process is repeated for all 72 lead times.
# 

# In[5]:


def grid_key(lon, lat, grid_deg):
    return (
        int(np.floor((lon - GRID_ORIGIN_LON) / grid_deg)),
        int(np.floor((lat - GRID_ORIGIN_LAT) / grid_deg)),
    )


def lookup_mean_flow(latitudes, longitudes):
    ve_out = np.empty(len(latitudes), dtype=float)
    vn_out = np.empty(len(latitudes), dtype=float)
    sources = np.empty(len(latitudes), dtype=object)

    for i, (lat, lon) in enumerate(zip(latitudes, longitudes)):
        fine = fine_lookup.get(grid_key(lon, lat, FINE_GRID_DEG))
        if fine is not None:
            ve_out[i], vn_out[i] = fine
            sources[i] = 'fine_0.25deg'
            continue

        coarse = coarse_lookup.get(grid_key(lon, lat, COARSE_GRID_DEG))
        if coarse is not None:
            ve_out[i], vn_out[i] = coarse
            sources[i] = 'coarse_1deg'
            continue

        ve_out[i], vn_out[i] = regional_mean_ve, regional_mean_vn
        sources[i] = 'regional_mean'

    return ve_out, vn_out, sources


state = origins[[
    'forecast_case_id', 'ID', 'segment_id', 'partition', 'target_start_step',
    'origin_time', 'origin_latitude', 'origin_longitude',
]].copy()
state['pred_latitude'] = state['origin_latitude'].astype(float)
state['pred_longitude'] = state['origin_longitude'].astype(float)

prediction_steps = []
for lead_time in range(1, HORIZON_HOURS + 1):
    flow_ve, flow_vn, flow_source = lookup_mean_flow(
        state['pred_latitude'].to_numpy(),
        state['pred_longitude'].to_numpy(),
    )

    lat_rad = np.deg2rad(state['pred_latitude'].to_numpy())
    lon_rad = np.deg2rad(state['pred_longitude'].to_numpy())
    pred_lat_rad = lat_rad + flow_vn * 3600.0 / EARTH_RADIUS_M
    cos_lat = np.cos(lat_rad)
    if np.any(np.abs(cos_lat) < 1e-8):
        raise ValueError('A position is too close to a pole for stable longitude integration.')
    pred_lon_rad = lon_rad + flow_ve * 3600.0 / (EARTH_RADIUS_M * cos_lat)

    state['pred_latitude'] = np.rad2deg(pred_lat_rad)
    state['pred_longitude'] = (np.rad2deg(pred_lon_rad) + 180.0) % 360.0 - 180.0

    step = state[[
        'forecast_case_id', 'ID', 'segment_id', 'partition', 'target_start_step',
        'origin_time', 'origin_latitude', 'origin_longitude',
        'pred_latitude', 'pred_longitude',
    ]].copy()
    step['lead_time_hours'] = lead_time
    step['flow_ve'] = flow_ve
    step['flow_vn'] = flow_vn
    step['flow_source'] = flow_source
    step['segment_step'] = step['target_start_step'] + lead_time - 1
    prediction_steps.append(step)

evaluation = pd.concat(prediction_steps, ignore_index=True)
display(evaluation[[
    'forecast_case_id', 'lead_time_hours', 'flow_source',
    'flow_ve', 'flow_vn', 'pred_latitude', 'pred_longitude',
]].head())

assert len(evaluation) == len(origins) * HORIZON_HOURS
assert evaluation.groupby('forecast_case_id').size().eq(HORIZON_HOURS).all()


# ## 5. Join observations and calculate Haversine error
# 
# Predicted positions are matched to the fixed observed future positions and evaluated with the same great-circle distance used by every RQ1 model.
# 

# In[6]:


truth = observations[['ID', 'segment_id', 'segment_step', 'time', 'latitude', 'longitude']].rename(columns={
    'time': 'valid_time',
    'latitude': 'true_latitude',
    'longitude': 'true_longitude',
})
evaluation = evaluation.merge(
    truth,
    on=['ID', 'segment_id', 'segment_step'],
    how='left',
    validate='many_to_one',
)

def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.deg2rad(np.asarray(lat1, dtype=float))
    lon1 = np.deg2rad(np.asarray(lon1, dtype=float))
    lat2 = np.deg2rad(np.asarray(lat2, dtype=float))
    lon2 = np.deg2rad(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * (EARTH_RADIUS_M / 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

evaluation['separation_error_km'] = haversine_km(
    evaluation['pred_latitude'], evaluation['pred_longitude'],
    evaluation['true_latitude'], evaluation['true_longitude'],
)

checks = pd.Series({
    'All observed positions matched successfully': evaluation[['true_latitude', 'true_longitude']].notna().all().all(),
    'All forecast errors are finite': np.isfinite(evaluation['separation_error_km']).all(),
    'Every case has exactly 72 lead times': evaluation.groupby('forecast_case_id').size().eq(72).all(),
}, name='Passed')
display(checks.to_frame())
assert checks.all()


# ## 6. Held-out test results by lead time
# 
# Report median, mean, interquartile range and RMSE across the 757 test cases, together with the frequency of each flow-field fallback level.
# 

# In[7]:


test_eval = evaluation[evaluation['partition'].eq('test')].copy()
lead_metrics = (
    test_eval.groupby('lead_time_hours')['separation_error_km']
    .agg(
        n='size',
        median_error_km='median',
        mean_error_km='mean',
        q25_error_km=lambda x: x.quantile(0.25),
        q75_error_km=lambda x: x.quantile(0.75),
        rmse_km=lambda x: np.sqrt(np.mean(np.square(x))),
    )
    .reset_index()
)

fallback_usage = (
    test_eval['flow_source'].value_counts()
    .rename_axis('Flow source').reset_index(name='Prediction steps')
)
fallback_usage['Share of test prediction steps'] = fallback_usage['Prediction steps'] / len(test_eval)

assert test_eval['forecast_case_id'].nunique() == 757
assert lead_metrics['n'].eq(757).all()
display(lead_metrics[lead_metrics['lead_time_hours'].isin([1, 6, 12, 24, 48, 72])].round(2))
display(fallback_usage.style.format({'Share of test prediction steps': '{:.1%}'}))


# In[8]:


fig, ax = plt.subplots(figsize=(10, 4.8))
x = lead_metrics['lead_time_hours'].to_numpy()
median = lead_metrics['median_error_km'].to_numpy()
q25 = lead_metrics['q25_error_km'].to_numpy()
q75 = lead_metrics['q75_error_km'].to_numpy()
ax.plot(x, median, color='#7b3294', linewidth=2.5, label='Median separation error')
ax.fill_between(x, q25, q75, color='#7b3294', alpha=0.18, label='25th--75th percentile interval')
ax.set(
    xlabel='Lead time (hours)', ylabel='Separation error (km)',
    title='Spatial mean flow: test error by forecast lead time',
)
ax.set_xlim(1, 72)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'spatial_mean_flow_error_curve.png', dpi=200, bbox_inches='tight')
plt.show()


# ## 7. Case-level integrated trajectory error and summary
# 
# ITE integrates each case's separation error over the full forecast horizon. The compact summary also reports the 24-, 48- and 72-hour errors.
# 

# In[9]:


def case_metrics(group):
    ordered = group.sort_values('lead_time_hours')
    errors = ordered['separation_error_km'].to_numpy()
    hours = ordered['lead_time_hours'].to_numpy()
    ite = np.trapezoid(np.r_[0.0, errors], x=np.r_[0, hours])
    return pd.Series({
        'integrated_trajectory_error_km_h': ite,
        'final_72h_error_km': errors[-1],
        'mean_trajectory_error_km': errors.mean(),
    })

test_case_metrics = test_eval.groupby('forecast_case_id', sort=False).apply(
    case_metrics, include_groups=False
).reset_index()

def metric_at(hour, column):
    return float(lead_metrics.loc[lead_metrics['lead_time_hours'].eq(hour), column].iloc[0])

summary = pd.DataFrame({
    'Metric': [
        'Test cases', '24h median error (km)', '48h median error (km)',
        '72h median error (km)', '72h RMSE (km)', 'Median ITE (km·h)',
    ],
    'Value': [
        test_eval['forecast_case_id'].nunique(),
        metric_at(24, 'median_error_km'), metric_at(48, 'median_error_km'),
        metric_at(72, 'median_error_km'), metric_at(72, 'rmse_km'),
        test_case_metrics['integrated_trajectory_error_km_h'].median(),
    ],
})
display(summary.round(2))


# ## 8. Fair comparison with velocity persistence
# 
# Both baselines use identical test cases, lead times, forecast origins and Haversine evaluation, so differences reflect the forecasting rule rather than a different sample.
# 

# In[10]:


persistence_path = OUTPUT_DIR / 'velocity_persistence_lead_metrics.csv'
if persistence_path.exists():
    persistence_metrics = pd.read_csv(persistence_path)
    comparison = lead_metrics[['lead_time_hours', 'median_error_km', 'rmse_km']].merge(
        persistence_metrics[['lead_time_hours', 'median_error_km', 'rmse_km']],
        on='lead_time_hours', suffixes=('_mean_flow', '_persistence'), validate='one_to_one',
    )

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(
        comparison['lead_time_hours'], comparison['median_error_km_persistence'],
        color='#1764ab', linewidth=2.4, label='Velocity persistence',
    )
    ax.plot(
        comparison['lead_time_hours'], comparison['median_error_km_mean_flow'],
        color='#7b3294', linewidth=2.4, label='Spatial mean flow',
    )
    ax.set(
        xlabel='Lead time (hours)', ylabel='Median separation error (km)',
        title='RQ1 baselines: error comparison on identical test cases',
    )
    ax.set_xlim(1, 72)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'baseline_comparison_error_curve.png', dpi=200, bbox_inches='tight')
    plt.show()

    display(comparison[comparison['lead_time_hours'].isin([24, 48, 72])].round(2))
else:
    print('velocity_persistence_lead_metrics.csv was not found; run the first baseline notebook to create the comparison.')


# ## 9. Representative trajectory
# 
# The example is selected because its 72-hour error is closest to the test median.
# 

# In[11]:


median_final = test_case_metrics['final_72h_error_km'].median()
example_id = test_case_metrics.loc[
    (test_case_metrics['final_72h_error_km'] - median_final).abs().idxmin(),
    'forecast_case_id',
]
example = test_eval[test_eval['forecast_case_id'].eq(example_id)].sort_values('lead_time_hours')

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(
    example['true_longitude'], example['true_latitude'], '-o', markersize=2.5,
    linewidth=2, color='#1b9e77', label='Observed trajectory',
)
ax.plot(
    example['pred_longitude'], example['pred_latitude'], '--o', markersize=2.5,
    linewidth=2, color='#7b3294', label='Spatial mean flow',
)
ax.scatter(
    example['origin_longitude'].iloc[0], example['origin_latitude'].iloc[0],
    marker='*', s=150, color='black', label='Forecast origin', zorder=5,
)
ax.set(
    xlabel='Longitude (degrees E)', ylabel='Latitude (degrees N)',
    title=f'Representative test case: {int(example_id)} (72 h error near the median)',
)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / 'spatial_mean_flow_example_trajectory.png', dpi=200, bbox_inches='tight')
plt.show()


# ## 10. Save reproducible results
# 
# Save the flow grids, fallback diagnostics, test metrics, summary and poster-ready figures.
# 

# In[12]:


fine_flow.to_csv(OUTPUT_DIR / 'spatial_mean_flow_grid.csv', index=False)
lead_metrics.to_csv(OUTPUT_DIR / 'spatial_mean_flow_lead_metrics.csv', index=False)
test_case_metrics.to_csv(OUTPUT_DIR / 'spatial_mean_flow_test_case_metrics.csv', index=False)
fallback_usage.to_csv(OUTPUT_DIR / 'spatial_mean_flow_fallback_usage.csv', index=False)
summary.to_csv(OUTPUT_DIR / 'spatial_mean_flow_summary.csv', index=False)

saved_files = sorted(p.name for p in OUTPUT_DIR.glob('spatial_mean_flow_*'))
print('Saved:')
for name in saved_files:
    print(' -', name)

