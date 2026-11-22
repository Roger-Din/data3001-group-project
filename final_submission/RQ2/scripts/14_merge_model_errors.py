from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RQ1 = ROOT / 'output' / 'rq1' / 'en'
RQ2 = ROOT / 'RQ2' / 'data'
OUTPUT = RQ2 / 'rq2_test_model_errors_long.csv'
SUMMARY = RQ2 / 'rq2_regime_model_summary.csv'

MODELS = {
    'Persistence': 'velocity_persistence_test_case_metrics.csv',
    'Mean Flow': 'spatial_mean_flow_test_case_metrics.csv',
    'Ridge': 'ridge_test_case_metrics.csv',
    'FNN': 'fnn_test_case_metrics.csv',
    'VAR(1)': 'var1_test_case_metrics.csv',
    'KNN': 'knn_test_case_metrics.csv',
}
METRICS = ['final_72h_error_km', 'mean_trajectory_error_km', 'integrated_trajectory_error_km_h']
REGIMES = ['Shelf', 'Kuroshio Core', 'Transition', 'Other Offshore']

print('=== RQ2 STEP 14: MERGE RQ1 TEST ERRORS ===')
regimes = pd.read_csv(RQ2 / 'forecast_case_regimes.csv')
required = ['forecast_case_id', 'ID', 'partition', 'regime']
assert set(required).issubset(regimes.columns), 'RQ2 regime file missing columns'
assert not regimes['forecast_case_id'].duplicated().any(), 'RQ2 duplicate case IDs'
test = regimes.loc[regimes['partition'].eq('test')].copy()
assert len(test) == 757, f'Expected 757 test cases, found {len(test)}'
assert test['regime'].isin(REGIMES).all(), 'Unclassified/unknown test regimes'
assert test['ID'].nunique() == 71, 'Expected 71 unique test drifters'
ids = set(test['forecast_case_id'])
print(f'RQ2 test cases: {len(test)}; drifters: {test["ID"].nunique()}')

parts = []
for model, filename in MODELS.items():
    df = pd.read_csv(RQ1 / filename)
    assert {'forecast_case_id', *METRICS}.issubset(df.columns), f'{model}: missing required columns'
    assert len(df) == 757, f'{model}: expected 757 rows, found {len(df)}'
    assert df['forecast_case_id'].is_unique, f'{model}: duplicate case IDs'
    assert set(df['forecast_case_id']) == ids, f'{model}: test case IDs differ from RQ2'
    for metric in METRICS:
        vals = pd.to_numeric(df[metric], errors='coerce')
        assert np.isfinite(vals).all() and (vals >= 0).all(), f'{model}: invalid {metric}'
        df[metric] = vals
    df['model'] = model
    parts.append(df[['forecast_case_id', 'model', *METRICS]])
    print(f'  PASS: {model}: 757 matching cases')

errors = pd.concat(parts, ignore_index=True)
merged = errors.merge(test, on='forecast_case_id', how='left', validate='many_to_one', indicator=True)
assert merged['_merge'].eq('both').all(), 'Merge failed'
merged = merged.drop(columns='_merge')
assert len(merged) == 757 * 6
assert merged.groupby('forecast_case_id')['model'].nunique().eq(6).all()

summary = (merged.groupby(['regime', 'model'], observed=True)
    .agg(n_cases=('forecast_case_id', 'size'),
         n_drifters=('ID', 'nunique'),
         median_72h_km=('final_72h_error_km', 'median'),
         mean_72h_km=('final_72h_error_km', 'mean'),
         rmse_72h_km=('final_72h_error_km', lambda x: np.sqrt(np.mean(np.square(x)))),
         median_mean_trajectory_km=('mean_trajectory_error_km', 'median'))
    .reset_index())
summary['regime'] = pd.Categorical(summary['regime'], categories=REGIMES, ordered=True)
summary['model'] = pd.Categorical(summary['model'], categories=list(MODELS), ordered=True)
summary = summary.sort_values(['regime', 'model']).reset_index(drop=True)

merged.to_csv(OUTPUT, index=False)
summary.to_csv(SUMMARY, index=False)
print('\n=== MEDIAN 72H ENDPOINT ERROR (KM) ===')
print(summary.pivot(index='regime', columns='model', values='median_72h_km').round(2).to_string())
print('\n=== RMSE 72H ENDPOINT ERROR (KM) ===')
print(summary.pivot(index='regime', columns='model', values='rmse_72h_km').round(2).to_string())
print('\n=== OVERALL TEST MEDIAN AND RMSE ===')
for model in MODELS:
    x = merged.loc[merged['model'].eq(model), 'final_72h_error_km'].to_numpy()
    print(f'{model:15s} median={np.median(x):.4f} km; RMSE={np.sqrt(np.mean(x*x)):.4f} km')
print(f'\nSaved: {OUTPUT}\nSaved: {SUMMARY}')
print('=== STEP 14 COMPLETE ===')
