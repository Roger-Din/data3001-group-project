"""RQ2 Step 15: drifter-clustered bootstrap for frozen RQ1 test errors.
Run from project root: python RQ2/15_cluster_bootstrap.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'RQ2' / 'data'
INPUT = DATA / 'rq2_test_model_errors_long.csv'
REGIMES = ['Shelf', 'Kuroshio Core', 'Transition', 'Other Offshore']
MODELS = ['Persistence', 'Mean Flow', 'Ridge', 'FNN', 'VAR(1)', 'KNN']
B = 2000
SEED = 3001

print('=== RQ2 STEP 15: DRIFTER-CLUSTERED BOOTSTRAP ===')
df = pd.read_csv(INPUT)
required = {'forecast_case_id', 'ID', 'partition', 'regime', 'model', 'final_72h_error_km'}
assert required.issubset(df.columns), f'Missing columns: {required - set(df.columns)}'
assert df['partition'].eq('test').all(), 'Non-test cases found'
assert len(df) == 757 * 6, f'Expected 4542 model-case rows, found {len(df)}'
assert df[['forecast_case_id', 'model']].duplicated().sum() == 0
assert set(df['model']) == set(MODELS)
assert df['regime'].isin(REGIMES).all()
assert np.isfinite(df['final_72h_error_km']).all()

cases = df[['forecast_case_id', 'ID', 'regime']].drop_duplicates('forecast_case_id').sort_values('forecast_case_id').reset_index(drop=True)
assert len(cases) == 757 and cases['ID'].nunique() == 71
wide = df.pivot(index='forecast_case_id', columns='model', values='final_72h_error_km').reindex(index=cases['forecast_case_id'], columns=MODELS)
assert not wide.isna().any().any()
E = wide.to_numpy(dtype=float)
regimes = cases['regime'].to_numpy()
drifters = cases['ID'].to_numpy()
unique_drifters, cluster_id = np.unique(drifters, return_inverse=True)
G = len(unique_drifters)
rng = np.random.default_rng(SEED)

# Resample the 71 test drifters globally, not separately by regime.
# A selected drifter contributes all of its cases, including those in other regimes.
# Replicates with no observations in a particular regime are marked NaN for that statistic.
counts = rng.multinomial(G, np.full(G, 1 / G), size=B)
case_weights = counts[:, cluster_id]  # B x 757; drifter multiplicities

def weighted_median(values, weights):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    total = w.sum()
    if total == 0:
        return np.nan
    # Exact np.median for repeated observations, including the even-sample midpoint.
    cs = np.cumsum(w)
    left = v[np.searchsorted(cs, (total - 1) // 2 + 1)]
    right = v[np.searchsorted(cs, total // 2 + 1)]
    return (left + right) / 2

def weighted_rmse(values, weights):
    total = weights.sum()
    return np.sqrt(np.dot(weights, values * values) / total) if total else np.nan

def ci(samples):
    samples = np.asarray(samples, dtype=float)
    valid = samples[np.isfinite(samples)]
    if len(valid) < int(B * 0.95):
        raise ValueError(f'Only {len(valid)}/{B} valid bootstrap draws; regime may be too sparse')
    return np.quantile(valid, [0.025, 0.975]), len(valid)

# Statistic arrays indexed by regime and model.
med = np.full((B, len(REGIMES), len(MODELS)), np.nan)
rmse = np.full_like(med, np.nan)
point_med = np.zeros((len(REGIMES), len(MODELS)))
point_rmse = np.zeros_like(point_med)

for r, regime in enumerate(REGIMES):
    mask = regimes == regime
    x = E[mask]
    w = case_weights[:, mask]
    for m, model in enumerate(MODELS):
        v = x[:, m]
        point_med[r, m] = np.median(v)
        point_rmse[r, m] = np.sqrt(np.mean(v * v))
        for b in range(B):
            med[b, r, m] = weighted_median(v, w[b])
        denominators = w.sum(axis=1)
        rmse[:, r, m] = np.sqrt(np.divide(w @ (v * v), denominators,
                                              out=np.full(B, np.nan), where=denominators > 0))
    print(f'Processed {regime}: {mask.sum()} cases, {len(np.unique(drifters[mask]))} drifters')

summary = []
for r, regime in enumerate(REGIMES):
    for m, model in enumerate(MODELS):
        med_ci, n_valid = ci(med[:, r, m])
        rmse_ci, _ = ci(rmse[:, r, m])
        summary.append(dict(regime=regime, model=model, n_cases=int((regimes == regime).sum()),
                            n_drifters=int(len(np.unique(drifters[regimes == regime]))),
                            median_72h_km=point_med[r, m], median_ci_lower_km=med_ci[0],
                            median_ci_upper_km=med_ci[1], rmse_72h_km=point_rmse[r, m],
                            rmse_ci_lower_km=rmse_ci[0], rmse_ci_upper_km=rmse_ci[1],
                            valid_bootstrap_draws=n_valid))
summary = pd.DataFrame(summary)

# Paired comparisons: model A minus model B, on identical cases and bootstrap samples.
# Negative = A has smaller error. Difference of group medians is NOT median of paired errors.
comparisons = []
for r, regime in enumerate(REGIMES):
    for competitor in ['KNN', 'Persistence', 'Ridge', 'Mean Flow', 'VAR(1)']:
        a, z = MODELS.index('FNN'), MODELS.index(competitor)
        for metric, boot, point in [('median', med, point_med), ('rmse', rmse, point_rmse)]:
            delta = boot[:, r, a] - boot[:, r, z]
            bounds, valid = ci(delta)
            comparisons.append(dict(regime=regime, metric=metric, model_a='FNN', model_b=competitor,
                                    difference_km=point[r, a] - point[r, z],
                                    ci_lower_km=bounds[0], ci_upper_km=bounds[1],
                                    ci_excludes_zero=bool(bounds[0] > 0 or bounds[1] < 0),
                                    valid_bootstrap_draws=valid))
comparisons = pd.DataFrame(comparisons)

# Within-model regime contrast: Kuroshio Core minus Shelf (positive = higher in Core).
contrasts = []
r_core, r_shelf = REGIMES.index('Kuroshio Core'), REGIMES.index('Shelf')
for m, model in enumerate(MODELS):
    for metric, boot, point in [('median', med, point_med), ('rmse', rmse, point_rmse)]:
        delta = boot[:, r_core, m] - boot[:, r_shelf, m]
        bounds, valid = ci(delta)
        contrasts.append(dict(model=model, metric=metric, regime_a='Kuroshio Core', regime_b='Shelf',
                              difference_km=point[r_core, m] - point[r_shelf, m],
                              ci_lower_km=bounds[0], ci_upper_km=bounds[1],
                              ci_excludes_zero=bool(bounds[0] > 0 or bounds[1] < 0),
                              valid_bootstrap_draws=valid))
contrasts = pd.DataFrame(contrasts)

out_summary = DATA / 'rq2_cluster_bootstrap_summary.csv'
out_comp = DATA / 'rq2_cluster_bootstrap_model_comparisons.csv'
out_contrast = DATA / 'rq2_cluster_bootstrap_regime_contrasts.csv'
summary.to_csv(out_summary, index=False)
comparisons.to_csv(out_comp, index=False)
contrasts.to_csv(out_contrast, index=False)

print('\n=== MEDIAN 72H ERROR, 95% CLUSTER CI (KM) ===')
for regime in REGIMES:
    print(f'\n{regime}')
    for _, row in summary.loc[summary.regime == regime].iterrows():
        print(f"  {row['model']:12s} {row['median_72h_km']:6.2f} [{row['median_ci_lower_km']:6.2f}, {row['median_ci_upper_km']:6.2f}]")
print('\n=== FNN MINUS KNN (NEGATIVE FAVOURS FNN) ===')
print(comparisons.loc[comparisons.model_b == 'KNN',
                      ['regime', 'metric', 'difference_km', 'ci_lower_km', 'ci_upper_km', 'ci_excludes_zero']].round(3).to_string(index=False))
print('\n=== CORE MINUS SHELF (POSITIVE = CORE HAS LARGER ERROR) ===')
print(contrasts[['model', 'metric', 'difference_km', 'ci_lower_km', 'ci_upper_km', 'ci_excludes_zero']].round(3).to_string(index=False))
print(f'\nBootstrap: {B} replicates; seed={SEED}; resampling unit=drifter ({G} clusters)')
print(f'Saved: {out_summary}\nSaved: {out_comp}\nSaved: {out_contrast}')
print('=== STEP 15 COMPLETE ===')
