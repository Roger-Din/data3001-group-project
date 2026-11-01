#!/usr/bin/env python3
"""RQ2 Step 16A: Paired model advantages by ocean regime, clustered by drifter."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'RQ2/data/rq2_test_model_errors_long.csv'
OUT = ROOT / 'RQ2/data'
OUT.mkdir(parents=True, exist_ok=True)
SEED = 3001
B = 2000
REGIMES = ['Shelf', 'Kuroshio Core', 'Transition', 'Other Offshore']
COMPARATORS = ['Persistence', 'Mean Flow', 'Ridge', 'VAR(1)', 'KNN']

raw = pd.read_csv(INPUT)
required = {'forecast_case_id', 'ID', 'partition', 'model', 'regime', 'final_72h_error_km'}
missing = required - set(raw.columns)
if missing:
    raise ValueError(f'Missing columns: {sorted(missing)}')
if raw.duplicated(['forecast_case_id', 'model']).any():
    raise ValueError('Duplicate forecast_case_id/model combinations')
raw = raw.loc[raw['partition'].eq('test')].copy()
if len(raw) != 4542:
    raise ValueError(f'Expected 4542 test model-case rows, found {len(raw)}')
meta = raw[['forecast_case_id', 'ID', 'regime']].drop_duplicates()
if meta['forecast_case_id'].duplicated().any():
    raise ValueError('Inconsistent ID/regime for a forecast case')
wide = raw.pivot(index='forecast_case_id', columns='model', values='final_72h_error_km')
expected = ['FNN'] + COMPARATORS
if set(wide.columns) != set(expected) or len(wide) != 757 or wide.isna().any().any():
    raise ValueError('Incomplete six-model test predictions')
df = meta.set_index('forecast_case_id').join(wide, validate='one_to_one').reset_index()
if df['ID'].nunique() != 71 or set(df['regime']) != set(REGIMES):
    raise ValueError('Unexpected drifter count or regime categories')
for m in expected:
    if not np.isfinite(df[m]).all() or (df[m] < 0).any():
        raise ValueError(f'Invalid errors for {m}')

# Paired advantage: positive means FNN has smaller endpoint error.
for comparator in COMPARATORS:
    df[f'advantage_vs_{comparator}'] = df[comparator] - df['FNN']

# Resample all 71 drifters together, preserving cases and cross-regime membership.
rng = np.random.default_rng(SEED)
ids = df['ID'].unique()
cluster_idx = [np.flatnonzero(df['ID'].to_numpy() == d) for d in ids]
regime_arr = df['regime'].to_numpy()
results = []
contrasts = []
print('=== RQ2 STEP 16A: PAIRED ADVANTAGE AND REGIME INTERACTION ===')
print(f'Test cases={len(df)}; drifters={len(ids)}; bootstrap={B}; seed={SEED}')

for comparator in COMPARATORS:
    col = f'advantage_vs_{comparator}'
    values = df[col].to_numpy(dtype=float)
    point = {r: float(values[regime_arr == r].mean()) for r in REGIMES}
    point_med = {r: float(np.median(values[regime_arr == r])) for r in REGIMES}
    boot = {r: np.full(B, np.nan) for r in REGIMES}
    # Same cluster samples across all regimes within each comparator.
    # Resetting seed gives identical resamples across comparators too.
    rng = np.random.default_rng(SEED)
    for b in range(B):
        picks = rng.integers(0, len(ids), size=len(ids))
        ix = np.concatenate([cluster_idx[k] for k in picks])
        for r in REGIMES:
            sel = ix[regime_arr[ix] == r]
            if len(sel):
                boot[r][b] = values[sel].mean()
    for r in REGIMES:
        v = boot[r][np.isfinite(boot[r])]
        lo, hi = np.percentile(v, [2.5, 97.5])
        results.append(dict(regime=r, comparator=comparator,
                            n_cases=int((regime_arr == r).sum()),
                            n_drifters=int(df.loc[df['regime'].eq(r), 'ID'].nunique()),
                            mean_advantage_km=point[r], median_paired_advantage_km=point_med[r],
                            ci_lower_km=lo, ci_upper_km=hi,
                            ci_excludes_zero=bool(lo > 0 or hi < 0),
                            valid_bootstrap_replicates=len(v)))
    for r in REGIMES:
        if r == 'Shelf':
            continue
        diff = boot[r] - boot['Shelf']
        valid = diff[np.isfinite(diff)]
        lo, hi = np.percentile(valid, [2.5, 97.5])
        contrasts.append(dict(comparator=comparator, contrast=f'{r} minus Shelf',
                              difference_in_mean_advantage_km=point[r] - point['Shelf'],
                              ci_lower_km=lo, ci_upper_km=hi,
                              ci_excludes_zero=bool(lo > 0 or hi < 0),
                              valid_bootstrap_replicates=len(valid)))

summary = pd.DataFrame(results)
interaction = pd.DataFrame(contrasts)
paired = df[['forecast_case_id', 'ID', 'regime'] + [f'advantage_vs_{m}' for m in COMPARATORS]]
paired.to_csv(OUT / 'rq2_paired_case_advantages.csv', index=False)
summary.to_csv(OUT / 'rq2_model_advantage_by_regime.csv', index=False)
interaction.to_csv(OUT / 'rq2_model_advantage_regime_contrasts.csv', index=False)

print('\n=== MEAN PAIRED ADVANTAGE: BASELINE ERROR MINUS FNN ERROR (KM) ===')
print(summary.pivot(index='regime', columns='comparator', values='mean_advantage_km').reindex(REGIMES).round(2).to_string())
print('\n=== FNN ADVANTAGE VS PERSISTENCE: CLUSTER 95% CI ===')
print(summary.loc[summary['comparator'].eq('Persistence'),
                  ['regime', 'mean_advantage_km', 'ci_lower_km', 'ci_upper_km', 'ci_excludes_zero']].round(3).to_string(index=False))
print('\n=== INTERACTION: CHANGE IN FNN ADVANTAGE RELATIVE TO SHELF ===')
print(interaction[['comparator', 'contrast', 'difference_in_mean_advantage_km',
                   'ci_lower_km', 'ci_upper_km', 'ci_excludes_zero']].round(3).to_string(index=False))
print('\nSaved 3 files in:', OUT)
print('=== STEP 16A COMPLETE ===')
