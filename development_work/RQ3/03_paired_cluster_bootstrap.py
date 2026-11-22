"""RQ3: Paired drifter-cluster bootstrap comparing Global and Mondrian CP.

Run from DATA3001 project root:
    python RQ3/03_paired_cluster_bootstrap.py

Requires the existing RQ3/02_regime_and_drifter_aware.py and its inputs.
Resamples TEST drifters (not cases) while keeping both calibrated methods fixed.
This quantifies test-set sampling uncertainty; it does not establish CP validity
under dependent calibration cases or erase test-informed method selection.
"""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd

# Prevent blocking matplotlib windows in the imported RQ3 scripts.
import matplotlib
matplotlib.use('Agg')

HERE = Path(__file__).resolve().parent
previous = runpy.run_path(str(HERE / '02_regime_and_drifter_aware.py'))
meta = previous['meta']
errors = previous['errors']
test_idx = previous['test_idx']
result = previous['result']
out = HERE / 'output'
out.mkdir(exist_ok=True)

N_BOOT = 2000
SEED = 3001
METHODS = ['Global case CP', 'Mondrian case CP']
REGIMES = sorted(result['regime'].unique())

# Build one row per test forecasting case; preserve the exact original ordering.
test = meta.iloc[test_idx][['forecast_case_id', 'ID', 'regime']].copy().reset_index(drop=True)
test['error_km'] = errors[test_idx, 71]
assert test['forecast_case_id'].is_unique
assert len(test) == 757 and test['ID'].nunique() == 71

radii = result.pivot(index='regime', columns='method', values='radius_km')
areas = result.pivot(index='regime', columns='method', values='area_km2')
for method in METHODS:
    test[f'covered_{method}'] = (test['error_km'].to_numpy() <=
        test['regime'].map(radii[method]).to_numpy()).astype(float)
    test[f'area_{method}'] = test['regime'].map(areas[method]).to_numpy()

# Method differences are Mondrian minus Global; positive coverage means
# improved coverage, positive area means a larger (less efficient) region.
def statistics(rows):
    subset = test.iloc[rows]
    g_cov = subset['covered_Global case CP'].mean()
    m_cov = subset['covered_Mondrian case CP'].mean()
    g_area = subset['area_Global case CP'].mean()
    m_area = subset['area_Mondrian case CP'].mean()
    d = {
        'overall_coverage_global': g_cov,
        'overall_coverage_mondrian': m_cov,
        'overall_coverage_diff_pp': 100 * (m_cov - g_cov),
        'overall_mean_area_global_km2': g_area,
        'overall_mean_area_mondrian_km2': m_area,
        'overall_mean_area_diff_km2': m_area - g_area,
        'overall_mean_area_diff_pct': 100 * (m_area / g_area - 1),
    }
    gc = []
    mc = []
    for regime in REGIMES:
        r = subset[subset['regime'].eq(regime)]
        # Some cluster bootstrap draws can omit a regime entirely. Treat the
        # regime-specific statistic as undefined rather than inventing 0%.
        if len(r) == 0:
            d[f'{regime}__coverage_global'] = np.nan
            d[f'{regime}__coverage_mondrian'] = np.nan
            d[f'{regime}__coverage_diff_pp'] = np.nan
            continue
        g = r['covered_Global case CP'].mean()
        m = r['covered_Mondrian case CP'].mean()
        gc.append(g)
        mc.append(m)
        d[f'{regime}__coverage_global'] = g
        d[f'{regime}__coverage_mondrian'] = m
        d[f'{regime}__coverage_diff_pp'] = 100 * (m - g)
    d['regime_coverage_range_global_pp'] = 100 * (max(gc) - min(gc)) if len(gc) == len(REGIMES) else np.nan
    d['regime_coverage_range_mondrian_pp'] = 100 * (max(mc) - min(mc)) if len(mc) == len(REGIMES) else np.nan
    d['regime_coverage_range_diff_pp'] = (d['regime_coverage_range_mondrian_pp'] -
                                            d['regime_coverage_range_global_pp'])
    return d

rng = np.random.default_rng(SEED)
ids = test['ID'].unique()
by_id = {drifter: np.flatnonzero(test['ID'].eq(drifter).to_numpy()) for drifter in ids}
observed = statistics(np.arange(len(test)))
boot = []
for _ in range(N_BOOT):
    chosen = rng.choice(ids, size=len(ids), replace=True)
    sample_rows = np.concatenate([by_id[drifter] for drifter in chosen])
    boot.append(statistics(sample_rows))
boot = pd.DataFrame(boot)

summaries = []
for key, estimate in observed.items():
    valid = boot[key].dropna()
    if valid.empty:
        low = high = np.nan
    else:
        low, high = np.quantile(valid, [0.025, 0.975])
    summaries.append({'metric': key, 'estimate': estimate,
                      'ci95_lower': low, 'ci95_upper': high,
                      'valid_bootstrap_reps': len(valid)})
summary = pd.DataFrame(summaries)
summary.to_csv(out / 'rq3_paired_cluster_bootstrap_summary.csv', index=False)
boot.to_csv(out / 'rq3_paired_cluster_bootstrap_replicates.csv', index=False)

focus = ['overall_coverage_diff_pp', 'overall_mean_area_diff_km2',
         'overall_mean_area_diff_pct', 'regime_coverage_range_diff_pp',
         'Kuroshio Core__coverage_diff_pp', 'Shelf__coverage_diff_pp',
         'Transition__coverage_diff_pp', 'Other Offshore__coverage_diff_pp']
print('\n=== RQ3: Paired drifter-cluster bootstrap (2,000 repetitions) ===')
print('Differences are Mondrian MINUS Global; pp = percentage points.')
print(summary.set_index('metric').loc[focus].to_string(float_format=lambda x: f'{x:.3f}'))
print('\nSaved:')
print(out / 'rq3_paired_cluster_bootstrap_summary.csv')
print(out / 'rq3_paired_cluster_bootstrap_replicates.csv')
print('\nNOTE: intervals quantify test-drifter sampling variability with calibration radii held fixed.')
print('They do NOT prove finite-sample conformal coverage with dependent calibration cases.')
