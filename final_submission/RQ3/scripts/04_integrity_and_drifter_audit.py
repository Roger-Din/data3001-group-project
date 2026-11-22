"""DATA3001 RQ3 independent integrity + drifter-level audit.
Run from the DATA3001 project root:
    python RQ3/04_integrity_and_drifter_audit.py

Runs existing RQ3 pipeline; does not retrain or alter model predictions.
Writes CSV reports to RQ3/output/. Fails loudly on integrity mismatches.
"""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # prevent GUI blocking; before importing prior scripts

HERE = Path(__file__).resolve().parent
ctx = runpy.run_path(str(HERE / '02_regime_and_drifter_aware.py'))
meta = ctx['meta'].copy().reset_index(drop=True)
origins = ctx['origins'].reset_index(drop=True)
errors = np.asarray(ctx['errors'])
cal_idx = np.asarray(ctx['cal_idx'])
test_idx = np.asarray(ctx['test_idx'])
result = ctx['result']
qcp = ctx['qcp']
project = Path(ctx['project'])
out = HERE / 'output'
out.mkdir(exist_ok=True)
checks = []

def check(name, passed, detail=''):
    passed = bool(passed)
    checks.append({'check': name, 'passed': passed, 'detail': str(detail)})
    print(('PASS' if passed else 'FAIL') + ' | ' + name + (f' | {detail}' if detail else ''))

print('\n=== RQ3 INTEGRITY AUDIT ===')
check('Unique case IDs', meta['forecast_case_id'].is_unique)
check('Origin/meta order and case IDs match', meta['forecast_case_id'].equals(origins['forecast_case_id']))
check('Expected data dimensions', len(meta) == 5117 and errors.shape == (5117, 72), f'{len(meta)} cases; errors {errors.shape}')
check('Finite prediction errors', np.isfinite(errors).all())
check('Nonnegative prediction errors', (errors >= 0).all())
check('Calibration case count', len(cal_idx) == 769, len(cal_idx))
check('Test case count', len(test_idx) == 757, len(test_idx))
check('Calibration/test case rows disjoint', not bool(set(cal_idx) & set(test_idx)))
cal_ids = set(meta.iloc[cal_idx]['ID'])
test_ids = set(meta.iloc[test_idx]['ID'])
train_ids = set(meta.loc[meta['partition'].eq('training'), 'ID'])
check('Calibration/test drifters disjoint', not bool(cal_ids & test_ids), f'{len(cal_ids)} cal; {len(test_ids)} test')
check('Training/calibration/test drifters disjoint', not bool((train_ids & cal_ids) | (train_ids & test_ids)))
check('Correct partition indices', meta.iloc[cal_idx]['partition'].eq('calibration').all() and meta.iloc[test_idx]['partition'].eq('test').all())
# Only calibration/test require one of the four RQ3 regimes. Five training
# cases are legitimately labelled Unclassified and are not used for CP calibration/evaluation.
valid_regimes = {'Shelf', 'Kuroshio Core', 'Transition', 'Other Offshore'}
evaluated = meta.loc[meta['partition'].isin(['calibration', 'test'])]
check('Four complete regime labels (calibration/test)',
      evaluated['regime'].notna().all()
      and set(evaluated['regime']) == valid_regimes,
      f'evaluated={len(evaluated)}; training_unclassified={int((meta.loc[meta.partition.eq("training"), "regime"] == "Unclassified").sum())}')

# Compare saved RQ1 errors by forecast_case_id, never by row position.
rq1_path = project / 'output' / 'rq1' / 'en' / 'fnn_test_case_metrics.csv'
saved = pd.read_csv(rq1_path)
check('RQ1 saved CSV has required columns', {'forecast_case_id', 'final_72h_error_km'}.issubset(saved.columns), list(saved.columns))
if {'forecast_case_id', 'final_72h_error_km'}.issubset(saved.columns):
    check('RQ1 saved test IDs unique', saved['forecast_case_id'].is_unique)
    local = meta.iloc[test_idx][['forecast_case_id']].copy()
    local['rq3_72h_error_km'] = errors[test_idx, 71]
    matched = local.merge(saved[['forecast_case_id', 'final_72h_error_km']], on='forecast_case_id', how='outer', indicator=True, validate='one_to_one')
    check('RQ1 and RQ3 test case ID sets identical', matched['_merge'].eq('both').all(), matched['_merge'].value_counts().to_dict())
    if matched['_merge'].eq('both').all():
        matched['absolute_difference_km'] = (matched['rq3_72h_error_km'] - matched['final_72h_error_km']).abs()
        maxdiff = float(matched['absolute_difference_km'].max())
        check('RQ1 vs RQ3 per-case error tolerance < 1e-8 km', np.isfinite(maxdiff) and maxdiff < 1e-8, f'max absolute difference = {maxdiff:.3e} km')
    matched.to_csv(out / 'rq3_case_id_error_audit.csv', index=False)

# Independently recompute the conformal thresholds and coverage.
cal = meta.iloc[cal_idx].copy()
cal['error_km'] = errors[cal_idx, 71]
test = meta.iloc[test_idx].copy()
test['error_km'] = errors[test_idx, 71]
expected_regimes = ['Kuroshio Core', 'Other Offshore', 'Shelf', 'Transition']
expected_methods = ['Global case CP', 'Mondrian case CP', 'Regime drifter-max CP']
check('Expected regime/method comparison rows', len(result) == 12 and set(result['method']) == set(expected_methods) and set(result['regime']) == set(expected_regimes))
q_global = qcp(cal['error_km'])
check('Global 72h conformal radius matches original', np.isclose(q_global, result.loc[result.method.eq('Global case CP'), 'radius_km']).all(), f'{q_global:.6f} km')

case_rows = []
drifter_rows = []
quantile_rows = []
for regime in expected_regimes:
    c = cal.loc[cal.regime.eq(regime)]
    t = test.loc[test.regime.eq(regime)]
    group_max = c.groupby('ID')['error_km'].max()
    expected = {
        'Global case CP': q_global,
        'Mondrian case CP': qcp(c['error_km']),
        'Regime drifter-max CP': qcp(group_max),
    }
    quantile_rows.append({'regime': regime, 'n_cal_cases': len(c), 'n_cal_drifters': c['ID'].nunique(), 'n_test_cases': len(t), 'n_test_drifters': t['ID'].nunique(), 'global_radius_km': expected['Global case CP'], 'mondrian_radius_km': expected['Mondrian case CP'], 'drifter_max_radius_km': expected['Regime drifter-max CP']})
    for method, radius in expected.items():
        row = result.loc[result.regime.eq(regime) & result.method.eq(method)]
        check(f'Radius and sample counts: {regime} / {method}', len(row) == 1 and np.isclose(float(row.radius_km.iloc[0]), radius, rtol=0, atol=1e-9) and int(row.n_cal_cases.iloc[0]) == len(c) and int(row.n_cal_drifters.iloc[0]) == c.ID.nunique() and int(row.n_test_cases.iloc[0]) == len(t) and int(row.n_test_drifters.iloc[0]) == t.ID.nunique())
        covered = t.error_km.le(radius)
        case_cov = float(covered.mean())
        check(f'Case coverage: {regime} / {method}', np.isclose(float(row.coverage.iloc[0]), case_cov, atol=1e-12))
        case_rows.append({'regime': regime, 'method': method, 'n_test_cases': len(t), 'n_test_drifters': t.ID.nunique(), 'radius_km': radius, 'case_level_coverage': case_cov})
        # An entire drifter is covered within a regime only if EVERY test case is covered.
        max_by_drifter = t.groupby('ID')['error_km'].max()
        drifter_rows.append({'regime': regime, 'method': method, 'n_test_drifters': len(max_by_drifter), 'n_test_cases': len(t), 'radius_km': radius, 'all_cases_covered_drifters': int(max_by_drifter.le(radius).sum()), 'drifter_level_all_case_coverage': float(max_by_drifter.le(radius).mean()), 'interpretation': 'All observed 72h endpoint cases for a test drifter within this regime are covered; descriptive, not a formal guarantee.'})

pd.DataFrame(quantile_rows).to_csv(out / 'rq3_audit_calibration_quantiles.csv', index=False)
pd.DataFrame(case_rows).to_csv(out / 'rq3_audit_case_level_coverage.csv', index=False)
drifter_df = pd.DataFrame(drifter_rows)
drifter_df.to_csv(out / 'rq3_audit_drifter_level_coverage.csv', index=False)

# Same drifter may appear in multiple regimes; do not sum regime-level drifter counts.
print('\n=== DRIFTER-LEVEL ALL-CASE COVERAGE (within each regime) ===')
print(drifter_df[['regime', 'method', 'n_test_drifters', 'all_cases_covered_drifters', 'drifter_level_all_case_coverage']].to_string(index=False, float_format=lambda v: f'{v:.3f}'))
print('\nNOTE: regime drifter counts overlap. Case-level and all-case drifter-level coverage are different estimands.')
print('NOTE: no finite-sample guarantee is claimed without drifter-cluster exchangeability and comparable sampling.')

check_df = pd.DataFrame(checks)
check_df.to_csv(out / 'rq3_integrity_audit_checks.csv', index=False)
print(f'\nAudit: {int(check_df.passed.sum())}/{len(check_df)} checks passed.')
print('Saved files under:', out)
for name in ['rq3_integrity_audit_checks.csv', 'rq3_case_id_error_audit.csv', 'rq3_audit_calibration_quantiles.csv', 'rq3_audit_case_level_coverage.csv', 'rq3_audit_drifter_level_coverage.csv']:
    print(' -', name)
if not check_df.passed.all():
    raise SystemExit('AUDIT FAILED: review FAIL lines and output CSV before reporting results.')
print('AUDIT PASSED: numerical and data-integrity checks completed; statistical assumptions still require discussion.')
