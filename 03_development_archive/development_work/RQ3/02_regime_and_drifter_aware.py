"""RQ3 extension: Mondrian case-level CP and conservative drifter-max CP.
Run from DATA3001 root: python RQ3/02_regime_and_drifter_aware.py
Requires original 01_global_conformal.py in the same folder.
"""
from pathlib import Path
import runpy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
base = runpy.run_path(str(ROOT / '01_global_conformal.py'))
origins = base['origins']; errors = base['errors_km']
cal_idx = base['cal_idx']; test_idx = base['test_idx']
project = base['PROJECT_ROOT']
out = ROOT / 'output'; out.mkdir(exist_ok=True)
ALPHA = 0.10
R = 6371.0

def qcp(values, alpha=ALPHA):
    v = np.asarray(values, dtype=float)
    if not len(v) or not np.isfinite(v).all():
        raise ValueError('Missing or non-finite calibration scores')
    k = int(np.ceil((len(v) + 1) * (1 - alpha)))
    return float(np.inf if k > len(v) else np.partition(v, k - 1)[k - 1])

def area(radius):
    return float(2*np.pi*R**2*(1-np.cos(radius/R))) if np.isfinite(radius) else float(4*np.pi*R**2)

regimes = pd.read_csv(project / 'RQ2' / 'data' / 'forecast_case_regimes.csv')
regimes = regimes[['forecast_case_id', 'regime']]
assert regimes['forecast_case_id'].is_unique, 'Duplicate regime labels'
meta = origins[['forecast_case_id','ID','partition']].merge(regimes, on='forecast_case_id', how='left', validate='one_to_one')
assert len(meta)==len(origins) and meta['regime'].notna().all()
assert meta['forecast_case_id'].equals(origins['forecast_case_id']), 'Ordering changed'
assert set(meta.iloc[cal_idx]['partition']) == {'calibration'}
assert set(meta.iloc[test_idx]['partition']) == {'test'}

# Primary analysis: 72-hour endpoint, fixed 90% target, groupwise case scores.
cal = meta.iloc[cal_idx].copy(); cal['score'] = errors[cal_idx,71]
test = meta.iloc[test_idx].copy(); test['score'] = errors[test_idx,71]
q_global = qcp(cal['score'])
rows=[]
for regime in sorted(test['regime'].unique()):
    c = cal[cal['regime']==regime]
    t = test[test['regime']==regime]
    if c.empty: raise ValueError(f'No calibration cases for {regime}')
    q_group = qcp(c['score'])
    # Conservative drifter-max calibration: one maximum 72h error per calibration drifter.
    # This targets a NEW DRIFTER's maximum across its cases in a given regime, conditional
    # on cluster exchangeability and a comparable case-sampling process; it is not a
    # distribution-free guarantee for the present observational dataset.
    max_scores = c.groupby('ID')['score'].max().to_numpy()
    q_drifter = qcp(max_scores)
    for method,q in [('Global case CP',q_global),('Mondrian case CP',q_group),('Regime drifter-max CP',q_drifter)]:
        rows.append(dict(regime=regime,method=method,n_cal_cases=len(c),n_cal_drifters=c.ID.nunique(),
            n_test_cases=len(t),n_test_drifters=t.ID.nunique(),radius_km=q,
            area_km2=area(q),coverage=float((t.score<=q).mean())))
result = pd.DataFrame(rows)
result.to_csv(out/'rq3_72h_regime_comparison.csv',index=False)
print('\n72h REGIME COMPARISON (90% nominal)\n')
print(result.to_string(index=False,float_format=lambda x:f'{x:.3f}'))

# Overall coverage and case-weighted mean area for each method.
overall=[]
for method,g in result.groupby('method'):
    weights=g.n_test_cases.to_numpy()
    overall.append(dict(method=method,coverage=np.average(g.coverage,weights=weights),
        mean_area_km2=np.average(g.area_km2,weights=weights),
        mean_radius_km=np.average(g.radius_km,weights=weights),
        min_regime_coverage=g.coverage.min(),max_regime_coverage=g.coverage.max()))
overall=pd.DataFrame(overall)
overall.to_csv(out/'rq3_72h_overall_comparison.csv',index=False)
print('\nOVERALL COMPARISON\n',overall.to_string(index=False,float_format=lambda x:f'{x:.3f}'))

# Drifter-max diagnostic: fraction of test drifters whose ALL cases in a regime are covered.
cluster=[]
for regime in sorted(test.regime.unique()):
    t=test[test.regime==regime]
    worst=t.groupby('ID')['score'].max()
    q=float(result[(result.regime==regime)&(result.method=='Regime drifter-max CP')].radius_km.iloc[0])
    cluster.append(dict(regime=regime,n_test_drifters=len(worst),
                        all_cases_covered=float((worst<=q).mean()),radius_km=q))
pd.DataFrame(cluster).to_csv(out/'rq3_drifter_max_coverage.csv',index=False)

fig,ax=plt.subplots(figsize=(10,5))
labels=sorted(test.regime.unique()); methods=['Global case CP','Mondrian case CP','Regime drifter-max CP']
x=np.arange(len(labels)); w=.26
for j,method in enumerate(methods):
    g=result[result.method==method].set_index('regime').loc[labels]
    ax.bar(x+(j-1)*w,g.coverage,w,label=method)
ax.axhline(.9,ls='--',color='black',label='90% target')
ax.set_xticks(x,labels,rotation=15);ax.set_ylim(0,1.08);ax.set_ylabel('72h empirical coverage')
ax.legend(fontsize=8);fig.tight_layout()
fig.savefig(out/'rq3_regime_coverage_comparison.png',dpi=180)
plt.close(fig)
print('\nSaved outputs in:',out)
