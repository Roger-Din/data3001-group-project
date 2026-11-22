#!/usr/bin/env python3
"""DATA3001 RQ2 Step 17: publication-ready figures from Steps 14–16 CSV outputs.
Run from any directory: python RQ2/17_final_figures.py
Requires: pandas, numpy, matplotlib. Does NOT train models or change data.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'RQ2' / 'data'
OUT = ROOT / 'RQ2' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
REGIMES = ['Shelf', 'Kuroshio Core', 'Transition', 'Other Offshore']
MODELS = ['Persistence', 'Mean Flow', 'Ridge', 'FNN', 'VAR(1)', 'KNN']
COLORS = {'Shelf':'#3C78A8', 'Kuroshio Core':'#C84C47',
          'Transition':'#D6A33A', 'Other Offshore':'#598C70'}
MODEL_COLORS = dict(zip(MODELS, ['#777777','#C08B46','#6C83B5','#B34549','#8875A7','#3A968D']))
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'figure.dpi':130,
                     'savefig.dpi':250, 'axes.titleweight':'semibold'})

def load(name, columns):
    p = DATA/name
    if not p.is_file():
        raise FileNotFoundError(f'Missing {p}. Run its preceding step first.')
    d = pd.read_csv(p)
    missing = set(columns)-set(d.columns)
    if missing:
        raise ValueError(f'{p.name} missing columns: {sorted(missing)}')
    return d

def save(fig, stem):
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'{stem}.{ext}', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'Saved: {OUT/stem}.png  +  .pdf')

raw = load('rq2_test_model_errors_long.csv',
           ['forecast_case_id','ID','regime','model','origin_longitude','origin_latitude','final_72h_error_km'])
case = raw.drop_duplicates('forecast_case_id').copy()
assert len(case)==757 and case['ID'].nunique()==71, 'Unexpected test case/drifter counts'
assert set(case['regime'])==set(REGIMES), 'Unexpected regime labels'
assert len(raw)==757*6 and set(raw.model)==set(MODELS), 'Unexpected model rows'

# Figure 1. Forecast-origin locations (NOT a full ocean-regime raster).
fig, ax = plt.subplots(figsize=(8.2,6.4))
for r in REGIMES[::-1]:
    d = case.loc[case.regime.eq(r)]
    ax.scatter(d.origin_longitude, d.origin_latitude, s=21, alpha=.76,
               c=COLORS[r], label=f'{r} (n={len(d)})', edgecolors='white', linewidths=.25)
ax.set(xlabel='Longitude (°E)', ylabel='Latitude (°N)',
       title='Test forecast origins by ocean regime')
ax.set_xlim(125,135); ax.set_ylim(25,35)
ax.set_aspect('equal', adjustable='box')
ax.grid(alpha=.16)
ax.legend(loc='lower left', frameon=True, fontsize=9)
fig.text(.13,.01,'Dots show test-case origins only; they are not the full spatial regime boundaries.', fontsize=8, color='#555555')
save(fig,'17_fig1_test_origins_by_regime')

# Figure 2. Regime-wise model median 72-hour endpoint error with cluster bootstrap 95% CI.
summary = load('rq2_cluster_bootstrap_summary.csv',
               ['regime','model','median_72h_km','median_ci_lower_km','median_ci_upper_km'])
fig, axes = plt.subplots(2,2,figsize=(12,8.4), sharey=True)
for ax, r in zip(axes.flat,REGIMES):
    d=summary.loc[summary.regime.eq(r)].set_index('model').reindex(MODELS)
    if d[['median_72h_km','median_ci_lower_km','median_ci_upper_km']].isna().any().any():
        raise ValueError(f'Missing model median/CI for {r}')
    x=np.arange(len(MODELS))
    y=d.median_72h_km.to_numpy(float)
    low=d.median_ci_lower_km.to_numpy(float)
    high=d.median_ci_upper_km.to_numpy(float)
    ax.bar(x,y,color=[MODEL_COLORS[m] for m in MODELS],width=.72,alpha=.88)
    ax.errorbar(x,y,yerr=np.vstack([np.maximum(y-low,0),np.maximum(high-y,0)]),
                fmt='none',ecolor='#262626',elinewidth=1.1,capsize=3)
    ax.set_xticks(x,MODELS,rotation=35,ha='right')
    ax.set_title(f'{r}  (n={int(d.n_cases.iloc[0])} cases)' if 'n_cases' in d else r)
    ax.grid(axis='y',alpha=.18)
    ax.set_axisbelow(True)
for ax in axes[:,0]: ax.set_ylabel('Median 72-hour endpoint error (km)')
fig.suptitle('Forecast error across ocean regimes and models',fontsize=15,y=1.01)
fig.text(.5,-.005,'Error bars: 95% drifter-clustered bootstrap percentile intervals (2,000 replicates).',
         ha='center',fontsize=9,color='#555555')
fig.tight_layout()
save(fig,'17_fig2_model_error_by_regime')

# Figure 3. Mean paired advantage (comparator error minus FNN error) and cluster CIs.
adv = load('rq2_model_advantage_by_regime.csv',
           ['regime','comparator','mean_advantage_km','ci_lower_km','ci_upper_km'])
comparators=['Persistence','VAR(1)','KNN']
fig, axes=plt.subplots(1,3,figsize=(14.2,5.1),sharey=False)
for ax,comp in zip(axes,comparators):
    d=adv.loc[adv.comparator.eq(comp)].set_index('regime').reindex(REGIMES)
    if d[['mean_advantage_km','ci_lower_km','ci_upper_km']].isna().any().any():
        raise ValueError(f'Missing paired advantage rows for {comp}')
    y=np.arange(len(REGIMES))
    mid=d.mean_advantage_km.to_numpy(float)
    low=d.ci_lower_km.to_numpy(float)
    high=d.ci_upper_km.to_numpy(float)
    ax.axvline(0,color='#444444',lw=1,ls='--')
    ax.errorbar(mid,y,xerr=np.vstack([np.maximum(mid-low,0),np.maximum(high-mid,0)]),
                fmt='none',ecolor='#333333',capsize=4,lw=1.5)
    ax.scatter(mid,y,s=58,c=[COLORS[r] for r in REGIMES],zorder=4)
    ax.set_yticks(y,REGIMES)
    ax.invert_yaxis()
    ax.set_title(f'FNN vs {comp}')
    ax.grid(axis='x',alpha=.17)
    ax.set_xlabel('Mean paired advantage (km)')
fig.suptitle('FNN improvement over selected comparators',fontsize=15,y=1.04)
fig.text(.5,-.02,'Positive values favour FNN; error bars are 95% drifter-clustered bootstrap CIs.',
         ha='center',fontsize=9,color='#555555')
fig.tight_layout()
save(fig,'17_fig3_fnn_paired_advantage')

# Figure 4. Sensitivity: unique speed-threshold comparisons, no redundant buffer copies.
sens=load('rq2_sensitivity_core_shelf_contrasts.csv',
          ['speed_threshold_m_s','buffer_km','model','metric','core_minus_shelf_km','ci_lower_km','ci_upper_km'])
d=sens.loc[sens.model.eq('FNN') & sens.metric.eq('median')].copy()
if d.empty: raise ValueError('No FNN median sensitivity comparisons')
# Buffer affects Transition/Other Offshore, not Core or Shelf. Check rather than assume.
for threshold, grp in d.groupby('speed_threshold_m_s'):
    for col in ['core_minus_shelf_km','ci_lower_km','ci_upper_km']:
        if not np.allclose(grp[col],grp[col].iloc[0],atol=1e-8,rtol=1e-8):
            raise ValueError(f'Buffer changes {col} at speed {threshold}; cannot collapse duplicates')
d=d.sort_values('buffer_km').drop_duplicates('speed_threshold_m_s').sort_values('speed_threshold_m_s')
x=d.speed_threshold_m_s.to_numpy(float)
y=d.core_minus_shelf_km.to_numpy(float)
lo=d.ci_lower_km.to_numpy(float)
hi=d.ci_upper_km.to_numpy(float)
fig,ax=plt.subplots(figsize=(8.2,5.5))
ax.axhline(0,ls='--',color='#555555',lw=1)
ax.errorbar(x,y,yerr=np.vstack([np.maximum(y-lo,0),np.maximum(hi-y,0)]),
            fmt='o-',color=MODEL_COLORS['FNN'],ecolor='#333333',lw=2,markersize=7,capsize=6)
for xi,yi in zip(x,y): ax.annotate(f'{yi:.2f} km',(xi,yi),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
ax.set_xticks(x,[f'{v:.1f}' for v in x])
ax.set(xlabel='Climatological current-speed threshold (m/s)',
       ylabel='Core − Shelf median endpoint error (km)',
       title='Sensitivity of the FNN Core–Shelf error gap')
ax.set_ylim(min(0,lo.min()-5),hi.max()+12)
ax.grid(axis='y',alpha=.18)
fig.text(.13,.01,'95% drifter-clustered bootstrap CIs; buffer distance does not alter Core or Shelf membership.',
         fontsize=8,color='#555555')
save(fig,'17_fig4_core_shelf_threshold_sensitivity')

# Optional check: what changes when Transition buffer is varied?
counts_path=DATA/'rq2_sensitivity_regime_counts.csv'
if counts_path.exists():
    counts=pd.read_csv(counts_path)
    print('Sensitivity count columns:',list(counts.columns))
print('\n=== STEP 17 COMPLETE: 4 figures, PNG and PDF each ===')
print('Figures:',OUT)
print('Note: Fig 1 shows test origins, not full gridded physical regime boundaries.')
print('Note: These are descriptive/uncorrected bootstrap intervals; avoid causal claims.')
