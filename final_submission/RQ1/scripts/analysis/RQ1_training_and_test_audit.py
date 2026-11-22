#!/usr/bin/env python
# coding: utf-8

# # RQ1 Training And Test Audit
# 
# Audits the leakage-safe training, calibration and held-out test partitions used throughout RQ1.
# 
# **Provenance.** This is an English-facing execution copy of the original project notebook. The computational cells are preserved so that the analysis remains reproducible. Chinese narrative cells have been replaced by concise English stage notes; the original notebook remains unchanged in the source project.

# ## Analysis stage 1
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


from pathlib import Path 

import matplotlib .pyplot as plt 
import numpy as np 
import pandas as pd 
from IPython .display import display 

PROJECT_ROOT =next (
(p for p in [Path .cwd (),*Path .cwd ().parents ]if (p /'data'/'processed').exists ()),
None ,
)
if PROJECT_ROOT is None :
    raise FileNotFoundError ('\u627e\u4e0d\u5230 data/processed；\u8bf7\u4ece data3001-project-main \u5185\u6253\u5f00\u672c notebook。')

PROCESSED =PROJECT_ROOT /'data'/'processed'
RESULTS =PROJECT_ROOT /'output'/'rq1'/'en'
plt .rcParams ['font.sans-serif']=['PingFang SC','Arial Unicode MS','DejaVu Sans']
plt .rcParams ['axes.unicode_minus']=False 
print ('\u9879\u76ee\u76ee\u5f55：',PROJECT_ROOT .resolve ())
print ('\u8bfb\u53d6\u7ed3\u679c：',RESULTS .resolve ())


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


origins =pd .read_parquet (PROCESSED /'forecast_origins.parquet').sort_values ('forecast_case_id')
splits =pd .read_csv (PROCESSED /'drifter_splits.csv')

assigned =splits .groupby ('partition')['ID'].nunique ()
usable =origins .groupby ('partition')['ID'].nunique ()
cases =origins .groupby ('partition')['forecast_case_id'].nunique ()
partition_table =pd .DataFrame ({
'assigned_drifters':assigned ,
'usable_drifters':usable ,
'forecast_cases':cases ,
}).reindex (['training','calibration','test'])
partition_table .loc ['Total']=partition_table .sum ()
display (partition_table .astype (int ))

id_sets ={p :set (splits .loc [splits ['partition'].eq (p ),'ID'])for p in ['training','calibration','test']}
assert id_sets ['training'].isdisjoint (id_sets ['calibration'])
assert id_sets ['training'].isdisjoint (id_sets ['test'])
assert id_sets ['calibration'].isdisjoint (id_sets ['test'])
assert tuple (cases .reindex (['training','calibration','test']))==(3591 ,769 ,757 )
print ('\u68c0\u67e5\u901a\u8fc7：\u4e09\u4e2a partition \u7684 drifter ID \u5b8c\u5168\u4e0d\u91cd\u53e0。')


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


fig, ax = plt.subplots(figsize=(12, 3.8))
ax.axis('off')
boxes = [
    (0.02, 0.32, 0.21, 0.42, '#eee6ff', 'Candidate settings\n(alpha / architecture / K,p)'),
    (0.28, 0.32, 0.16, 0.42, '#fff4d8', 'Grouped 5-fold CV\ntraining drifters only'),
    (0.49, 0.32, 0.16, 0.42, '#e4f6ef', 'Select configuration\nlowest validation RMSE'),
    (0.70, 0.32, 0.13, 0.42, '#dff1ff', 'Final refit\nall 3,591 train cases'),
    (0.88, 0.32, 0.10, 0.42, '#ffe8ee', 'Held-out test\n757 cases'),
]
for x, y, w, h, colour, text in boxes:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=colour, edgecolor='#2aa7df', lw=1.5))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10)
for x1, x2 in [(0.23, 0.28), (0.44, 0.49), (0.65, 0.70), (0.83, 0.88)]:
    ax.annotate('', xy=(x2, 0.53), xytext=(x1, 0.53), arrowprops={'arrowstyle': '-|>', 'lw': 1.5})
ax.text(0.5, 0.9, 'Cases from one drifter never cross folds or partitions', ha='center', fontsize=12, color='#274c66')
ax.text(0.5, 0.12, 'The 769-case calibration partition is not used in RQ1', ha='center', fontsize=11, color='#8e44ad')
plt.show()


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


ridge_cv =pd .read_csv (RESULTS /'ridge_cross_validation_summary.csv')
best_ridge =ridge_cv .loc [ridge_cv ['mean_cv_rmse_km'].idxmin ()]
display (ridge_cv .style .format (precision =3 ))

fig ,ax =plt .subplots (figsize =(8 ,4 ))
ax .errorbar (ridge_cv ['alpha'],ridge_cv ['mean_cv_rmse_km'],yerr =ridge_cv ['std_cv_rmse_km'],marker ='o',capsize =4 )
ax .scatter (best_ridge ['alpha'],best_ridge ['mean_cv_rmse_km'],marker ='*',s =180 ,color ='#d81b60',zorder =5 ,label =f"Selected alpha={best_ridge ['alpha']:g}")
ax .set_xscale ('log')
ax .set (xlabel ='Ridge alpha',ylabel ='Mean validation displacement RMSE (km)',title ='Training-only grouped CV')
ax .grid (alpha =.25 );ax .legend (frameon =False );plt .show ()
print ('\u9009\u62e9\u7ed3\u679c：alpha =',best_ridge ['alpha'])


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


fnn_cv =pd .read_csv (RESULTS /'fnn_cross_validation_summary.csv')
best_fnn =fnn_cv .loc [fnn_cv ['mean_cv_rmse_km'].idxmin ()]
display (fnn_cv .style .format (precision =3 ))

fig ,ax =plt .subplots (figsize =(8 ,4 ))
ax .bar (fnn_cv ['config'],fnn_cv ['mean_cv_rmse_km'],yerr =fnn_cv ['std_cv_rmse_km'],capsize =5 ,color =['#9ecae1','#6baed6','#2171b5'])
ax .set (ylabel ='Mean validation displacement RMSE (km)',title ='FNN configuration selection: training folds only')
ax .tick_params (axis ='x',rotation =10 );ax .grid (axis ='y',alpha =.25 );plt .show ()
print ('\u9009\u62e9\u7ed3\u679c：',best_fnn ['config'],' hidden=',best_fnn ['hidden_sizes'],' L2=',best_fnn ['l2'],' epochs=',int (best_fnn ['median_best_epoch']))


# ## Analysis stage 8
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


knn_cv = pd.read_csv(RESULTS / 'knn_cross_validation_summary.csv')
best_knn = knn_cv.loc[knn_cv['mean_cv_rmse_km'].idxmin()]
display(knn_cv.sort_values('mean_cv_rmse_km').style.format(precision=3))

fig, ax = plt.subplots(figsize=(8, 4))
for power, group in knn_cv.groupby('distance_power'):
    group = group.sort_values('k')
    ax.errorbar(group['k'], group['mean_cv_rmse_km'], yerr=group['std_cv_rmse_km'], marker='o', capsize=4, label=f'p={power:g}')
ax.scatter(best_knn['k'], best_knn['mean_cv_rmse_km'], marker='*', s=180, color='#d81b60', zorder=5, label=f"Selected K={int(best_knn['k'])}, p={best_knn['distance_power']:g}")
ax.set_xscale('log'); ax.set_xticks([5,10,20,50,100], labels=['5','10','20','50','100'])
ax.set(xlabel='K', ylabel='Mean validation displacement RMSE (km)', title='KNN selection: training folds only')
ax.grid(alpha=.25); ax.legend(frameon=False); plt.show()


# ## Analysis stage 9
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


model_files = {
    'Velocity persistence': 'velocity_persistence_summary.csv',
    'Spatial mean flow': 'spatial_mean_flow_summary.csv',
    'Pooled VAR(1)': 'var1_summary.csv',
    'Ridge': 'ridge_summary.csv',
    'Weighted KNN': 'knn_summary.csv',
    'FNN': 'fnn_summary.csv',
}
rows = []
for model, filename in model_files.items():
    frame = pd.read_csv(RESULTS / filename)
    values = dict(zip(frame.iloc[:, 0], frame.iloc[:, 1]))
    rows.append({
        'Model': model,
        '24h median': float(values['24h median error (km)']),
        '48h median': float(values['48h median error (km)']),
        '72h median': float(values['72h median error (km)']),
        '72h RMSE': float(values['72h RMSE (km)']),
        'Median ITE': float(values['Median ITE (km·h)']),
    })
heldout = pd.DataFrame(rows).sort_values('72h RMSE')
display(heldout.style.format({c: '{:.2f}' for c in heldout.columns if c != 'Model'}))


# In[ ]:


curves = pd.read_csv(RESULTS / 'rq1_six_model_comparison.csv')
line_spec = [
    ('persistence_median_km', 'Velocity persistence', '#636b73', '--'),
    ('mean_flow_median_km', 'Spatial mean flow', '#9aa1a8', ':'),
    ('var1_median_km', 'Pooled VAR(1)', '#dd7f00', '-'),
    ('ridge_median_km', 'Ridge', '#2864dc', '-'),
    ('knn_median_km', 'Weighted KNN', '#159566', '-'),
    ('fnn_median_km', 'FNN', '#d81b60', '-'),
]
fig, ax = plt.subplots(figsize=(10, 6))
for column, label, colour, style in line_spec:
    ax.plot(curves['lead_time_hours'], curves[column], label=label, color=colour, linestyle=style, linewidth=2.4)
ax.set(xlabel='Lead time (h)', ylabel='Median test separation error (km)', title='All points are summaries of the same 757 held-out test cases')
ax.set_xlim(1,72); ax.grid(alpha=.25); ax.legend(ncol=2, frameon=False); plt.show()


# ## Analysis stage 10
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# ## Analysis stage 11
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


test_case_ids = origins.loc[origins['partition'].eq('test'), ['forecast_case_id', 'ID']]
case_files = {
    'Velocity persistence': 'velocity_persistence_test_case_metrics.csv',
    'Spatial mean flow': 'spatial_mean_flow_test_case_metrics.csv',
    'Pooled VAR(1)': 'var1_test_case_metrics.csv',
    'Ridge': 'ridge_test_case_metrics.csv',
    'Weighted KNN': 'knn_test_case_metrics.csv',
    'FNN': 'fnn_test_case_metrics.csv',
}

def clustered_ci(frame, n_boot=2000, seed=3001):
    rng = np.random.default_rng(seed)
    groups = {drifter: g['final_72h_error_km'].to_numpy() for drifter, g in frame.groupby('ID')}
    ids = np.array(list(groups))
    estimates = np.empty((n_boot, 2))
    for b in range(n_boot):
        sample_ids = rng.choice(ids, size=len(ids), replace=True)
        errors = np.concatenate([groups[i] for i in sample_ids])
        estimates[b] = [np.median(errors), np.sqrt(np.mean(errors**2))]
    point = frame['final_72h_error_km'].to_numpy()
    return {
        '72h median': np.median(point),
        'median CI low': np.quantile(estimates[:,0], .025),
        'median CI high': np.quantile(estimates[:,0], .975),
        '72h RMSE': np.sqrt(np.mean(point**2)),
        'RMSE CI low': np.quantile(estimates[:,1], .025),
        'RMSE CI high': np.quantile(estimates[:,1], .975),
    }

ci_rows = []
for j, (model, filename) in enumerate(case_files.items()):
    metrics = pd.read_csv(RESULTS / filename).merge(test_case_ids, on='forecast_case_id', how='left', validate='one_to_one')
    result = clustered_ci(metrics, seed=3001+j)
    result['Model'] = model
    ci_rows.append(result)
ci_table = pd.DataFrame(ci_rows).sort_values('72h RMSE')
display(ci_table[['Model','72h median','median CI low','median CI high','72h RMSE','RMSE CI low','RMSE CI high']].style.format(precision=2))


# ## Analysis stage 12
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.
