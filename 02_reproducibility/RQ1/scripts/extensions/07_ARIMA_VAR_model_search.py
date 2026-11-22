#!/usr/bin/env python
# coding: utf-8

# # 07 ARIMA VAR Model Search
# 
# Explores higher-order autoregressive and VAR specifications before retaining pooled VAR(1) as the stable dynamic benchmark.
# 
# **Provenance.** This is an English-facing execution copy of the original project notebook. The computational cells are preserved so that the analysis remains reproducible. Chinese narrative cells have been replaced by concise English stage notes; the original notebook remains unchanged in the source project.

# ## Analysis stage 1
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[1]:


from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data/processed').exists())
OUT = PROJECT_ROOT / 'output/rq1/var_ar_search'
print('Project:', PROJECT_ROOT)
print('Output:', OUT)


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


# Implementation note translated in the organised English notebook.
# %run ./ar_var_model_search.py


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


coarse = pd.read_csv(OUT / 'ar_var_grouped_cv_results.csv')
display(coarse.sort_values('mean_rmse_72h_km').head(15).round(3))

family_best = (coarse.sort_values('mean_rmse_72h_km')
               .groupby('family', as_index=False).first())
display(family_best[['family','p','d','alpha','mean_rmse_72h_km','mean_median_72h_km']].round(3))


# In[ ]:


fig, ax = plt.subplots(figsize=(9, 4.8))
for family, part in coarse.groupby('family'):
    best_by_p = part.loc[part.groupby('p')['mean_rmse_72h_km'].idxmin()].sort_values('p')
    ax.plot(best_by_p.p, best_by_p.mean_rmse_72h_km, marker='o', label=family)
ax.set(xlabel='Lag order p', ylabel='Grouped-CV 72-h endpoint RMSE (km)',
       title='Coarse model search (training partition only)')
ax.grid(alpha=.25); ax.legend(); plt.show()


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


refined = pd.read_csv(OUT / 'var_refined_grouped_cv_results.csv')
final_refined = pd.read_csv(OUT / 'var_final_refinement_cv_results.csv')
display(final_refined.sort_values('mean_rmse_72h_km').head(12).round(3))

pivot = final_refined.pivot(index='p', columns='alpha', values='mean_rmse_72h_km')
display(pivot.round(3).style.background_gradient(cmap='Blues_r'))


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


summary = json.loads((OUT / 'best_ar_var_summary.json').read_text())
display(pd.Series(summary, name='value').to_frame())

old = {'Model':'Original pooled VAR(1)', '72-h RMSE (km)':77.89,
       '72-h median (km)':55.89, 'Median ITE (km h)':1895.0}
new = {'Model':'Selected regularised VAR(20)',
       '72-h RMSE (km)':summary['haversine_test_rmse_72h_km'],
       '72-h median (km)':summary['haversine_test_median_72h_km'],
       'Median ITE (km h)':summary['median_ite_kmh']}
comparison = pd.DataFrame([old,new])
comparison['RMSE improvement vs VAR(1)'] = [np.nan,
    100*(old['72-h RMSE (km)']-new['72-h RMSE (km)'])/old['72-h RMSE (km)']]
display(comparison.round(2))


# In[ ]:


lead = pd.read_csv(OUT / 'best_ar_var_test_metrics_by_lead.csv')
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(lead.lead_time_hours, lead.median_error_km, lw=2.5, label='Median separation error')
ax.plot(lead.lead_time_hours, lead.rmse_km, lw=2.5, label='RMSE')
ax.set(xlabel='Lead time (h)', ylabel='Error (km)',
       title='Held-out test performance of selected VAR(20)')
ax.grid(alpha=.25); ax.legend(); plt.show()


# ## Analysis stage 8
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.
