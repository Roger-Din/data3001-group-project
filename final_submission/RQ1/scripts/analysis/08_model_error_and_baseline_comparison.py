#!/usr/bin/env python
# coding: utf-8

# # 08 Model Error And Baseline Comparison
# 
# Compares lead-time error curves and endpoint metrics across candidate models and physical baselines.
# 
# **Provenance.** This is an English-facing execution copy of the original project notebook. The computational cells are preserved so that the analysis remains reproducible. Chinese narrative cells have been replaced by concise English stage notes; the original notebook remains unchanged in the source project.

# ## Analysis stage 1
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[1]:


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'output/rq1').exists())
RQ1 = PROJECT_ROOT/'output/rq1/zh'
VAR20 = PROJECT_ROOT/'output/rq1/var_ar_search'
OUT = PROJECT_ROOT/'output/rq1/model_error_analysis'
OUT.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    'Velocity persistence':'velocity_persistence',
    'Spatial mean flow':'spatial_mean_flow',
    'Ridge':'ridge', 'FNN':'fnn', 'Weighted KNN':'knn', 'VAR(1)':'var1',
}
BASELINES = ['Velocity persistence','Spatial mean flow']
COLORS = {'Velocity persistence':'#6B7280','Spatial mean flow':'#9CA3AF',
          'Ridge':'#2563EB','FNN':'#D81B60','Weighted KNN':'#1B9E77',
          'VAR(1)':'#E68600','VAR(20)':'#6F42C1'}
print('Project:', PROJECT_ROOT)


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


lead_frames =[];case_frames =[]
for model ,prefix in MODEL_FILES .items ():
    lead =pd .read_csv (RQ1 /f'{prefix }_lead_metrics.csv')
    lead ['model']=model ;lead_frames .append (lead )
    case =pd .read_csv (RQ1 /f'{prefix }_test_case_metrics.csv')
    case ['model']=model ;case_frames .append (case )

    # Implementation note translated in the organised English notebook.
v20_raw =pd .read_csv (VAR20 /'best_ar_var_test_predictions.csv').sort_values (['forecast_case_id','lead_time_hours'])
v20_lead =(v20_raw .groupby ('lead_time_hours').separation_error_km 
.agg (n ='size',median_error_km ='median',mean_error_km ='mean',
q25_error_km =lambda s :s .quantile (.25 ),q75_error_km =lambda s :s .quantile (.75 ),
rmse_km =lambda s :np .sqrt (np .mean (np .square (s ))))
.reset_index ())
v20_lead ['model']='VAR(20)';lead_frames .append (v20_lead )
v20_case =(v20_raw .groupby ('forecast_case_id').separation_error_km 
.agg (final_72h_error_km ='last',mean_trajectory_error_km ='mean',
integrated_trajectory_error_km_h =lambda s :np .trapezoid (s .to_numpy (),dx =1.0 ))
.reset_index ())
v20_case ['model']='VAR(20)';case_frames .append (v20_case )

lead_metrics =pd .concat (lead_frames ,ignore_index =True )
case_metrics =pd .concat (case_frames ,ignore_index =True )
reference =set (case_frames [0 ].forecast_case_id )
audit =[]
for model ,g in case_metrics .groupby ('model'):
    ids =set (g .forecast_case_id )
    audit .append ({'model':model ,'n_cases':len (g ),'unique_cases':len (ids ),
    'same_757_case_ids':ids ==reference ,'finite':np .isfinite (g .select_dtypes ('number')).all ().all ()})
audit =pd .DataFrame (audit )
display (audit )
assert audit [['same_757_case_ids','finite']].all ().all ()and audit .n_cases .eq (757 ).all ()
lead_metrics .to_csv (OUT /'all_models_lead_metrics.csv',index =False )
case_metrics .to_csv (OUT /'all_models_test_case_metrics.csv',index =False )


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


horizon_table=[]
for model,g in lead_metrics.groupby('model'):
    row={'Model':model}
    for h in [24,48,72]:
        r=g[g.lead_time_hours.eq(h)].iloc[0]
        row[f'{h}h Median']=r.median_error_km
        row[f'{h}h RMSE']=r.rmse_km
    cm=case_metrics[case_metrics.model.eq(model)]
    row['Median 72h ITE']=cm.integrated_trajectory_error_km_h.median()
    horizon_table.append(row)
summary=pd.DataFrame(horizon_table).sort_values('72h RMSE')
display(summary.style.format({c:'{:.2f}' for c in summary.columns if c!='Model'})
        .background_gradient(subset=[c for c in summary.columns if c!='Model'],cmap='RdYlGn_r'))
summary.to_csv(OUT/'model_error_summary_24_48_72.csv',index=False)


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(14,5),sharex=True)
for model,g in lead_metrics.groupby('model'):
    g=g.sort_values('lead_time_hours'); style='--' if model in BASELINES else '-'
    axes[0].plot(g.lead_time_hours,g.median_error_km,label=model,color=COLORS[model],ls=style,lw=2.2)
    axes[1].plot(g.lead_time_hours,g.rmse_km,label=model,color=COLORS[model],ls=style,lw=2.2)
axes[0].set_title('Median separation error'); axes[1].set_title('RMSE')
for ax in axes:
    ax.set(xlabel='Lead time (h)',ylabel='Error (km)');ax.grid(alpha=.25)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.05))
fig.tight_layout();fig.savefig(OUT/'lead_time_error_curves.png',dpi=200,bbox_inches='tight');plt.show()


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


s72=summary.set_index('Model')
rows=[]
for model in s72.index:
    for baseline in BASELINES:
        rows.append({'Model':model,'Baseline':baseline,
          'RMSE difference (km)':s72.loc[model,'72h RMSE']-s72.loc[baseline,'72h RMSE'],
          'RMSE skill (%)':100*(1-s72.loc[model,'72h RMSE']/s72.loc[baseline,'72h RMSE']),
          'Median difference (km)':s72.loc[model,'72h Median']-s72.loc[baseline,'72h Median'],
          'ITE difference (km h)':s72.loc[model,'Median 72h ITE']-s72.loc[baseline,'Median 72h ITE']})
skill72=pd.DataFrame(rows)
display(skill72[~skill72.Model.isin(BASELINES)].pivot(index='Model',columns='Baseline',values='RMSE skill (%)')
        .round(2).style.background_gradient(cmap='RdYlGn',vmin=-35,vmax=35))
skill72.to_csv(OUT/'72h_skill_vs_baselines.csv',index=False)


# In[ ]:


plot_skill=skill72[(~skill72.Model.isin(BASELINES))].pivot(index='Model',columns='Baseline',values='RMSE skill (%)')
ax=plot_skill.plot(kind='barh',figsize=(9,5),color=['#5DADE2','#A9CCE3'])
ax.axvline(0,color='black',lw=1);ax.set(xlabel='Reduction in 72-h RMSE (%)',title='Forecast skill relative to physical baselines')
ax.grid(axis='x',alpha=.25);ax.legend(title='Baseline',frameon=False)
plt.tight_layout();plt.savefig(OUT/'72h_skill_vs_baselines.png',dpi=200,bbox_inches='tight');plt.show()


# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


wide=case_metrics.pivot(index='forecast_case_id',columns='model',values='final_72h_error_km').sort_index()
rng=np.random.default_rng(2026);B=5000;n=len(wide)
draws=rng.integers(0,n,size=(B,n));boot=[]
for model in wide.columns:
    if model in BASELINES:continue
    me=wide[model].to_numpy()
    for baseline in BASELINES:
        be=wide[baseline].to_numpy()
        rm=np.sqrt(np.mean(me[draws]**2,axis=1)); rb=np.sqrt(np.mean(be[draws]**2,axis=1))
        skill=100*(1-rm/rb)
        boot.append({'Model':model,'Baseline':baseline,'skill_pct':100*(1-np.sqrt(np.mean(me**2))/np.sqrt(np.mean(be**2))),
                     'ci_lower_pct':np.quantile(skill,.025),'ci_upper_pct':np.quantile(skill,.975),
                     'interpretation':'Improved' if np.quantile(skill,.025)>0 else ('Worse' if np.quantile(skill,.975)<0 else 'Inconclusive')})
bootstrap=pd.DataFrame(boot).sort_values(['Baseline','skill_pct'],ascending=[True,False])
display(bootstrap.round(2))
bootstrap.to_csv(OUT/'paired_bootstrap_skill_vs_baselines.csv',index=False)


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


ranking=summary[['Model','72h Median','72h RMSE','Median 72h ITE']].copy()
for metric in ['72h Median','72h RMSE','Median 72h ITE']:
    ranking[metric+' rank']=ranking[metric].rank(method='min').astype(int)
display(ranking.sort_values('72h RMSE rank').style.format({c:'{:.2f}' for c in ['72h Median','72h RMSE','Median 72h ITE']}))
ranking.to_csv(OUT/'model_ranking_by_metric.csv',index=False)


# ## Analysis stage 8
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.
