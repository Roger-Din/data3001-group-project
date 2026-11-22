#!/usr/bin/env python
# coding: utf-8

# # 14 Champion Models Comparison
# 
# Compares the leading RQ1 models under RMSE, median endpoint error and integrated trajectory error.
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
from IPython import get_ipython 
from IPython .display import display ,Markdown 

ip =get_ipython ()
if ip is not None :ip .run_line_magic ('matplotlib','inline')
PROJECT_ROOT =next ((p for p in [Path .cwd (),*Path .cwd ().parents ]if (p /'output'/'rq1'/'zh').exists ()),None )
if PROJECT_ROOT is None :raise FileNotFoundError ('\u627e\u4e0d\u5230output/rq1/zh。')
OUTPUT =PROJECT_ROOT /'output'/'rq1'/'zh';PROCESSED =PROJECT_ROOT /'data'/'processed'
plt .style .use ('seaborn-v0_8-whitegrid');plt .rcParams ['font.sans-serif']=['PingFang SC','Arial Unicode MS','DejaVu Sans'];plt .rcParams ['axes.unicode_minus']=False 

MODELS ={
'Boosting residual':('boosting_residual_lead_metrics.csv','boosting_residual_test_case_metrics.csv'),
'Optimised FNN':('optimised_fnn_lead_metrics.csv','optimised_fnn_test_case_metrics.csv'),
'Optimised KNN':('optimised_knn_lead_metrics.csv','optimised_knn_test_case_metrics.csv'),
'Optimised Random Forest':('optimised_random_forest_lead_metrics.csv','optimised_random_forest_test_case_metrics.csv'),
}
BASELINES ={
'Velocity persistence':('velocity_persistence_lead_metrics.csv','velocity_persistence_test_case_metrics.csv'),
'Spatial mean flow':('spatial_mean_flow_lead_metrics.csv','spatial_mean_flow_test_case_metrics.csv'),
}
COLORS ={'Boosting residual':'#d81b60','Optimised FNN':'#7b1fa2','Optimised KNN':'#009e73','Optimised Random Forest':'#0072b2','Velocity persistence':'#666666','Spatial mean flow':'#a0a7ad'}
origins =pd .read_parquet (PROCESSED /'forecast_origins.parquet',columns =['forecast_case_id','ID','partition'])
test_map =origins [origins .partition .eq ('test')][['forecast_case_id','ID']].rename (columns ={'ID':'drifter_id'})
print ('\u7ed3\u679c\u76ee\u5f55：',OUTPUT )


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def load_lead(filename,label,kind):
    d=pd.read_csv(OUTPUT/filename)[['lead_time_hours','median_error_km','rmse_km']].copy(); d['model']=label; d['kind']=kind; return d
def load_case(filename,label,kind):
    d=pd.read_csv(OUTPUT/filename).copy()
    if 'error_72h_km' not in d: d=d.rename(columns={'final_72h_error_km':'error_72h_km'})
    if 'ite_km_h' not in d: d=d.rename(columns={'integrated_trajectory_error_km_h':'ite_km_h'})
    if 'drifter_id' not in d: d=d.merge(test_map,on='forecast_case_id',how='left')
    return d[['forecast_case_id','drifter_id','error_72h_km','ite_km_h']].assign(model=label,kind=kind)

lead=pd.concat([load_lead(f[0],name,'learned') for name,f in MODELS.items()]+[load_lead(f[0],name,'baseline') for name,f in BASELINES.items()],ignore_index=True)
cases=pd.concat([load_case(f[1],name,'learned') for name,f in MODELS.items()]+[load_case(f[1],name,'baseline') for name,f in BASELINES.items()],ignore_index=True)
assert lead.groupby('model').size().eq(72).all() and cases.groupby('model').size().eq(757).all()
print('lead rows:',len(lead),'case rows:',len(cases))


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


rows =[]
for model ,g in cases .groupby ('model',sort =False ):
    e =g .error_72h_km .to_numpy ();q =np .quantile (e ,.9 )
    rows .append ({'model':model ,'kind':g .kind .iloc [0 ],'median_72h_km':np .median (e ),'rmse_72h_km':np .sqrt (np .mean (e **2 )),'median_ite_km_h':np .median (g .ite_km_h ),'worst10_rmse_72h_km':np .sqrt (np .mean (e [e >=q ]**2 )),'p90_72h_km':q })
leaderboard =pd .DataFrame (rows ).sort_values ('rmse_72h_km')
display (leaderboard .round (2 ))

learned =leaderboard [leaderboard .kind .eq ('learned')].copy ()
metrics =['median_72h_km','rmse_72h_km','median_ite_km_h','worst10_rmse_72h_km']
for m in metrics :learned [f'rank_{m }']=learned [m ].rank (method ='min')
learned ['balanced_rank']=learned [[f'rank_{m }'for m in metrics ]].mean (axis =1 )
learned =learned .sort_values (['balanced_rank','rmse_72h_km'])
display (Markdown ('### \u56db\u9879\u6307\u6807\u7b49\u6743\u6392\u540d（\u4ec5\u7528\u4e8e\u603b\u7ed3，\u4e0d\u66ff\u4ee3\u5404\u6307\u6807\u542b\u4e49）'));display (learned [['model',*metrics ,'balanced_rank']].round (2 ))


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(15,5.5))
for model,g in lead.groupby('model',sort=False):
    ls='--' if g.kind.iloc[0]=='baseline' else '-'; lw=2.8 if g.kind.iloc[0]=='learned' else 2
    axes[0].plot(g.lead_time_hours,g.median_error_km,label=model,color=COLORS[model],ls=ls,lw=lw)
    axes[1].plot(g.lead_time_hours,g.rmse_km,label=model,color=COLORS[model],ls=ls,lw=lw)
axes[0].set(title='Median separation error versus lead time',xlabel='Lead time (h)',ylabel='Median error (km)',xlim=(1,72))
axes[1].set(title='RMSE versus lead time',xlabel='Lead time (h)',ylabel='RMSE (km)',xlim=(1,72))
for ax in axes: ax.legend(frameon=False,fontsize=9); ax.grid(alpha=.25)
plt.tight_layout(); fig.savefig(OUTPUT/'champion_models_lead_time_comparison.png',dpi=200,bbox_inches='tight'); plt.show()


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


fig,axes=plt.subplots(2,2,figsize=(14,10)); panel_metrics=[('median_72h_km','72-h median error','km'),('rmse_72h_km','72-h RMSE','km'),('median_ite_km_h','Median 72-h ITE','km·h'),('worst10_rmse_72h_km','Worst 10% RMSE','km')]
for ax,(metric,title,unit) in zip(axes.flat,panel_metrics):
    q=leaderboard.sort_values(metric,ascending=False); ax.barh(q.model,q[metric],color=[COLORS[x] for x in q.model]); ax.set_title(title); ax.set_xlabel(f'{unit}; lower is better')
    for i,v in enumerate(q[metric]): ax.text(v+.008*q[metric].max(),i,f'{v:.1f}',va='center',fontsize=9)
plt.tight_layout(); fig.savefig(OUTPUT/'champion_models_four_metrics.png',dpi=200,bbox_inches='tight'); plt.show()


# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


fig,axes=plt.subplots(2,2,figsize=(15,10),sharex=True,sharey=True)
for ax,model in zip(axes.flat,MODELS):
    for name in [model,*BASELINES]:
        g=lead[lead.model.eq(name)]; ax.plot(g.lead_time_hours,g.rmse_km,color=COLORS[name],lw=2.8 if name==model else 2,ls='-' if name==model else '--',label=name)
    m72=float(lead[(lead.model.eq(model))&lead.lead_time_hours.eq(72)].rmse_km.iloc[0]); p72=float(lead[(lead.model.eq('Velocity persistence'))&lead.lead_time_hours.eq(72)].rmse_km.iloc[0]); f72=float(lead[(lead.model.eq('Spatial mean flow'))&lead.lead_time_hours.eq(72)].rmse_km.iloc[0])
    ax.set_title(f'{model}\n72-h skill: {100*(1-m72/p72):.1f}% vs persistence; {100*(1-m72/f72):.1f}% vs mean flow'); ax.set_xlabel('Lead time (h)'); ax.set_ylabel('RMSE (km)'); ax.legend(frameon=False,fontsize=8)
plt.tight_layout(); fig.savefig(OUTPUT/'champion_models_individual_vs_baselines.png',dpi=200,bbox_inches='tight'); plt.show()


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


skills=[]
for model in MODELS:
    m=lead[lead.model.eq(model)].set_index('lead_time_hours')
    for base in BASELINES:
        b=lead[lead.model.eq(base)].set_index('lead_time_hours')
        for h in m.index: skills.append({'model':model,'baseline':base,'lead_time_hours':h,'skill_pct':100*(1-m.loc[h,'rmse_km']/b.loc[h,'rmse_km'])})
skills=pd.DataFrame(skills)
fig,axes=plt.subplots(1,2,figsize=(15,5.2),sharey=True)
for ax,base in zip(axes,BASELINES):
    for model,g in skills[skills.baseline.eq(base)].groupby('model',sort=False): ax.plot(g.lead_time_hours,g.skill_pct,label=model,color=COLORS[model],lw=2.5)
    ax.axhline(0,color='black',lw=1); ax.set(title=f'RMSE skill vs {base}',xlabel='Lead time (h)',ylabel='Skill (%)'); ax.legend(frameon=False,fontsize=9)
plt.tight_layout(); fig.savefig(OUTPUT/'champion_models_forecast_skill_curves.png',dpi=200,bbox_inches='tight'); plt.show()

skill72=skills[skills.lead_time_hours.eq(72)].pivot(index='model',columns='baseline',values='skill_pct').loc[list(MODELS)]
display(skill72.round(2))


# ## Analysis stage 8
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def bootstrap_contrast(model,baseline,B=5000):
    a=cases[cases.model.eq(model)][['forecast_case_id','drifter_id','error_72h_km']].rename(columns={'error_72h_km':'model_error'})
    b=cases[cases.model.eq(baseline)][['forecast_case_id','error_72h_km']].rename(columns={'error_72h_km':'base_error'})
    d=a.merge(b,on='forecast_case_id'); groups={k:g for k,g in d.groupby('drifter_id')}; ids=np.array(list(groups)); rng=np.random.default_rng(2026); diffs=[]
    for _ in range(B):
        parts=[groups[i] for i in rng.choice(ids,len(ids),replace=True)]; x=np.concatenate([p.model_error for p in parts]); y=np.concatenate([p.base_error for p in parts]); diffs.append(np.sqrt(np.mean(x*x))-np.sqrt(np.mean(y*y)))
    point=np.sqrt(np.mean(d.model_error**2))-np.sqrt(np.mean(d.base_error**2)); lo,hi=np.quantile(diffs,[.025,.975])
    return {'model':model,'baseline':baseline,'rmse_difference_km':point,'ci_2.5_km':lo,'ci_97.5_km':hi,'improvement_supported':hi<0}

bootstrap=pd.DataFrame([bootstrap_contrast(m,b) for m in MODELS for b in BASELINES]); display(bootstrap.round(2))


# ## Analysis stage 9
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


winner_median =leaderboard .loc [leaderboard .median_72h_km .idxmin (),'model'];winner_rmse =leaderboard .loc [leaderboard .rmse_72h_km .idxmin (),'model'];winner_ite =leaderboard .loc [leaderboard .median_ite_km_h .idxmin (),'model'];winner_tail =leaderboard .loc [leaderboard .worst10_rmse_72h_km .idxmin (),'model'];balanced =learned .iloc [0 ].model 
display (Markdown (f"""### \u7ed3\u8bba

- **\u5178\u578b72\u5c0f\u65f6case（median）\u6700\u4f73：{winner_median }**。
- **\u603b\u4f53\u5e73\u65b9\u8bef\u5dee（RMSE）\u6700\u4f73：{winner_rmse }**。
- **\u6574\u6761\u8f68\u8ff9\u7d2f\u8ba1\u8bef\u5dee（ITE）\u6700\u4f73：{winner_ite }**。
- **\u6781\u7aef\u5931\u8d25\u63a7\u5236\u6700\u4f73：{winner_tail }**。
- \u56db\u9879\u6307\u6807\u7b49\u6743\u6392\u540d\u4e0b，\u6700\u5747\u8861\u7684\u6a21\u578b\u662f **{balanced }**。

\u56e0\u6b64\u4e0d\u80fd\u628a“\u6700\u4f73\u6a21\u578b”\u538b\u7f29\u6210\u4e00\u4e2a\u4e0d\u5e26\u6761\u4ef6\u7684\u540d\u5b57：

1. \u82e5\u4e1a\u52a1\u6700\u6015\u641c\u7d22\u533a\u57df\u4e25\u91cd\u504f\u79bb\u6216\u6f0f\u6389\u76ee\u6807，\u5e94\u4f18\u5148 **Boosting residual**，\u56e0\u4e3aRMSE\u548c\u6700\u5dee10%\u8bef\u5dee\u6700\u4f4e；
2. \u82e5\u5f3a\u8c0372\u5c0f\u65f6\u6574\u6761\u8f68\u8ff9\u7684\u7efc\u5408\u7a33\u5b9a\u6027，\u5e94\u4f18\u5148 **Optimised FNN**，\u56e0\u4e3aITE\u6700\u4f4e\u4e14\u5404\u9879\u90fd\u5904\u4e8e\u7b2c\u4e00\u68af\u961f；
3. \u82e5\u53ea\u63cf\u8ff0\u5178\u578bcase\u6700\u7ec8\u4f4d\u7f6e，\u5e94\u62a5\u544a **Optimised KNN** \u7684\u6700\u4f4emedian，\u4f46\u540c\u65f6\u8bf4\u660e\u5176\u5c3e\u90e8\u8bef\u5dee\u66f4\u5927；
4. Random Forest\u662f\u7a33\u5b9a\u53c2\u7167，\u4f46\u76ee\u524d\u6ca1\u6709\u5355\u9879\u51a0\u519b。
"""))


# ## Analysis stage 10
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


leaderboard .to_csv (OUTPUT /'champion_models_leaderboard.csv',index =False );learned .to_csv (OUTPUT /'champion_models_balanced_ranking.csv',index =False );skills .to_csv (OUTPUT /'champion_models_forecast_skill_by_lead.csv',index =False );skill72 .to_csv (OUTPUT /'champion_models_72h_skill_vs_baselines.csv');bootstrap .to_csv (OUTPUT /'champion_models_bootstrap_vs_baselines.csv',index =False )
print ('\u5df2\u4fdd\u5b58\u7efc\u5408\u8868\u683c\u4e0e\u56fe\u7247\u5230：',OUTPUT )

