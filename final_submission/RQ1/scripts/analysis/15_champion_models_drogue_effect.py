#!/usr/bin/env python
# coding: utf-8

# # 15 Champion Models Drogue Effect
# 
# Examines whether drogue status independently explains forecast error after adjustment for environmental and location-related conditions.
# 
# **Provenance.** This is an English-facing execution copy of the original project notebook. The computational cells are preserved so that the analysis remains reproducible. Chinese narrative cells have been replaced by concise English stage notes; the original notebook remains unchanged in the source project.

# ## Analysis stage 1
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


from pathlib import Path 
import warnings 
import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 
from IPython .display import display ,Markdown 

candidates =[Path .cwd (),*Path .cwd ().parents ]
ROOT =next ((p for p in candidates if (p /'data'/'processed'/'forecast_origins.parquet').exists ()),None )
if ROOT is None :raise FileNotFoundError ('\u8bf7\u4ece data3001-project \u6216\u5176\u5b50\u76ee\u5f55\u8fd0\u884c。')
OUT =ROOT /'output'/'rq1'/'zh'
FIG =OUT /'drogue_champion_analysis'
FIG .mkdir (parents =True ,exist_ok =True )
plt .style .use ('seaborn-v0_8-whitegrid')
plt .rcParams .update ({'figure.dpi':120 ,'axes.spines.top':False ,'axes.spines.right':False })
print ('Project:',ROOT )


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


origins=pd.read_parquet(ROOT/'data'/'processed'/'forecast_origins.parquet')
origins=origins.loc[origins.partition.eq('test')].copy()
regimes=pd.read_csv(ROOT/'RQ2'/'data'/'forecast_case_regimes.csv')
regimes=regimes[['forecast_case_id','regime','depth_m','current_speed_clim']]

files={
 'Optimised FNN':'optimised_fnn_test_case_metrics.csv',
 'Boosting residual':'boosting_residual_test_case_metrics.csv',
 'Optimised Weighted KNN':'optimised_knn_test_case_metrics.csv',
 'Optimised Random Forest':'optimised_random_forest_test_case_metrics.csv',
}
parts=[]
for model,fn in files.items():
    d=pd.read_csv(OUT/fn).rename(columns={'ID':'drifter_id','final_72h_error_km':'error_72h_km','integrated_trajectory_error_km_h':'ite_km_h'})
    d=d[['forecast_case_id','drifter_id','error_72h_km','ite_km_h']].copy(); d['model']=model; parts.append(d)
errors=pd.concat(parts,ignore_index=True)
meta=origins[['forecast_case_id','ID','drogue_status_at_origin','origin_ve','origin_vn']].merge(regimes,on='forecast_case_id',how='left',validate='one_to_one')
meta['origin_speed_m_s']=np.hypot(meta.origin_ve,meta.origin_vn)
data=errors.merge(meta,on='forecast_case_id',how='left',validate='many_to_one')
data['drogue_status']=data.drogue_status_at_origin.replace({'attached':'Attached','lost':'Lost'})
assert data.groupby('model').size().eq(757).all()
display(data.groupby(['model','drogue_status']).agg(cases=('forecast_case_id','size'),drifters=('drifter_id','nunique')).reset_index())


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


summary=(data.groupby(['model','drogue_status']).agg(
    n_cases=('forecast_case_id','size'),n_drifters=('drifter_id','nunique'),
    median_72h_error_km=('error_72h_km','median'),
    rmse_72h_km=('error_72h_km',lambda x: np.sqrt(np.mean(np.square(x)))),
    p90_72h_error_km=('error_72h_km',lambda x: np.quantile(x,.9)),
    median_ite_km_h=('ite_km_h','median')).reset_index())
display(summary.round(3))

order=list(files)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,metric,title in [(axes[0],'rmse_72h_km','72-h RMSE'),(axes[1],'median_72h_error_km','72-h median error')]:
    p=summary.pivot(index='model',columns='drogue_status',values=metric).reindex(order)
    p.plot.bar(ax=ax,color=['#2C7BB6','#D7191C'],width=.75)
    ax.set_title(title); ax.set_ylabel('km; lower is better'); ax.set_xlabel(''); ax.tick_params(axis='x',rotation=20); ax.legend(title='Drogue at origin',frameon=False)
fig.suptitle('Champion models: observed errors by drogue status',fontweight='bold'); fig.tight_layout()
fig.savefig(FIG/'drogue_status_descriptive_errors.png',dpi=200,bbox_inches='tight'); plt.show()


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def metric_difference(frame,metric):
    a=frame.loc[frame.drogue_status.eq('Attached'),metric].to_numpy(); l=frame.loc[frame.drogue_status.eq('Lost'),metric].to_numpy()
    if not len(a) or not len(l): return np.nan
    if metric=='error_72h_km': return np.sqrt(np.mean(l*l))-np.sqrt(np.mean(a*a))
    return np.median(l)-np.median(a)

def cluster_bootstrap_difference(frame,metric,n_boot=5000,seed=2026):
    groups={k:g for k,g in frame.groupby('drifter_id')}; ids=np.array(list(groups)); rng=np.random.default_rng(seed)
    point=metric_difference(frame,metric); draws=[]
    for _ in range(n_boot):
        sample=pd.concat([groups[i] for i in rng.choice(ids,len(ids),replace=True)],ignore_index=True)
        value=metric_difference(sample,metric)
        if np.isfinite(value): draws.append(value)
    lo,hi=np.quantile(draws,[.025,.975])
    return point,lo,hi,len(draws)

rows=[]
for model,g in data.groupby('model',sort=False):
    for metric,label in [('error_72h_km','RMSE difference'),('ite_km_h','Median ITE difference')]:
        point,lo,hi,n=cluster_bootstrap_difference(g,metric)
        rows.append({'model':model,'contrast':label,'lost_minus_attached':point,'ci_lower':lo,'ci_upper':hi,'bootstrap_draws':n,'ci_excludes_zero':bool(lo>0 or hi<0)})
bootstrap=pd.DataFrame(rows)
display(bootstrap.round(3))


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def design_matrix(frame):
    z=frame.copy()
    continuous=z[['current_speed_clim','depth_m','origin_speed_m_s']].astype(float)
    continuous=continuous.fillna(continuous.median())
    continuous=(continuous-continuous.mean())/continuous.std().replace(0,1)
    regimes=pd.get_dummies(z.regime.fillna('Unknown'),prefix='regime',drop_first=True,dtype=float)
    X=np.column_stack([np.ones(len(z)),z.drogue_status.eq('Lost').astype(float),continuous.to_numpy(),regimes.to_numpy()])
    y=np.log1p(z.error_72h_km.to_numpy())
    return X,y

def adjusted_lost_effect(frame,n_boot=3000,seed=3001):
    X,y=design_matrix(frame); beta=np.linalg.lstsq(X,y,rcond=None)[0][1]
    groups={k:g for k,g in frame.groupby('drifter_id')}; ids=np.array(list(groups)); rng=np.random.default_rng(seed); draws=[]
    for _ in range(n_boot):
        sample=pd.concat([groups[i] for i in rng.choice(ids,len(ids),replace=True)],ignore_index=True)
        xb,yb=design_matrix(sample)
        draws.append(np.linalg.lstsq(xb,yb,rcond=None)[0][1])
    lo,hi=np.quantile(draws,[.025,.975])
    return 100*np.expm1(beta),100*np.expm1(lo),100*np.expm1(hi)

adjusted=[]
for model,g in data.groupby('model',sort=False):
    point,lo,hi=adjusted_lost_effect(g)
    adjusted.append({'model':model,'adjusted_lost_error_change_pct':point,'ci_lower_pct':lo,'ci_upper_pct':hi,'ci_excludes_zero':bool(lo>0 or hi<0)})
adjusted=pd.DataFrame(adjusted)
display(adjusted.round(2))

fig,ax=plt.subplots(figsize=(9,4.8)); y=np.arange(len(adjusted)); x=adjusted.adjusted_lost_error_change_pct.to_numpy(); lo=adjusted.ci_lower_pct.to_numpy(); hi=adjusted.ci_upper_pct.to_numpy()
ax.errorbar(x,y,xerr=np.vstack([x-lo,hi-x]),fmt='o',color='#6A3D9A',capsize=4)
ax.axvline(0,color='black',lw=1,ls='--'); ax.set_yticks(y,adjusted.model); ax.set_xlabel('Adjusted error change for Lost vs Attached (%)'); ax.set_title('Adjusted drogue-status association (95% drifter-cluster bootstrap CI)',fontweight='bold')
fig.tight_layout(); fig.savefig(FIG/'drogue_status_adjusted_effect.png',dpi=200,bbox_inches='tight'); plt.show()


# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


lines =[]
for r in adjusted .itertuples ():
    evidence ='\u5b58\u5728\u6e05\u6670\u7684\u8c03\u6574\u540e\u5173\u8054'if r .ci_excludes_zero else '\u8c03\u6574\u540e\u7f6e\u4fe1\u533a\u95f4\u8de8\u8fc70，\u8bc1\u636e\u4e0d\u8db3'
    lines .append (f'- **{r .model }**：Lost \u76f8\u5bf9 Attached \u7684\u8c03\u6574\u540e\u8bef\u5dee\u53d8\u5316\u4e3a **{r .adjusted_lost_error_change_pct :.1f}%**（95% CI {r .ci_lower_pct :.1f}% \u81f3 {r .ci_upper_pct :.1f}%）；{evidence }。')
display (Markdown ('### \u7ed3\u8bba\n\n'+'\n'.join (lines )+'\n\n\u603b\u4f53\u5224\u65ad\u5e94\u4ee5\u56db\u4e2a\u6a21\u578b\u7684\u65b9\u5411\u662f\u5426\u4e00\u81f4、\u7f6e\u4fe1\u533a\u95f4\u4ee5\u53ca\u63a7\u5236\u73af\u5883\u56e0\u7d20\u540e\u7684\u7ed3\u679c\u4e3a\u51c6，\u800c\u4e0d\u80fd\u53ea\u6bd4\u8f83\u539f\u59cb RMSE。'))
summary .to_csv (FIG /'drogue_champion_descriptive_summary.csv',index =False )
bootstrap .to_csv (FIG /'drogue_champion_cluster_bootstrap.csv',index =False )
adjusted .to_csv (FIG /'drogue_champion_adjusted_effect.csv',index =False )
data .to_csv (FIG /'drogue_champion_case_level_analysis.csv',index =False )
print ('Saved to',FIG )

