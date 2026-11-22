#!/usr/bin/env python
# coding: utf-8

# # 09 Each Model Vs Two Baselines
# 
# Computes model skill relative to velocity persistence and spatial mean flow.
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
from IPython.display import display, Markdown

ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'output/rq1/model_error_analysis').exists())
DATA=ROOT/'output/rq1/model_error_analysis'
OUT=DATA/'individual_baseline_comparisons';OUT.mkdir(exist_ok=True)
lead=pd.read_csv(DATA/'all_models_lead_metrics.csv')
summary=pd.read_csv(DATA/'model_error_summary_24_48_72.csv')
BASELINES=['Velocity persistence','Spatial mean flow']
MODELS=['FNN','Weighted KNN','Ridge','VAR(20)','VAR(1)']
COLORS={'Velocity persistence':'#606872','Spatial mean flow':'#A0A7AF','FNN':'#D81B60',
        'Weighted KNN':'#159D73','Ridge':'#2864DC','VAR(20)':'#7043A5','VAR(1)':'#E17C05'}


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[2]:


display(summary.set_index('Model').loc[MODELS+BASELINES].round(2))


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[3]:


def model_table(model):
    rows=[]
    for h in [24,48,72]:
        values={}
        for name in [model]+BASELINES:
            r=lead[(lead.model.eq(name))&(lead.lead_time_hours.eq(h))].iloc[0]
            values[name]=(r.median_error_km,r.rmse_km)
        rows.append({'Lead time':f'{h} h',
          f'{model} Median':values[model][0],f'{model} RMSE':values[model][1],
          'Persistence Median':values[BASELINES[0]][0],'Persistence RMSE':values[BASELINES[0]][1],
          'Mean flow Median':values[BASELINES[1]][0],'Mean flow RMSE':values[BASELINES[1]][1],
          'RMSE skill vs persistence (%)':100*(1-values[model][1]/values[BASELINES[0]][1]),
          'RMSE skill vs mean flow (%)':100*(1-values[model][1]/values[BASELINES[1]][1])})
    return pd.DataFrame(rows)

for model in MODELS:
    display(Markdown(f'### {model} vs baselines'))
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharex=True)
    for name in [model]+BASELINES:
        g=lead[lead.model.eq(name)].sort_values('lead_time_hours')
        style='-' if name==model else '--'; width=3 if name==model else 2
        axes[0].plot(g.lead_time_hours,g.median_error_km,label=name,color=COLORS[name],ls=style,lw=width)
        axes[1].plot(g.lead_time_hours,g.rmse_km,label=name,color=COLORS[name],ls=style,lw=width)
    axes[0].set_title('Median separation error');axes[1].set_title('RMSE')
    for ax in axes:
        ax.set(xlabel='Lead time (h)',ylabel='Error (km)');ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.suptitle(f'{model} compared with physical baselines',fontweight='bold')
    fig.tight_layout();fig.savefig(OUT/f"{model.lower().replace(' ','_').replace('(','').replace(')','')}_vs_baselines.png",dpi=200,bbox_inches='tight')
    plt.show()
    table=model_table(model)
    display(table.round(2))
    table.to_csv(OUT/f"{model.lower().replace(' ','_').replace('(','').replace(')','')}_vs_baselines.csv",index=False)


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[4]:


skill=[]
for model in MODELS:
    t=model_table(model).iloc[-1]
    skill.append({'Model':model,'vs Velocity persistence (%)':t['RMSE skill vs persistence (%)'],
                  'vs Spatial mean flow (%)':t['RMSE skill vs mean flow (%)']})
skill=pd.DataFrame(skill).set_index('Model')
display(skill.round(2))
ax=skill.plot.barh(figsize=(8.5,4.5),color=['#5DADE2','#A9CCE3'])
ax.axvline(0,color='black',lw=1);ax.set(xlabel='Reduction in 72-h RMSE (%)',title='72-h skill relative to baselines')
ax.grid(axis='x',alpha=.25);ax.legend(frameon=False)
plt.tight_layout();plt.savefig(OUT/'all_models_72h_skill.png',dpi=200,bbox_inches='tight');plt.show()
skill.to_csv(OUT/'all_models_72h_skill.csv')

