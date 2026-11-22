"""Generate RQ3 report figures from the canonical final outputs."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
PROJECT=Path(__file__).resolve().parents[2]
SRC=PROJECT/'RQ3'/'optimised_fnn'/'output'
OUT=PROJECT/'figures'/'report'/'RQ3';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
leads=pd.read_csv(SRC/'optimised_fnn_multi_nominal_by_lead.csv')
reg=pd.read_csv(SRC/'optimised_fnn_conformal_regime_summary.csv')
boot=pd.read_csv(SRC/'optimised_fnn_paired_bootstrap_summary.csv')
levels=(.8,.9,.95)
# Figure 1: 90% coverage and radius all 72 lead times
s=leads[leads.nominal_coverage.eq(.9)].sort_values('lead_time_hours')
fig,axs=plt.subplots(1,2,figsize=(11,3.65),layout='constrained')
axs[0].plot(s.lead_time_hours,100*s.empirical_coverage,lw=2)
axs[0].axhline(90,ls='--',c='grey',label='90% nominal target')
axs[0].set(xlabel='Lead time (hours)',ylabel='Empirical coverage (%)',title='A. Global 90% coverage',xlim=(1,72),ylim=(80,100));axs[0].legend(loc='lower right',frameon=False)
axs[1].plot(s.lead_time_hours,s.radius_km,lw=2)
axs[1].set(xlabel='Lead time (hours)',ylabel='Calibrated radius (km)',title='B. Spatial extent of global regions',xlim=(1,72))
for ax in axs:
 ax.set_xticks([1,24,48,72]);ax.grid(alpha=.18)
fig.savefig(OUT/'Fig1_global_leadtime.png',dpi=250,bbox_inches='tight');plt.close(fig)
# Figure 2: per-regime coverage bars
order=['Shelf','Kuroshio Core','Transition','Other Offshore']; a=reg.pivot(index='regime',columns='method',values='coverage_72h').loc[order]
fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(4);width=.36
for off,col in [(-width/2,'Global'),(width/2,'Mondrian')]:
 bars=ax.bar(x+off,100*a[col],width,label=col)
 for bar in bars: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f'{bar.get_height():.1f}%',ha='center',fontsize=9)
ax.axhline(90,ls='--',c='grey',label='Nominal 90%');ax.set(xticks=x,xticklabels=order,ylabel='72h empirical coverage (%)',ylim=(0,108),title='Regional reliability: global vs Mondrian');ax.legend(ncol=3,frameon=False,loc='upper center');ax.grid(axis='y',alpha=.15)
fig.savefig(OUT/'Fig2_regime_coverage.png',dpi=250,bbox_inches='tight');plt.close(fig)
# Figure 3: region radius and area change
b=reg.pivot(index='regime',columns='method',values='area_72h_km2').loc[order]
c=reg.pivot(index='regime',columns='method',values='radius_72h_km').loc[order]
fig,axs=plt.subplots(1,2,figsize=(11,3.8),layout='constrained');xx=np.arange(4)
axs[0].bar(xx-.18,c['Global'],.36,label='Global');axs[0].bar(xx+.18,c['Mondrian'],.36,label='Mondrian')
axs[0].set(xticks=xx,xticklabels=['Shelf','Kuroshio','Transition','Offshore'],ylabel='72h radius (km)',title='A. Region radius');axs[0].legend(frameon=False)
pct=100*(b['Mondrian']/b['Global']-1);bars=axs[1].bar(xx,pct)
axs[1].axhline(0,color='grey',lw=1);axs[1].set(xticks=xx,xticklabels=['Shelf','Kuroshio','Transition','Offshore'],ylabel='Mondrian area change vs global (%)',title='B. Change in area')
for bar,val in zip(bars,pct): axs[1].text(bar.get_x()+bar.get_width()/2,val+(3 if val>=0 else -6),f'{val:+.1f}%',ha='center',fontsize=9)
for ax in axs:ax.grid(axis='y',alpha=.15)
fig.savefig(OUT/'Fig3_radius_area_tradeoff.png',dpi=250,bbox_inches='tight');plt.close(fig)
# Figure 4: 72h calibration across nominal levels, 4 panels
multi=pd.read_csv(SRC/'optimised_fnn_regime_multi_nominal_72h.csv')
fig,axs=plt.subplots(2,2,figsize=(9.5,7),layout='constrained',sharex=True,sharey=True)
for ax,regime in zip(axs.flat,order):
 for method in ['Global','Mondrian']:
  z=multi[(multi.regime==regime)&(multi.method==method)].sort_values('nominal_coverage')
  ax.plot(z.nominal_coverage*100,z.empirical_coverage*100,marker='o',lw=2,label=method)
 ax.plot([75,100],[75,100],'--',c='grey',label='Ideal calibration')
 ax.set(title=regime,xlim=(78,97),ylim=(60,101),xticks=[80,90,95]);ax.grid(alpha=.15)
for ax in axs[1,:]:ax.set_xlabel('Nominal coverage (%)')
for ax in axs[:,0]:ax.set_ylabel('Empirical coverage (%)')
axs[0,0].legend(fontsize=8,frameon=False)
fig.savefig(OUT/'Fig4_regime_multinominal.png',dpi=250,bbox_inches='tight');plt.close(fig)
# export main tables
s0=leads[(leads.nominal_coverage==.9)&(leads.lead_time_hours.isin([24,48,72]))][['lead_time_hours','empirical_coverage','radius_km']].copy()
s0['empirical_coverage']=100*s0['empirical_coverage'];s0.to_csv(OUT/'Table1_global_90_by_lead.csv',index=False)
reg.to_csv(OUT/'Table2_global_vs_mondrian.csv',index=False)
boot.to_csv(OUT/'Table3_paired_bootstrap.csv',index=False)
print('Generated:',*[f.name for f in sorted(OUT.iterdir())],sep='\n- ')
