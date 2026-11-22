#!/usr/bin/env python
# coding: utf-8

# # 12 Optimised Weighted KNN
# 
# Tunes the weighted analogue model, including feature weighting and neighbour choices.
# 
# **Provenance.** This is an English-facing execution copy of the original project notebook. The computational cells are preserved so that the analysis remains reproducible. Chinese narrative cells have been replaced by concise English stage notes; the original notebook remains unchanged in the source project.

# ## Analysis stage 1
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


from pathlib import Path 
import json ,os ,time 
import matplotlib .pyplot as plt 
import numpy as np 
import pandas as pd 
from IPython import get_ipython 
from IPython .display import display ,Markdown 
from sklearn .decomposition import PCA 

ip =get_ipython ()
if ip is not None :ip .run_line_magic ('matplotlib','inline')
PROJECT_ROOT =next ((p for p in [Path .cwd (),*Path .cwd ().parents ]if (p /'data'/'processed').exists ()),None )
if PROJECT_ROOT is None :raise FileNotFoundError ('\u627e\u4e0d\u5230data/processed。')
PROCESSED =PROJECT_ROOT /'data'/'processed';OUTPUT_DIR =PROJECT_ROOT /'output'/'rq1'/'zh';OUTPUT_DIR .mkdir (parents =True ,exist_ok =True )
EARTH_RADIUS_KM =6371. ;HISTORY_HOURS =24 ;HORIZON_HOURS =72 ;N_FOLDS =5 ;SEED =3001 
FAST_MODE =os .environ .get ('KNN_FAST_MODE','0')=='1'
K_VALUES =[8 ,16 ,32 ]if FAST_MODE else [4 ,6 ,8 ,10 ,12 ,16 ,20 ,24 ,32 ,48 ,64 ,96 ]
POWERS =[1. ,2. ]if FAST_MODE else [.5 ,1. ,1.5 ,2. ,3. ,4. ]
QUERY_BATCH_SIZE =256 
plt .style .use ('seaborn-v0_8-whitegrid');plt .rcParams ['font.sans-serif']=['PingFang SC','Arial Unicode MS','DejaVu Sans'];plt .rcParams ['axes.unicode_minus']=False 
print ('\u9879\u76ee\u76ee\u5f55：',PROJECT_ROOT ,'\u6a21\u5f0f：','\u5feb\u901f\u68c0\u67e5'if FAST_MODE else '\u5b8c\u6574\u641c\u7d22')


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def latlon_to_local_km(origin_lat,origin_lon,latitude,longitude):
    lat0,lon0=np.deg2rad(origin_lat),np.deg2rad(origin_lon); lat1,lon1=np.deg2rad(latitude),np.deg2rad(longitude)
    dlon=(lon1-lon0+np.pi)%(2*np.pi)-np.pi; dlat=lat1-lat0
    a=np.sin(dlat/2)**2+np.cos(lat0)*np.cos(lat1)*np.sin(dlon/2)**2
    angle=2*np.arcsin(np.sqrt(np.clip(a,0,1)))
    bearing=np.arctan2(np.sin(dlon)*np.cos(lat1),np.cos(lat0)*np.sin(lat1)-np.sin(lat0)*np.cos(lat1)*np.cos(dlon))
    d=EARTH_RADIUS_KM*angle; return d*np.sin(bearing),d*np.cos(bearing)

def local_km_to_latlon(origin_lat,origin_lon,east,north):
    lat0,lon0=np.deg2rad(origin_lat),np.deg2rad(origin_lon); dist=np.hypot(east,north); b=np.arctan2(east,north); a=dist/EARTH_RADIUS_KM
    lat=np.arcsin(np.sin(lat0)*np.cos(a)+np.cos(lat0)*np.sin(a)*np.cos(b))
    lon=lon0+np.arctan2(np.sin(b)*np.sin(a)*np.cos(lat0),np.cos(a)-np.sin(lat0)*np.sin(lat))
    return np.rad2deg(lat),(np.rad2deg(lon)+180)%360-180

def haversine_km(lat1,lon1,lat2,lon2):
    lat1,lon1,lat2,lon2=map(np.deg2rad,[lat1,lon1,lat2,lon2]); dlat=lat2-lat1; dlon=lon2-lon1
    a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*EARTH_RADIUS_KM*np.arcsin(np.sqrt(np.clip(a,0,1)))

origins=pd.read_parquet(PROCESSED/'forecast_origins.parquet').sort_values('forecast_case_id').reset_index(drop=True)
obs=pd.read_parquet(PROCESSED/'region_A_observations.parquet',columns=['ID','segment_id','segment_step','latitude','longitude','ve','vn'])
n=len(origins)
hist=origins[['forecast_case_id','ID','segment_id','history_start_step','origin_latitude','origin_longitude']].loc[origins.index.repeat(24)].reset_index(drop=True)
hist['history_index']=np.tile(np.arange(24),n); hist['segment_step']=hist.history_start_step+hist.history_index
hist=hist.merge(obs,on=['ID','segment_id','segment_step'],how='left',validate='many_to_one').sort_values(['forecast_case_id','history_index']).reset_index(drop=True)
hist['east'],hist['north']=latlon_to_local_km(hist.origin_latitude,hist.origin_longitude,hist.latitude,hist.longitude)
hv=hist[['east','north','ve','vn']].to_numpy().reshape(n,24,4)
X=np.column_stack([origins.origin_latitude,origins.origin_longitude,hv.reshape(n,-1)]).astype(float)

targ=origins[['forecast_case_id','ID','segment_id','target_start_step','origin_latitude','origin_longitude']].loc[origins.index.repeat(72)].reset_index(drop=True)
targ['lead_time_hours']=np.tile(np.arange(1,73),n); targ['segment_step']=targ.target_start_step+targ.lead_time_hours-1
targ=targ.merge(obs[['ID','segment_id','segment_step','latitude','longitude']],on=['ID','segment_id','segment_step'],how='left',validate='many_to_one').sort_values(['forecast_case_id','lead_time_hours']).reset_index(drop=True)
targ['east'],targ['north']=latlon_to_local_km(targ.origin_latitude,targ.origin_longitude,targ.latitude,targ.longitude)
Y=targ[['east','north']].to_numpy().reshape(n,144).astype(float)
hours=np.arange(1,73)[None,:]; lv=hv[:,-1,2:4]
persistence_Y=np.stack([lv[:,0,None]*3.6*hours,lv[:,1,None]*3.6*hours],axis=-1).reshape(n,144)
train_idx=np.flatnonzero(origins.partition.eq('training')); cal_idx=np.flatnonzero(origins.partition.eq('calibration')); test_idx=np.flatnonzero(origins.partition.eq('test'))
assert X.shape==(5117,98) and Y.shape==(5117,144) and (len(train_idx),len(cal_idx),len(test_idx))==(3591,769,757)
print('X',X.shape,'Y',Y.shape,'train/cal/test',len(train_idx),len(cal_idx),len(test_idx))


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


counts=origins.loc[train_idx].groupby('ID').size().sort_values(ascending=False)
fold_drifters=[[] for _ in range(5)]; fold_counts=np.zeros(5,dtype=int)
for did,count in counts.items():
    f=int(np.argmin(fold_counts)); fold_drifters[f].append(did); fold_counts[f]+=int(count)
display(pd.DataFrame({'fold':np.arange(1,6),'validation_drifters':[len(x) for x in fold_drifters],'validation_cases':fold_counts}))


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def feature_weights(scheme):
    w=np.ones(98); h=w[2:].reshape(24,4); rec=np.linspace(.4,1.6,24)
    if scheme=='equal': pass
    elif scheme=='spatial_boost': w[:2]=8
    elif scheme=='recent': w[:2]=2; h[:]=rec[:,None]
    elif scheme=='recent_velocity':
        w[:2]=2; h[:,:2]=(.75*rec)[:,None]; h[:,2:]=(1.5*rec)[:,None]
    elif scheme=='spatial_recent_velocity':
        w[:2]=8; h[:,:2]=(.75*rec)[:,None]; h[:,2:]=(1.5*rec)[:,None]
    else: raise ValueError(scheme)
    return w

SCHEMES=['equal','spatial_boost','recent','recent_velocity','spatial_recent_velocity']
PCA_DIMS=[None,30] if FAST_MODE else [None,30,60]
REPRESENTATIONS=[{'name':f'{s}|PCA={d or "none"}','scheme':s,'pca_dim':d} for s in SCHEMES for d in PCA_DIMS]

def transform_representation(Xtr,Xq,rep):
    mean=Xtr.mean(0); scale=Xtr.std(0); scale[scale<1e-12]=1
    root_w=np.sqrt(feature_weights(rep['scheme']))
    A=(Xtr-mean)/scale*root_w; B=(Xq-mean)/scale*root_w
    pca=None
    if rep['pca_dim'] is not None:
        pca=PCA(n_components=rep['pca_dim'],random_state=SEED).fit(A); A=pca.transform(A); B=pca.transform(B)
    return A,B,mean,scale,pca

def neighbours(A,B,max_k):
    tr2=np.sum(A*A,axis=1); out_i=[]; out_d=[]; nf=A.shape[1]
    for start in range(0,len(B),QUERY_BATCH_SIZE):
        q=B[start:start+QUERY_BATCH_SIZE]; d2=np.maximum((np.sum(q*q,axis=1,keepdims=True)+tr2[None,:]-2*q@A.T)/nf,0)
        ind=np.argpartition(d2,max_k-1,axis=1)[:,:max_k]; vals=np.take_along_axis(d2,ind,axis=1)
        order=np.argsort(vals,axis=1); out_i.append(np.take_along_axis(ind,order,axis=1)); out_d.append(np.sqrt(np.take_along_axis(vals,order,axis=1)))
    return np.vstack(out_i),np.vstack(out_d)

def weighted_prediction(Ytrain,idx,dist,k,power):
    ii=idx[:,:k]; dd=dist[:,:k]; w=(dd+1e-8)**(-power); w/=w.sum(1,keepdims=True)
    return np.einsum('nk,nko->no',w,Ytrain[ii],optimize=True),1/np.sum(w*w,axis=1)

def endpoint_rmse(pred,true,origin_rows):
    pp=pred.reshape(-1,72,2)[:,-1]; yy=true.reshape(-1,72,2)[:,-1]
    olat=origin_rows.origin_latitude.to_numpy(); olon=origin_rows.origin_longitude.to_numpy()
    plat,plon=local_km_to_latlon(olat,olon,pp[:,0],pp[:,1]); tlat,tlon=local_km_to_latlon(olat,olon,yy[:,0],yy[:,1])
    e=haversine_km(plat,plon,tlat,tlon); return float(np.sqrt(np.mean(e**2)))


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


stage1 =[];started =time .time ()
for rep in REPRESENTATIONS :
    print (rep ['name'])
    for f ,valid_ids in enumerate (fold_drifters ,1 ):
        mask =origins .loc [train_idx ,'ID'].isin (valid_ids ).to_numpy ();va =train_idx [mask ];tr =train_idx [~mask ]
        A ,B ,*_ =transform_representation (X [tr ],X [va ],rep );ni ,nd =neighbours (A ,B ,20 )
        pred ,_ =weighted_prediction (Y [tr ],ni ,nd ,20 ,2. )
        stage1 .append ({**rep ,'fold':f ,'validation_72h_rmse_km':endpoint_rmse (pred ,Y [va ],origins .loc [va ])})
stage1_folds =pd .DataFrame (stage1 )
stage1_summary =(stage1_folds .groupby (['name','scheme','pca_dim'],dropna =False ).validation_72h_rmse_km .agg (['mean','std']).reset_index ().rename (columns ={'mean':'mean_cv_72h_rmse_km','std':'std_cv_72h_rmse_km'}))
top_names =stage1_summary .nsmallest (2 if FAST_MODE else 3 ,'mean_cv_72h_rmse_km').name .tolist ()
top_reps =[r for r in REPRESENTATIONS if r ['name']in top_names ]
display (stage1_summary .sort_values ('mean_cv_72h_rmse_km').round (3 ));print ('\u8fdb\u5165Stage 2：',top_names ,'elapsed',round (time .time ()-started ,1 ),'s')


# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


stage2 =[];max_k =max (K_VALUES )
for rep in top_reps :
    print ('\n',rep ['name'])
    for f ,valid_ids in enumerate (fold_drifters ,1 ):
        mask =origins .loc [train_idx ,'ID'].isin (valid_ids ).to_numpy ();va =train_idx [mask ];tr =train_idx [~mask ]
        A ,B ,*_ =transform_representation (X [tr ],X [va ],rep );ni ,nd =neighbours (A ,B ,max_k )
        for target_mode in ['direct','persistence_residual']:
            train_target =Y [tr ]if target_mode =='direct'else Y [tr ]-persistence_Y [tr ]
            offset =np .zeros_like (Y [va ])if target_mode =='direct'else persistence_Y [va ]
            neighbour_targets =train_target [ni ]
            for power in POWERS :
                raw_w =(nd +1e-8 )**(-power );numerator =np .cumsum (raw_w [:,:,None ]*neighbour_targets ,axis =1 );denominator =np .cumsum (raw_w ,axis =1 )
                for k in K_VALUES :
                    pred =numerator [:,k -1 ,:]/denominator [:,k -1 ,None ]+offset 
                    norm_w =raw_w [:,:k ]/raw_w [:,:k ].sum (1 ,keepdims =True )
                    stage2 .append ({**rep ,'fold':f ,'target_mode':target_mode ,'k':k ,'power':power ,
                    'median_effective_neighbours':float (np .median (1 /np .sum (norm_w **2 ,axis =1 ))),
                    'validation_72h_rmse_km':endpoint_rmse (pred ,Y [va ],origins .loc [va ])})
        print (' fold',f ,'done')
stage2_folds =pd .DataFrame (stage2 )
group_cols =['name','scheme','pca_dim','target_mode','k','power']
stage2_summary =(stage2_folds .groupby (group_cols ,dropna =False )
.agg (mean_cv_72h_rmse_km =('validation_72h_rmse_km','mean'),std_cv_72h_rmse_km =('validation_72h_rmse_km','std'),
median_effective_neighbours =('median_effective_neighbours','median')).reset_index ())
best =stage2_summary .loc [stage2_summary .mean_cv_72h_rmse_km .idxmin ()]
best_rep =next (r for r in top_reps if r ['name']==best ['name']);best_k =int (best .k );best_power =float (best .power );best_mode =best .target_mode 
display (stage2_summary .sort_values ('mean_cv_72h_rmse_km').head (25 ).round (3 ))
print ('\u6700\u4f18：',best_rep ,'K=',best_k ,'p=',best_power ,'target=',best_mode )


# In[ ]:


fig,axes=plt.subplots(1,2,figsize=(13,4.8))
s1=stage1_summary.sort_values('mean_cv_72h_rmse_km'); axes[0].barh(s1.name,s1.mean_cv_72h_rmse_km,xerr=s1.std_cv_72h_rmse_km,color='#80b1d3',capsize=3); axes[0].invert_yaxis(); axes[0].set_title('Stage 1: distance representation'); axes[0].set_xlabel('CV 72-h RMSE (km)')
for mode,g in stage2_summary[stage2_summary.name.eq(best['name']) & stage2_summary.power.eq(best_power)].groupby('target_mode'):
    gg=g.sort_values('k'); axes[1].plot(gg.k,gg.mean_cv_72h_rmse_km,'-o',label=mode)
axes[1].axvline(best_k,color='black',ls='--',alpha=.5); axes[1].set_xscale('log'); axes[1].set_xticks(K_VALUES,labels=K_VALUES); axes[1].set_title(f'Stage 2: {best["name"]}, p={best_power:g}'); axes[1].set_xlabel('K'); axes[1].set_ylabel('CV 72-h RMSE (km)'); axes[1].legend(frameon=False)
plt.tight_layout(); fig.savefig(OUTPUT_DIR/'optimised_knn_cv.png',dpi=180,bbox_inches='tight'); plt.show()


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


A,B,xmean,xscale,pca=transform_representation(X[train_idx],X[test_idx],best_rep); ni,nd=neighbours(A,B,best_k)
train_target=Y[train_idx] if best_mode=='direct' else Y[train_idx]-persistence_Y[train_idx]
offset=np.zeros_like(Y[test_idx]) if best_mode=='direct' else persistence_Y[test_idx]
pred_flat,eff=weighted_prediction(train_target,ni,nd,best_k,best_power); pred_flat+=offset; pred=pred_flat.reshape(-1,72,2)
test_origins=origins.loc[test_idx].reset_index(drop=True); olat=np.repeat(test_origins.origin_latitude.to_numpy(),72); olon=np.repeat(test_origins.origin_longitude.to_numpy(),72)
plat,plon=local_km_to_latlon(olat,olon,pred[:,:,0].ravel(),pred[:,:,1].ravel())
truth=targ[targ.forecast_case_id.isin(test_origins.forecast_case_id)].sort_values(['forecast_case_id','lead_time_hours']).reset_index(drop=True)
errors=haversine_km(plat,plon,truth.latitude.to_numpy(),truth.longitude.to_numpy()).reshape(-1,72)
lead_metrics=pd.DataFrame({'lead_time_hours':np.arange(1,73),'median_error_km':np.median(errors,axis=0),'rmse_km':np.sqrt(np.mean(errors**2,axis=0))})
ite=np.trapezoid(np.column_stack([np.zeros(len(errors)),errors]),x=np.arange(73),axis=1)
case_metrics=pd.DataFrame({'forecast_case_id':test_origins.forecast_case_id,'drifter_id':test_origins.ID,'error_72h_km':errors[:,-1],'ite_km_h':ite})
display(lead_metrics[lead_metrics.lead_time_hours.isin([24,48,72])].round(3)); print('median ITE',round(float(np.median(ite)),2),'median effective neighbours',round(float(np.median(eff)),2))


# ## Analysis stage 8
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def load72(fn,label):
    d=pd.read_csv(OUTPUT_DIR/fn); r=d.loc[d.lead_time_hours.eq(72)].iloc[0]; return {'model':label,'median_72h_km':r.median_error_km,'rmse_72h_km':r.rmse_km}
comparison=[{'model':'Optimised Weighted KNN','median_72h_km':lead_metrics.iloc[-1].median_error_km,'rmse_72h_km':lead_metrics.iloc[-1].rmse_km}]
refs=[('knn_lead_metrics.csv','Original Weighted KNN'),('optimised_fnn_lead_metrics.csv','Optimised FNN'),('optimised_random_forest_lead_metrics.csv','Optimised Random Forest'),('spatial_mean_flow_lead_metrics.csv','Spatial mean flow'),('velocity_persistence_lead_metrics.csv','Velocity persistence')]
for fn,label in refs:
    if (OUTPUT_DIR/fn).exists(): comparison.append(load72(fn,label))
comparison=pd.DataFrame(comparison).sort_values('rmse_72h_km'); display(comparison.round(3))

def paired_bootstrap(path,label):
    ref=pd.read_csv(path); col='final_72h_error_km' if 'final_72h_error_km' in ref else 'error_72h_km'
    ref_small=ref[['forecast_case_id',col]].rename(columns={col:'ref_error'})
    m=case_metrics.merge(ref_small,on='forecast_case_id')
    groups={k:g for k,g in m.groupby('drifter_id')}; ids=np.array(list(groups)); rng=np.random.default_rng(2026); diffs=[]
    for _ in range(300 if FAST_MODE else 5000):
        parts=[groups[i] for i in rng.choice(ids,len(ids),replace=True)]; a=np.concatenate([x.error_72h_km for x in parts]); b=np.concatenate([x.ref_error for x in parts])
        diffs.append(np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b)))
    point=np.sqrt(np.mean(m.error_72h_km**2))-np.sqrt(np.mean(m.ref_error**2)); lo,hi=np.quantile(diffs,[.025,.975])
    return {'contrast':f'Optimised KNN - {label} RMSE','point_difference_km':point,'ci_2.5_km':lo,'ci_97.5_km':hi}

boot=[]
for fn,label in [('knn_test_case_metrics.csv','Original KNN'),('optimised_fnn_test_case_metrics.csv','Optimised FNN')]:
    if (OUTPUT_DIR/fn).exists(): boot.append(paired_bootstrap(OUTPUT_DIR/fn,label))
bootstrap=pd.DataFrame(boot); display(bootstrap.round(3))

fig,ax=plt.subplots(figsize=(9,5)); q=comparison.sort_values('rmse_72h_km',ascending=False); colors=['#1b9e77' if x=='Optimised Weighted KNN' else '#aaaaaa' for x in q.model]
ax.barh(q.model,q.rmse_72h_km,color=colors); [ax.text(v+.5,i,f'{v:.2f}',va='center') for i,v in enumerate(q.rmse_72h_km)]; ax.set_xlabel('72-h Haversine RMSE (km; lower is better)'); ax.set_title('Optimised Weighted KNN versus existing models')
plt.tight_layout(); fig.savefig(OUTPUT_DIR/'optimised_knn_vs_models.png',dpi=180,bbox_inches='tight'); plt.show()


# In[ ]:


new =comparison [comparison .model .eq ('Optimised Weighted KNN')].iloc [0 ];old =comparison [comparison .model .eq ('Original Weighted KNN')].iloc [0 ]
display (Markdown (f"""### \u5b9e\u9645\u7ed3\u8bba

- \u5f3a\u5316KNN\u768472 h RMSE\u4e3a **{new .rmse_72h_km :.2f} km**，\u65e7KNN\u4e3a **{old .rmse_72h_km :.2f} km**，\u76f8\u5bf9\u53d8\u5316 **{100 *(old .rmse_72h_km -new .rmse_72h_km )/old .rmse_72h_km :.2f}%**。
- 72 h median error\u4ece **{old .median_72h_km :.2f} km**\u53d8\u4e3a **{new .median_72h_km :.2f} km**。
- \u6700\u4f18\u8bbe\u7f6e：`{best_rep ['name']}`，$K={best_k }$，$p={best_power :g}$，\u76ee\u6807\u5f62\u5f0f\u4e3a `{best_mode }`。
- \u662f\u5426\u80fd\u5ba3\u79f0\u4f18\u4e8e\u65e7KNN\u6216\u5f3a\u5316FNN，\u8981\u770bdrifter-level bootstrap\u533a\u95f4\u662f\u5426\u5b8c\u5168\u4f4e\u4e8e0。
"""))


# ## Analysis stage 9
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


stage1_folds .to_csv (OUTPUT_DIR /'optimised_knn_stage1_folds.csv',index =False );stage1_summary .to_csv (OUTPUT_DIR /'optimised_knn_stage1_summary.csv',index =False )
stage2_folds .to_csv (OUTPUT_DIR /'optimised_knn_stage2_folds.csv',index =False );stage2_summary .to_csv (OUTPUT_DIR /'optimised_knn_stage2_summary.csv',index =False )
lead_metrics .to_csv (OUTPUT_DIR /'optimised_knn_lead_metrics.csv',index =False );case_metrics .to_csv (OUTPUT_DIR /'optimised_knn_test_case_metrics.csv',index =False )
comparison .to_csv (OUTPUT_DIR /'optimised_knn_72h_comparison.csv',index =False );bootstrap .to_csv (OUTPUT_DIR /'optimised_knn_bootstrap.csv',index =False )
with open (OUTPUT_DIR /'optimised_knn_best_parameters.json','w')as f :json .dump ({'representation':best_rep ,'k':best_k ,'power':best_power ,'target_mode':best_mode ,'selection_metric':'grouped 5-fold CV 72h Haversine RMSE'},f ,indent =2 )
print ('\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230',OUTPUT_DIR )

