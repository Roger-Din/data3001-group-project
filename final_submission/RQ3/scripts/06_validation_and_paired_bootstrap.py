"""RQ3-only validation: paired drifter bootstrap, regime provenance, RQ1 consistency.
Run from anywhere: python RQ3/06_validation_and_paired_bootstrap.py
No model fitting or retuning; reads the fixed prediction cache and original data.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PROJECT=Path(__file__).resolve().parents[2]
DATA=PROJECT/'data'/'processed'
OUT=PROJECT/'RQ3'/'optimised_fnn'/'output'
OUT.mkdir(parents=True,exist_ok=True)
R=6371.0
REGIMES=['Shelf','Kuroshio Core','Transition','Other Offshore']
N_BOOT=5000
SEED=3001
checks=[]
def check(name,passed,detail=''):
    checks.append({'check':name,'status':'PASS' if passed else 'FAIL','detail':str(detail)})
    print(('PASS' if passed else 'FAIL')+' | '+name+' | '+str(detail))
    return passed

def haversine(lat1,lon1,lat2,lon2):
    lat1,lon1,lat2,lon2=map(np.deg2rad,[lat1,lon1,lat2,lon2])
    a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(a,0,1)))

def local_to_latlon(olat,olon,east,north):
    lat0,lon0=np.deg2rad(olat),np.deg2rad(olon)
    angle=np.hypot(east,north)/R; bearing=np.arctan2(east,north)
    lat=np.arcsin(np.sin(lat0)*np.cos(angle)+np.cos(lat0)*np.sin(angle)*np.cos(bearing))
    lon=lon0+np.arctan2(np.sin(bearing)*np.sin(angle)*np.cos(lat0),np.cos(angle)-np.sin(lat0)*np.sin(lat))
    return np.rad2deg(lat),(np.rad2deg(lon)+180)%360-180

def qcp(x,level=.9):
    x=np.sort(np.asarray(x,float)); rank=int(np.ceil((len(x)+1)*level))
    if not len(x) or rank>len(x):raise ValueError('Insufficient calibration cases')
    return float(x[rank-1])

origins=pd.read_parquet(DATA/'forecast_origins.parquet').sort_values('forecast_case_id').reset_index(drop=True)
regimes=pd.read_csv(PROJECT/'RQ2'/'data'/'forecast_case_regimes.csv')
check('Unique origin case IDs',origins.forecast_case_id.is_unique)
check('Unique regime case IDs',regimes.forecast_case_id.is_unique)
meta=origins.merge(regimes[['forecast_case_id','regime']],on='forecast_case_id',how='left',validate='one_to_one',sort=False)
check('All cases have regime records',len(meta)==len(origins) and meta.regime.notna().all())
cal=meta.loc[meta.partition.eq('calibration')].reset_index(drop=True)
test=meta.loc[meta.partition.eq('test')].reset_index(drop=True)
train=meta.loc[meta.partition.eq('training')].reset_index(drop=True)
check('Expected partition sizes',(len(train),len(cal),len(test))==(3591,769,757),f'{len(train)}/{len(cal)}/{len(test)}')
check('Drifter-disjoint partitions',all(set(a.ID).isdisjoint(set(b.ID)) for a,b in [(train,cal),(train,test),(cal,test)]))
check('Known evaluation regimes',set(cal.regime).issubset(REGIMES) and set(test.regime).issubset(REGIMES))

cfg=PROJECT/'RQ3'/'config'/'optimised_fnn_best_parameters.json'
cache=OUT/'optimised_fnn_calibration_test_predictions.npz'
with np.load(cache,allow_pickle=False) as z:
    pred_cal=z['fnn_cal'].copy(); pred_test=z['fnn_test'].copy()
    check('Cache configuration hash matches JSON',str(z['config_hash'])==hashlib.sha256(cfg.read_bytes()).hexdigest())
    check('Calibration case IDs and order match',np.array_equal(z['cal_case_ids'],cal.forecast_case_id.to_numpy()))
    check('Test case IDs and order match',np.array_equal(z['test_case_ids'],test.forecast_case_id.to_numpy()))
check('Finite predictions and expected dimensions',pred_cal.shape==(len(cal),144) and pred_test.shape==(len(test),144) and np.isfinite(pred_cal).all() and np.isfinite(pred_test).all())

obs=pd.read_parquet(DATA/'region_A_observations.parquet',columns=['ID','segment_id','segment_step','latitude','longitude'])
def endpoint_errors(cases,pred):
    target=cases[['forecast_case_id','ID','segment_id','target_start_step','origin_latitude','origin_longitude']].copy()
    target['segment_step']=target.target_start_step+71
    target=target.merge(obs,on=['ID','segment_id','segment_step'],how='left',validate='many_to_one',sort=False)
    check('All 72h targets available',target.latitude.notna().all() and target.longitude.notna().all())
    xy=pred.reshape(-1,72,2)[:,-1,:]
    plat,plon=local_to_latlon(target.origin_latitude.to_numpy(),target.origin_longitude.to_numpy(),xy[:,0],xy[:,1])
    return haversine(plat,plon,target.latitude.to_numpy(),target.longitude.to_numpy())
err_cal=endpoint_errors(cal,pred_cal)
err_test=endpoint_errors(test,pred_test)
check('Finite nonnegative endpoint errors',np.isfinite(err_cal).all() and np.isfinite(err_test).all() and (err_cal>=0).all() and (err_test>=0).all())

summary=pd.read_csv(OUT/'optimised_fnn_conformal_regime_summary.csv')
q_global=qcp(err_cal)
rows=[]
for regime in REGIMES:
    cm=cal.regime.eq(regime).to_numpy(); tm=test.regime.eq(regime).to_numpy()
    if not cm.any() or not tm.any():raise ValueError(f'Missing evaluation regime: {regime}')
    qm=qcp(err_cal[cm])
    for method,radius in [('Global',q_global),('Mondrian',qm)]:
        coverage=float(np.mean(err_test[tm]<=radius))
        rows.append({'regime':regime,'method':method,'coverage_72h':coverage,'radius_72h_km':radius})
recalc=pd.DataFrame(rows)
compare=recalc.merge(summary,on=['regime','method'],suffixes=('_recomputed','_saved'),validate='one_to_one')
maxcov=float(np.max(np.abs(compare.coverage_72h_recomputed-compare.coverage_72h_saved)))
maxrad=float(np.max(np.abs(compare.radius_72h_km_recomputed-compare.radius_72h_km_saved)))
check('Saved regime coverage reproduced',maxcov<1e-10,f'max difference={maxcov:.3g}')
check('Saved regime radii reproduced',maxrad<1e-8,f'max difference={maxrad:.3g} km')

# Paired resampling: same test drifter draw for both methods; fixed calibration radii.
test_case=test[['forecast_case_id','ID','regime']].copy()
test_case['error_72h_km']=err_test
test_case['covered_global']=(err_test<=q_global).astype(int)
qmap={r:qcp(err_cal[cal.regime.eq(r).to_numpy()]) for r in REGIMES}
test_case['covered_mondrian']=(err_test<=test.regime.map(qmap).to_numpy()).astype(int)
test_case.to_csv(OUT/'optimised_fnn_case_level_coverage_audit.csv',index=False)
ids=test_case.ID.unique(); by_id={v:np.flatnonzero(test_case.ID.eq(v).to_numpy()) for v in ids}
rng=np.random.default_rng(SEED)
reps=[]
def summarize(indices):
    frame=test_case.iloc[indices]; res={}
    for regime in ['Overall']+REGIMES:
        sub=frame if regime=='Overall' else frame[frame.regime.eq(regime)]
        key=regime.lower().replace(' ','_')
        res[key+'_diff_pp']=100*(sub.covered_mondrian.mean()-sub.covered_global.mean()) if len(sub) else np.nan
    return res
observed=summarize(np.arange(len(test_case)))
for _ in range(N_BOOT):
    drawn=rng.choice(ids,size=len(ids),replace=True)
    reps.append(summarize(np.concatenate([by_id[v] for v in drawn])))
reps=pd.DataFrame(reps)
reps.to_csv(OUT/'optimised_fnn_paired_bootstrap_replicates.csv',index=False)
paired=[]
for metric,estimate in observed.items():
    values=reps[metric].dropna()
    paired.append({'metric':metric,'difference_pp':estimate,'ci95_lower_pp':values.quantile(.025),'ci95_upper_pp':values.quantile(.975),'valid_replicates':len(values)})
pd.DataFrame(paired).to_csv(OUT/'optimised_fnn_paired_bootstrap_summary.csv',index=False)

# Cross-RQ: compare saved per-case errors if provided. Never silently call a missing file a pass.
rq1=PROJECT/'output'/'rq1'/'zh'/'optimised_fnn_test_case_metrics.csv'
if rq1.is_file():
    saved=pd.read_csv(rq1)
    metric=next((c for c in ['final_72h_error_km','error_72h_km'] if c in saved.columns),None)
    if metric is None or 'forecast_case_id' not in saved:
        checks.append({'check':'RQ1 per-case consistency','status':'FAIL','detail':'RQ1 file lacks expected case ID/error columns'})
    else:
        m=test_case[['forecast_case_id','error_72h_km']].merge(saved[['forecast_case_id',metric]],on='forecast_case_id',how='outer',indicator=True,validate='one_to_one')
        allmatch=m['_merge'].eq('both').all()
        diff=float(np.max(np.abs(m.error_72h_km-m[metric]))) if allmatch else float('nan')
        check('RQ1 per-case consistency',allmatch and diff<1e-5,f'max abs error difference={diff} km')
        m.to_csv(OUT/'optimised_fnn_rq1_consistency_audit.csv',index=False)
else:
    checks.append({'check':'RQ1 per-case consistency','status':'NOT_VERIFIED','detail':f'Independent reference not available: {rq1}'})
    print('NOT_VERIFIED | RQ1 per-case consistency | independent RQ1 metrics unavailable')

# Regime provenance: observable checks do not prove how the upstream labels were generated.
future_cols=[c for c in regimes.columns if any(k in c.lower() for k in ['target','future','lead','error_','predicted','actual_72'])]
check('No explicit future/target columns in regime file',not future_cols,','.join(future_cols) or 'none')
origin_cols=[c for c in ['origin_latitude','origin_longitude','ID','segment_id'] if c in regimes.columns and c in origins.columns]
for col in origin_cols:
    merged=origins[['forecast_case_id',col]].merge(regimes[['forecast_case_id',col]],on='forecast_case_id',suffixes=('_origin','_label'),validate='one_to_one')
    a=merged[col+'_origin'];b=merged[col+'_label']
    check('Regime metadata origin consistency: '+col,bool(np.allclose(a,b,equal_nan=True)) if pd.api.types.is_numeric_dtype(a) else a.equals(b))
checks.append({'check':'Regime generation uses prediction-time information only','status':'NOT_VERIFIED','detail':'Requires independent review of upstream RQ2 regime-generation script and its input timestamps; output CSV alone cannot prove provenance.'})
print('NOT_VERIFIED | Regime generation provenance | review upstream RQ2 label-generation code')

report=pd.DataFrame(checks)
report.to_csv(OUT/'optimised_fnn_validation_checks.csv',index=False)
print('\nPaired bootstrap summary:')
print(pd.DataFrame(paired).to_string(index=False))
print('\nValidation status:',report.status.value_counts().to_dict())
if (report.status=='FAIL').any():sys.exit('One or more checks failed; inspect optimised_fnn_validation_checks.csv')
