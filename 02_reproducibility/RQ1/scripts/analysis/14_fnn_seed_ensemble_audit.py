#!/usr/bin/env python
# coding: utf-8

# # 14 FNN Seed Ensemble Audit
# 
# Checks the sensitivity of the selected FNN to random seeds and ensemble size.
# 
# **Provenance.** This is an English-facing execution copy of the original project notebook. The computational cells are preserved so that the analysis remains reproducible. Chinese narrative cells have been replaced by concise English stage notes; the original notebook remains unchanged in the source project.

# ## Analysis stage 1
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


from pathlib import Path 
import copy ,json ,os ,time 

import matplotlib .pyplot as plt 
import numpy as np 
import pandas as pd 
from IPython import get_ipython 
from IPython .display import display ,Markdown 

ipython =get_ipython ()
if ipython is not None :
    ipython .run_line_magic ('matplotlib','inline')

PROJECT_ROOT =next ((p for p in [Path .cwd (),*Path .cwd ().parents ]
if (p /'data'/'processed').exists ()),None )
if PROJECT_ROOT is None :
    raise FileNotFoundError ('\u627e\u4e0d\u5230 data/processed；\u8bf7\u4ece\u9879\u76ee\u6839\u76ee\u5f55\u6216\u5176\u5b50\u76ee\u5f55\u8fd0\u884c。')

PROCESSED =PROJECT_ROOT /'data'/'processed'
OUTPUT_DIR =PROJECT_ROOT /'output'/'rq1'/'zh'
OUTPUT_DIR .mkdir (parents =True ,exist_ok =True )

EARTH_RADIUS_KM =6371.0 
HISTORY_HOURS ,HORIZON_HOURS ,N_FOLDS =24 ,72 ,5 
RANDOM_SEED =3001 
FAST_MODE =False 
MAX_EPOCHS =160 
PATIENCE =18 

FNN_CONFIGS =[
dict (name ='64 direct',hidden_sizes =(64 ,),l2 =1e-4 ,learning_rate =1e-3 ,batch_size =256 ,target_mode ='direct'),
dict (name ='128-64 direct',hidden_sizes =(128 ,64 ),l2 =1e-4 ,learning_rate =1e-3 ,batch_size =256 ,target_mode ='direct'),
dict (name ='128-64 direct L2',hidden_sizes =(128 ,64 ),l2 =1e-3 ,learning_rate =1e-3 ,batch_size =256 ,target_mode ='direct'),
dict (name ='256-128 direct',hidden_sizes =(256 ,128 ),l2 =1e-4 ,learning_rate =5e-4 ,batch_size =256 ,target_mode ='direct'),
dict (name ='128-128-64 direct',hidden_sizes =(128 ,128 ,64 ),l2 =1e-4 ,learning_rate =5e-4 ,batch_size =128 ,target_mode ='direct'),
dict (name ='128-64 residual',hidden_sizes =(128 ,64 ),l2 =1e-4 ,learning_rate =1e-3 ,batch_size =256 ,target_mode ='persistence_residual'),
dict (name ='256-128 residual',hidden_sizes =(256 ,128 ),l2 =1e-4 ,learning_rate =5e-4 ,batch_size =256 ,target_mode ='persistence_residual'),
dict (name ='256-128-64 residual',hidden_sizes =(256 ,128 ,64 ),l2 =1e-3 ,learning_rate =5e-4 ,batch_size =128 ,target_mode ='persistence_residual'),
]
if FAST_MODE :
    FNN_CONFIGS =FNN_CONFIGS [:2 ]

plt .style .use ('seaborn-v0_8-whitegrid')
plt .rcParams ['font.sans-serif']=['PingFang SC','Arial Unicode MS','DejaVu Sans']
plt .rcParams ['axes.unicode_minus']=False 
print ('\u9879\u76ee\u76ee\u5f55：',PROJECT_ROOT )
print ('\u8fd0\u884c\u6a21\u5f0f：','\u5feb\u901f\u68c0\u67e5'if FAST_MODE else '\u5b8c\u6574\u5f3a\u5316\u8bad\u7ec3')
AUDIT_DIR =OUTPUT_DIR /'fnn_seed_ensemble_audit'
AUDIT_DIR .mkdir (parents =True ,exist_ok =True )


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def latlon_to_local_km (origin_lat ,origin_lon ,latitude ,longitude ):
    lat0 ,lon0 =np .deg2rad (origin_lat ),np .deg2rad (origin_lon )
    lat1 ,lon1 =np .deg2rad (latitude ),np .deg2rad (longitude )
    dlon =(lon1 -lon0 +np .pi )%(2 *np .pi )-np .pi 
    dlat =lat1 -lat0 
    a =np .sin (dlat /2 )**2 +np .cos (lat0 )*np .cos (lat1 )*np .sin (dlon /2 )**2 
    angle =2 *np .arcsin (np .sqrt (np .clip (a ,0 ,1 )))
    bearing =np .arctan2 (np .sin (dlon )*np .cos (lat1 ),
    np .cos (lat0 )*np .sin (lat1 )-np .sin (lat0 )*np .cos (lat1 )*np .cos (dlon ))
    distance =EARTH_RADIUS_KM *angle 
    return distance *np .sin (bearing ),distance *np .cos (bearing )

def local_km_to_latlon (origin_lat ,origin_lon ,east_km ,north_km ):
    lat0 ,lon0 =np .deg2rad (origin_lat ),np .deg2rad (origin_lon )
    distance =np .hypot (east_km ,north_km )
    bearing =np .arctan2 (east_km ,north_km )
    angular =distance /EARTH_RADIUS_KM 
    lat =np .arcsin (np .sin (lat0 )*np .cos (angular )+np .cos (lat0 )*np .sin (angular )*np .cos (bearing ))
    lon =lon0 +np .arctan2 (np .sin (bearing )*np .sin (angular )*np .cos (lat0 ),
    np .cos (angular )-np .sin (lat0 )*np .sin (lat ))
    return np .rad2deg (lat ),(np .rad2deg (lon )+180 )%360 -180 

def haversine_km (lat1 ,lon1 ,lat2 ,lon2 ):
    lat1 ,lon1 ,lat2 ,lon2 =map (np .deg2rad ,[lat1 ,lon1 ,lat2 ,lon2 ])
    a =np .sin ((lat2 -lat1 )/2 )**2 +np .cos (lat1 )*np .cos (lat2 )*np .sin ((lon2 -lon1 )/2 )**2 
    return 2 *EARTH_RADIUS_KM *np .arcsin (np .sqrt (np .clip (a ,0 ,1 )))

origins =pd .read_parquet (PROCESSED /'forecast_origins.parquet').sort_values ('forecast_case_id').reset_index (drop =True )
observations =pd .read_parquet (PROCESSED /'region_A_observations.parquet',
columns =['ID','segment_id','segment_step','time','latitude','longitude','ve','vn'])
n_cases =len (origins )

history =origins [['forecast_case_id','ID','segment_id','history_start_step','origin_latitude','origin_longitude']].loc [
origins .index .repeat (HISTORY_HOURS )].reset_index (drop =True )
history ['history_index']=np .tile (np .arange (HISTORY_HOURS ),n_cases )
history ['segment_step']=history ['history_start_step']+history ['history_index']
history =history .merge (observations ,on =['ID','segment_id','segment_step'],how ='left',validate ='many_to_one')
history =history .sort_values (['forecast_case_id','history_index']).reset_index (drop =True )
history ['relative_east_km'],history ['relative_north_km']=latlon_to_local_km (
history .origin_latitude ,history .origin_longitude ,history .latitude ,history .longitude )
history_values =history [['relative_east_km','relative_north_km','ve','vn']].to_numpy ().reshape (n_cases ,24 ,4 )
X =np .column_stack ([origins .origin_latitude ,origins .origin_longitude ,
history_values .reshape (n_cases ,-1 )]).astype (float )

targets =origins [['forecast_case_id','ID','segment_id','target_start_step','origin_latitude','origin_longitude']].loc [
origins .index .repeat (HORIZON_HOURS )].reset_index (drop =True )
targets ['lead_time_hours']=np .tile (np .arange (1 ,HORIZON_HOURS +1 ),n_cases )
targets ['segment_step']=targets ['target_start_step']+targets ['lead_time_hours']-1 
targets =targets .merge (observations [['ID','segment_id','segment_step','latitude','longitude']],
on =['ID','segment_id','segment_step'],how ='left',validate ='many_to_one')
targets =targets .sort_values (['forecast_case_id','lead_time_hours']).reset_index (drop =True )
targets ['target_east_km'],targets ['target_north_km']=latlon_to_local_km (
targets .origin_latitude ,targets .origin_longitude ,targets .latitude ,targets .longitude )
Y =targets [['target_east_km','target_north_km']].to_numpy ().reshape (n_cases ,144 ).astype (float )

# Implementation note translated in the organised English notebook.
hours =np .arange (1 ,73 )[None ,:]
last_velocity =history_values [:,-1 ,2 :4 ]
persistence_Y =np .stack ([last_velocity [:,0 ,None ]*3.6 *hours ,
last_velocity [:,1 ,None ]*3.6 *hours ],axis =-1 ).reshape (n_cases ,144 )

train_indices =np .flatnonzero (origins .partition .eq ('training'))
calibration_indices =np .flatnonzero (origins .partition .eq ('calibration'))
test_indices =np .flatnonzero (origins .partition .eq ('test'))
assert X .shape ==(5117 ,98 )and Y .shape ==(5117 ,144 )
assert (len (train_indices ),len (calibration_indices ),len (test_indices ))==(3591 ,769 ,757 )
print ('X:',X .shape ,'Y:',Y .shape ,'train/cal/test:',(3591 ,769 ,757 ))


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def initialise (layer_sizes ,rng ):
    return [{'W':rng .normal (0 ,np .sqrt (2 /fan_in ),(fan_in ,fan_out )),'b':np .zeros (fan_out )}
    for fan_in ,fan_out in zip (layer_sizes [:-1 ],layer_sizes [1 :])]

def forward (parameters ,Xb ):
    acts ,zs ,cur =[Xb ],[],Xb 
    for j ,layer in enumerate (parameters ):
        z =cur @layer ['W']+layer ['b'];zs .append (z )
        cur =np .maximum (z ,0 )if j <len (parameters )-1 else z 
        acts .append (cur )
    return cur ,acts ,zs 

def loss_and_grads (parameters ,Xb ,Yb ,l2 ):
    pred ,acts ,zs =forward (parameters ,Xb )
    residual =pred -Yb 
    # Implementation note translated in the organised English notebook.
    loss =np .mean (residual **2 )+l2 *sum (np .sum (p ['W']**2 )for p in parameters )
    delta =2 *residual /residual .size 
    grads =[None ]*len (parameters )
    for j in range (len (parameters )-1 ,-1 ,-1 ):
        grads [j ]={'W':acts [j ].T @delta +2 *l2 *parameters [j ]['W'],'b':delta .sum (0 )}
        if j >0 :delta =(delta @parameters [j ]['W'].T )*(zs [j -1 ]>0 )
    return loss ,grads 

def endpoint_errors (pred_abs ,true_abs ,origin_rows ):
    p =pred_abs .reshape (-1 ,72 ,2 )[:,-1 ,:]
    y =true_abs .reshape (-1 ,72 ,2 )[:,-1 ,:]
    olat =origin_rows .origin_latitude .to_numpy ();olon =origin_rows .origin_longitude .to_numpy ()
    plat ,plon =local_km_to_latlon (olat ,olon ,p [:,0 ],p [:,1 ])
    tlat ,tlon =local_km_to_latlon (olat ,olon ,y [:,0 ],y [:,1 ])
    return haversine_km (plat ,plon ,tlat ,tlon )

def fit_fnn (Xtr_raw ,Ytr_model ,Xva_raw ,Yva_model ,Yva_abs ,va_origins ,
offset_valid ,config ,seed ,max_epochs =MAX_EPOCHS ,patience =PATIENCE ):
    xm ,xs =Xtr_raw .mean (0 ),Xtr_raw .std (0 );xs [xs <1e-12 ]=1 
    ym ,ys =Ytr_model .mean (0 ),Ytr_model .std (0 );ys [ys <1e-12 ]=1 
    Xtr =(Xtr_raw -xm )/xs ;Ytr =(Ytr_model -ym )/ys ;Xva =(Xva_raw -xm )/xs 
    rng =np .random .default_rng (seed )
    params =initialise ((98 ,*config ['hidden_sizes'],144 ),rng )
    m =[{'W':np .zeros_like (p ['W']),'b':np .zeros_like (p ['b'])}for p in params ]
    v =copy .deepcopy (m );step =0 ;best =np .inf ;best_epoch =0 ;best_params =None ;stale =0 ;rows =[]
    for epoch in range (1 ,max_epochs +1 ):
        losses =[]
        permutation =rng .permutation (len (Xtr ))
        for start in range (0 ,len (Xtr ),config ['batch_size']):
            batch =permutation [start :start +config ['batch_size']]
            loss ,grads =loss_and_grads (params ,Xtr [batch ],Ytr [batch ],config ['l2']);losses .append (loss );step +=1 
            for j ,(p ,g )in enumerate (zip (params ,grads )):
                for key in ('W','b'):
                    m [j ][key ]=.9 *m [j ][key ]+.1 *g [key ]
                    v [j ][key ]=.999 *v [j ][key ]+.001 *g [key ]**2 
                    mh =m [j ][key ]/(1 -.9 **step );vh =v [j ][key ]/(1 -.999 **step )
                    p [key ]-=config ['learning_rate']*mh /(np .sqrt (vh )+1e-8 )
        pred_model =forward (params ,Xva )[0 ]*ys +ym 
        pred_abs =pred_model +offset_valid 
        rmse =float (np .sqrt (np .mean (endpoint_errors (pred_abs ,Yva_abs ,va_origins )**2 )))
        rows .append ({'epoch':epoch ,'training_objective':float (np .mean (losses )),'validation_72h_rmse_km':rmse })
        if rmse <best -1e-4 :
            best ,best_epoch ,best_params ,stale =rmse ,epoch ,copy .deepcopy (params ),0 
        else :stale +=1 
        if stale >=patience :break 
    return dict (parameters =best_params ,x_mean =xm ,x_scale =xs ,y_mean =ym ,y_scale =ys ,
    best_epoch =best_epoch ,best_rmse =best ,history =pd .DataFrame (rows ))

def predict_model (model ,Xnew ):
    z =(Xnew -model ['x_mean'])/model ['x_scale']
    return forward (model ['parameters'],z )[0 ]*model ['y_scale']+model ['y_mean']


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


BEST_CONFIG =dict (
name ='128-64 direct L2',
hidden_sizes =(128 ,64 ),
l2 =0.001 ,
learning_rate =0.001 ,
batch_size =256 ,
target_mode ='direct',
)
SELECTED_EPOCHS =129 
SEEDS =[4000 ,2026 ,2037 ,2048 ]

def fit_final (Xraw ,Ymodel ,config ,epochs ,seed ):
    xm ,xs =Xraw .mean (0 ),Xraw .std (0 )
    xs [xs <1e-12 ]=1 
    ym ,ys =Ymodel .mean (0 ),Ymodel .std (0 )
    ys [ys <1e-12 ]=1 
    Xtr =(Xraw -xm )/xs 
    Ytr =(Ymodel -ym )/ys 
    rng =np .random .default_rng (seed )
    params =initialise ((98 ,*config ['hidden_sizes'],144 ),rng )
    m =[{'W':np .zeros_like (p ['W']),'b':np .zeros_like (p ['b'])}for p in params ]
    v =copy .deepcopy (m )
    step =0 
    for epoch in range (epochs ):
        permutation =rng .permutation (len (Xtr ))
        for start in range (0 ,len (Xtr ),config ['batch_size']):
            batch =permutation [start :start +config ['batch_size']]
            _ ,grads =loss_and_grads (params ,Xtr [batch ],Ytr [batch ],config ['l2'])
            step +=1 
            for j ,(p ,g )in enumerate (zip (params ,grads )):
                for key in ('W','b'):
                    m [j ][key ]=.9 *m [j ][key ]+.1 *g [key ]
                    v [j ][key ]=.999 *v [j ][key ]+.001 *g [key ]**2 
                    mh =m [j ][key ]/(1 -.9 **step )
                    vh =v [j ][key ]/(1 -.999 **step )
                    p [key ]-=config ['learning_rate']*mh /(np .sqrt (vh )+1e-8 )
    return dict (parameters =params ,x_mean =xm ,x_scale =xs ,y_mean =ym ,y_scale =ys )

def save_model_npz (model ,seed ):
    archive ={
    'seed':np .array (seed ),
    'hidden_sizes':np .array (BEST_CONFIG ['hidden_sizes']),
    'l2':np .array (BEST_CONFIG ['l2']),
    'learning_rate':np .array (BEST_CONFIG ['learning_rate']),
    'batch_size':np .array (BEST_CONFIG ['batch_size']),
    'epochs':np .array (SELECTED_EPOCHS ),
    'x_mean':model ['x_mean'],'x_scale':model ['x_scale'],
    'y_mean':model ['y_mean'],'y_scale':model ['y_scale'],
    }
    for layer_number ,layer in enumerate (model ['parameters'],start =1 ):
        archive [f'W{layer_number }']=layer ['W']
        archive [f'b{layer_number }']=layer ['b']
    np .savez_compressed (AUDIT_DIR /f'fnn_seed_{seed }_parameters.npz',**archive )

def load_model_npz (path ):
    z =np .load (path )
    parameters =[]
    layer_number =1 
    while f'W{layer_number }'in z .files :
        parameters .append ({'W':z [f'W{layer_number }'],'b':z [f'b{layer_number }']})
        layer_number +=1 
    return dict (parameters =parameters ,x_mean =z ['x_mean'],x_scale =z ['x_scale'],
    y_mean =z ['y_mean'],y_scale =z ['y_scale'])

member_predictions ={}
training_times ={}
for seed in SEEDS :
    started =time .time ()
    model =fit_final (X [train_indices ],Y [train_indices ],BEST_CONFIG ,SELECTED_EPOCHS ,seed )
    prediction =predict_model (model ,X [test_indices ])
    save_model_npz (model ,seed )
    np .savez_compressed (AUDIT_DIR /f'fnn_seed_{seed }_test_predictions.npz',
    forecast_case_id =origins .loc [test_indices ,'forecast_case_id'].to_numpy (),
    prediction =prediction )
    # Implementation note translated in the organised English notebook.
    reloaded =load_model_npz (AUDIT_DIR /f'fnn_seed_{seed }_parameters.npz')
    reloaded_prediction =predict_model (reloaded ,X [test_indices ])
    assert np .allclose (prediction ,reloaded_prediction ,rtol =0 ,atol =1e-12 )
    member_predictions [seed ]=prediction 
    training_times [seed ]=time .time ()-started 
    print (f'seed {seed }: \u5b8c\u6210，{training_times [seed ]:.1f}s，checkpoint\u91cd\u65b0\u52a0\u8f7d\u9a8c\u8bc1\u901a\u8fc7')


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


test_origins = origins.loc[test_indices].reset_index(drop=True)
true_targets = targets[targets.forecast_case_id.isin(test_origins.forecast_case_id)].sort_values(
    ['forecast_case_id','lead_time_hours']).reset_index(drop=True)

def evaluate_prediction(name, prediction_flat):
    prediction = prediction_flat.reshape(-1, 72, 2)
    flat_olat = np.repeat(test_origins.origin_latitude.to_numpy(), 72)
    flat_olon = np.repeat(test_origins.origin_longitude.to_numpy(), 72)
    plat, plon = local_km_to_latlon(flat_olat, flat_olon,
                                    prediction[:,:,0].ravel(), prediction[:,:,1].ravel())
    errors = haversine_km(plat, plon, true_targets.latitude.to_numpy(),
                          true_targets.longitude.to_numpy()).reshape(-1,72)
    ite = np.trapezoid(np.column_stack([np.zeros(len(errors)), errors]), x=np.arange(73), axis=1)
    case_table = pd.DataFrame({
        'forecast_case_id': test_origins.forecast_case_id.to_numpy(),
        'drifter_id': test_origins.ID.to_numpy(),
        'model': name,
        'error_72h_km': errors[:,-1],
        'ite_km_h': ite,
    })
    summary = dict(
        model=name,
        median_72h_km=float(np.median(errors[:,-1])),
        mean_72h_km=float(np.mean(errors[:,-1])),
        rmse_72h_km=float(np.sqrt(np.mean(errors[:,-1]**2))),
        median_ite_km_h=float(np.median(ite)),
        n_error_gt_100km=int(np.sum(errors[:,-1] > 100)),
        n_error_gt_150km=int(np.sum(errors[:,-1] > 150)),
    )
    return summary, case_table

predictions = {f'Seed {seed}': pred for seed, pred in member_predictions.items()}
predictions['Ensemble 2026+2037+2048'] = np.mean(
    [member_predictions[s] for s in (2026,2037,2048)], axis=0)
predictions['Ensemble all four seeds'] = np.mean(list(member_predictions.values()), axis=0)

summary_rows, case_tables = [], []
for name, prediction in predictions.items():
    summary, cases = evaluate_prediction(name, prediction)
    summary_rows.append(summary)
    case_tables.append(cases)
summary_table = pd.DataFrame(summary_rows)
case_table = pd.concat(case_tables, ignore_index=True)

display(summary_table.round(3))
summary_table.to_csv(AUDIT_DIR/'fnn_seed_ensemble_summary.csv', index=False)
case_table.to_csv(AUDIT_DIR/'fnn_seed_ensemble_case_metrics.csv', index=False)
np.savez_compressed(AUDIT_DIR/'fnn_ensemble_2026_2037_2048_test_predictions.npz',
                    forecast_case_id=test_origins.forecast_case_id.to_numpy(),
                    prediction=predictions['Ensemble 2026+2037+2048'])


# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


single =summary_table [summary_table .model .str .startswith ('Seed')].copy ()
new_single =single [single .model .isin (['Seed 2026','Seed 2037','Seed 2048'])]
ensemble_row =summary_table [summary_table .model .eq ('Ensemble 2026+2037+2048')].iloc [0 ]
original_row =summary_table [summary_table .model .eq ('Seed 4000')].iloc [0 ]

diagnosis =pd .DataFrame ({
'comparison':[
'Original seed 4000 RMSE',
'\u4e09\u4e2a\u65b0seed\u7684\u5e73\u5747\u5355\u6a21\u578bRMSE',
'\u4e09\u4e2a\u65b0seed\u4e2d\u7684\u6700\u4f73\u5355\u6a21\u578bRMSE',
'\u4e09\u4e2a\u65b0seed ensemble RMSE',
'ensemble\u76f8\u5bf9seed 4000\u6539\u5584',
'ensemble\u76f8\u5bf9\u65b0seed\u5e73\u5747\u6539\u5584',
'ensemble\u76f8\u5bf9\u65b0seed\u6700\u4f73\u6539\u5584',
],
'value_km':[
original_row .rmse_72h_km ,
new_single .rmse_72h_km .mean (),
new_single .rmse_72h_km .min (),
ensemble_row .rmse_72h_km ,
original_row .rmse_72h_km -ensemble_row .rmse_72h_km ,
new_single .rmse_72h_km .mean ()-ensemble_row .rmse_72h_km ,
new_single .rmse_72h_km .min ()-ensemble_row .rmse_72h_km ,
]
})
display (diagnosis .round (4 ))

seed_effect =original_row .rmse_72h_km -new_single .rmse_72h_km .mean ()
ensemble_effect =new_single .rmse_72h_km .mean ()-ensemble_row .rmse_72h_km 
best_gap =new_single .rmse_72h_km .min ()-ensemble_row .rmse_72h_km 

if ensemble_effect >0 and best_gap >=0 :
    verdict ='ensemble\u4f18\u4e8e\u4e09\u4e2a\u65b0\u5355\u6a21\u578b\u7684\u5e73\u5747\u8868\u73b0，\u5e76\u4e14\u4e0d\u5dee\u4e8e\u5176\u4e2d\u6700\u4f73\u5355\u6a21\u578b；\u6539\u5584\u4e3b\u8981\u5f97\u5230\u96c6\u6210\u5e73\u5747\u7684\u76f4\u63a5\u652f\u6301。'
elif ensemble_effect >0 and best_gap <0 :
    verdict ='ensemble\u4f18\u4e8e\u4e09\u4e2a\u65b0\u5355\u6a21\u578b\u7684\u5e73\u5747\u8868\u73b0，\u4f46\u4ecd\u6709\u67d0\u4e2a\u5355seed\u66f4\u597d；\u6539\u5584\u540c\u65f6\u5305\u542b\u96c6\u6210\u964d\u65b9\u5dee\u548c\u968f\u673a\u79cd\u5b50\u5dee\u5f02。'
else :
    verdict ='ensemble\u6ca1\u6709\u4f18\u4e8e\u4e09\u4e2a\u65b0\u5355\u6a21\u578b\u7684\u5e73\u5747\u8868\u73b0；\u76f8\u5bf9Original\u7684\u6539\u5584\u4e3b\u8981\u6765\u81ea\u65b0\u968f\u673a\u79cd\u5b50\u6574\u4f53\u66f4\u6709\u5229。'

print ('\u968f\u673a\u79cd\u5b50\u6574\u4f53\u53d8\u5316\u8d21\u732e（seed 4000 RMSE - \u65b0seed\u5e73\u5747RMSE）：',round (seed_effect ,4 ),'km')
print ('\u96c6\u6210\u5e73\u5747\u8d21\u732e（\u65b0seed\u5e73\u5747RMSE - ensemble RMSE）：',round (ensemble_effect ,4 ),'km')
print ('ensemble\u76f8\u5bf9\u6700\u4f73\u65b0\u5355\u6a21\u578b\u5dee\u503c（\u6700\u4f73\u5355\u6a21\u578b - ensemble）：',round (best_gap ,4 ),'km')
print ('\n\u7ed3\u8bba：',verdict )


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


plot_table =summary_table [summary_table .model .isin ([
'Seed 4000','Seed 2026','Seed 2037','Seed 2048','Ensemble 2026+2037+2048'
])].copy ()
colors =['#6B7280','#60A5FA','#34D399','#F59E0B','#D81B60']
fig ,axes =plt .subplots (1 ,2 ,figsize =(13 ,5 ))
axes [0 ].bar (plot_table .model ,plot_table .rmse_72h_km ,color =colors )
axes [0 ].set_ylabel ('72 h RMSE (km)')
axes [0 ].set_title ('\u5355seed\u4e0e\u4e09\u6a21\u578bensemble')
axes [0 ].tick_params (axis ='x',rotation =25 )
for i ,v in enumerate (plot_table .rmse_72h_km ):axes [0 ].text (i ,v +.08 ,f'{v :.2f}',ha ='center')

axes [1 ].bar (plot_table .model ,plot_table .median_72h_km ,color =colors )
axes [1 ].set_ylabel ('72 h median error (km)')
axes [1 ].set_title ('Median error\u5bf9\u6bd4')
axes [1 ].tick_params (axis ='x',rotation =25 )
for i ,v in enumerate (plot_table .median_72h_km ):axes [1 ].text (i ,v +.05 ,f'{v :.2f}',ha ='center')
fig .tight_layout ()
fig .savefig (AUDIT_DIR /'fnn_seed_ensemble_comparison.png',dpi =200 ,bbox_inches ='tight')
plt .show ()

original_cases =case_table [case_table .model .eq ('Seed 4000')][['forecast_case_id','error_72h_km']].rename (columns ={'error_72h_km':'original'})
ensemble_cases =case_table [case_table .model .eq ('Ensemble 2026+2037+2048')][['forecast_case_id','error_72h_km']].rename (columns ={'error_72h_km':'ensemble'})
paired =original_cases .merge (ensemble_cases ,on ='forecast_case_id')
paired ['change_km']=paired .ensemble -paired .original 

fig ,ax =plt .subplots (figsize =(8 ,5 ))
ax .hist (paired .change_km ,bins =40 ,color ='#159FD3',alpha =.85 )
ax .axvline (0 ,color ='black',ls ='--')
ax .axvline (paired .change_km .mean (),color ='#D81B60',lw =2 ,
label =f"mean change = {paired .change_km .mean ():.2f} km")
ax .set_xlabel ('Optimised ensemble error − Original seed 4000 error (km)')
ax .set_ylabel ('Test cases')
ax .set_title ('\u9010case\u8bef\u5dee\u53d8\u5316：\u8d1f\u503c\u8868\u793aensemble\u66f4\u597d')
ax .legend (frameon =False )
fig .tight_layout ()
fig .savefig (AUDIT_DIR /'fnn_seed_ensemble_case_delta.png',dpi =200 ,bbox_inches ='tight')
plt .show ()

print ('ensemble\u53d8\u597d\u7684cases：',int ((paired .change_km <0 ).sum ()),'/',len (paired ),
f"({100 *(paired .change_km <0 ).mean ():.1f}%)")
print ('ensemble\u53d8\u5dee\u7684cases：',int ((paired .change_km >0 ).sum ()),'/',len (paired ),
f"({100 *(paired .change_km >0 ).mean ():.1f}%)")


# ## Analysis stage 8
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


report ={
'configuration':{**BEST_CONFIG ,'hidden_sizes':list (BEST_CONFIG ['hidden_sizes']),'epochs':SELECTED_EPOCHS },
'seeds':SEEDS ,
'summary':summary_table .to_dict (orient ='records'),
'seed_effect_km':float (seed_effect ),
'ensemble_effect_km':float (ensemble_effect ),
'ensemble_vs_best_new_single_km':float (best_gap ),
'verdict':verdict ,
}
with open (AUDIT_DIR /'fnn_seed_ensemble_audit.json','w')as f :
    json .dump (report ,f ,indent =2 ,ensure_ascii =False )

display (Markdown (f'''## \u6700\u7ec8\u7ed3\u8bba

- Original\u4e0eOptimised\u4f7f\u7528\u76f8\u540c\u7684128–64\u7ed3\u6784、L2、\u5b66\u4e60\u7387、batch size\u548c129\u4e2aepochs。
- seed 4000\u4e0e\u4e09\u4e2a\u65b0seed\u7684\u533a\u522b\u53ea\u6765\u81ea\u521d\u59cb\u5316\u6743\u91cd\u548cmini-batch\u987a\u5e8f。
- \u4e09\u4e2a\u65b0seed\u7684\u5e73\u5747\u5355\u6a21\u578bRMSE\u4e3a **{new_single .rmse_72h_km .mean ():.3f} km**。
- \u4e09\u6a21\u578bensemble RMSE\u4e3a **{ensemble_row .rmse_72h_km :.3f} km**。
- ensemble\u76f8\u5bf9\u4e09\u4e2a\u65b0\u5355\u6a21\u578b\u7684\u5e73\u5747RMSE\u6539\u5584 **{ensemble_effect :.3f} km**。
- \u76f8\u5bf9Original seed 4000\u6539\u5584 **{original_row .rmse_72h_km -ensemble_row .rmse_72h_km :.3f} km**。

**\u8bca\u65ad：{verdict }**

\u6240\u6709\u56db\u5957\u6743\u91cd、\u9884\u6d4b、\u9010case\u6307\u6807、\u6c47\u603b\u8868\u548c\u56fe\u7247\u5747\u5df2\u4fdd\u5b58\u5230：

`{AUDIT_DIR }`
'''))

