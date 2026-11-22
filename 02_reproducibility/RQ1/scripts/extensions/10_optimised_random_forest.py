#!/usr/bin/env python
# coding: utf-8

# # 10 Optimised Random Forest
# 
# Documents the later Random Forest tuning search and compares it with the retained original Random Forest.
# 
# **Provenance.** This is an English-facing execution copy of the original project notebook. The computational cells are preserved so that the analysis remains reproducible. Chinese narrative cells have been replaced by concise English stage notes; the original notebook remains unchanged in the source project.

# ## Analysis stage 1
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


from pathlib import Path 
import json 
import os 
import time 

import matplotlib .pyplot as plt 
import numpy as np 
import pandas as pd 
from IPython .display import display 
from sklearn .ensemble import RandomForestRegressor 
from sklearn .model_selection import GroupKFold ,ParameterSampler 

PROJECT_ROOT =next (
(p for p in [Path .cwd (),*Path .cwd ().parents ]if (p /"data"/"processed").exists ()),
None ,
)
if PROJECT_ROOT is None :
    raise FileNotFoundError ("\u627e\u4e0d\u5230data/processed；\u8bf7\u4ece\u9879\u76ee\u76ee\u5f55\u6216\u5176\u5b50\u76ee\u5f55\u8fd0\u884c。")

PROCESSED =PROJECT_ROOT /"data"/"processed"
OUTPUT_DIR =PROJECT_ROOT /"output"/"rq1"/"zh"
OUTPUT_DIR .mkdir (parents =True ,exist_ok =True )

EARTH_RADIUS_KM =6371.0 
HISTORY_HOURS =24 
HORIZON_HOURS =72 
RANDOM_STATE =2026 
FAST_MODE =os .getenv ("RQ1_FAST_MODE","0")=="1"

plt .rcParams ["font.sans-serif"]=["PingFang SC","Arial Unicode MS","DejaVu Sans"]
plt .rcParams ["axes.unicode_minus"]=False 

print ("\u9879\u76ee\u76ee\u5f55：",PROJECT_ROOT )
print ("\u5feb\u901f\u68c0\u67e5\u6a21\u5f0f：",FAST_MODE )


# ## Analysis stage 2
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def latlon_to_local_km (origin_lat ,origin_lon ,latitude ,longitude ):
    lat0 =np .deg2rad (np .asarray (origin_lat ,dtype =float ))
    lon0 =np .deg2rad (np .asarray (origin_lon ,dtype =float ))
    lat1 =np .deg2rad (np .asarray (latitude ,dtype =float ))
    lon1 =np .deg2rad (np .asarray (longitude ,dtype =float ))
    dlon =(lon1 -lon0 +np .pi )%(2 *np .pi )-np .pi 
    dlat =lat1 -lat0 
    a =np .sin (dlat /2 )**2 +np .cos (lat0 )*np .cos (lat1 )*np .sin (dlon /2 )**2 
    central =2 *np .arcsin (np .sqrt (np .clip (a ,0 ,1 )))
    bearing =np .arctan2 (
    np .sin (dlon )*np .cos (lat1 ),
    np .cos (lat0 )*np .sin (lat1 )-np .sin (lat0 )*np .cos (lat1 )*np .cos (dlon ),
    )
    distance =EARTH_RADIUS_KM *central 
    return distance *np .sin (bearing ),distance *np .cos (bearing )


def local_km_to_latlon (origin_lat ,origin_lon ,east_km ,north_km ):
    lat0 =np .deg2rad (np .asarray (origin_lat ,dtype =float ))
    lon0 =np .deg2rad (np .asarray (origin_lon ,dtype =float ))
    east =np .asarray (east_km ,dtype =float )
    north =np .asarray (north_km ,dtype =float )
    distance =np .hypot (east ,north )
    bearing =np .arctan2 (east ,north )
    angular =distance /EARTH_RADIUS_KM 
    latitude =np .arcsin (
    np .sin (lat0 )*np .cos (angular )
    +np .cos (lat0 )*np .sin (angular )*np .cos (bearing )
    )
    longitude =lon0 +np .arctan2 (
    np .sin (bearing )*np .sin (angular )*np .cos (lat0 ),
    np .cos (angular )-np .sin (lat0 )*np .sin (latitude ),
    )
    return np .rad2deg (latitude ),(np .rad2deg (longitude )+180 )%360 -180 


def haversine_km (lat1 ,lon1 ,lat2 ,lon2 ):
    lat1 ,lon1 ,lat2 ,lon2 =map (
    lambda x :np .deg2rad (np .asarray (x ,dtype =float )),
    [lat1 ,lon1 ,lat2 ,lon2 ],
    )
    dlat ,dlon =lat2 -lat1 ,lon2 -lon1 
    a =np .sin (dlat /2 )**2 +np .cos (lat1 )*np .cos (lat2 )*np .sin (dlon /2 )**2 
    return 2 *EARTH_RADIUS_KM *np .arcsin (np .sqrt (np .clip (a ,0 ,1 )))


origins =(
pd .read_parquet (PROCESSED /"forecast_origins.parquet")
.sort_values ("forecast_case_id")
.reset_index (drop =True )
)
observations =pd .read_parquet (
PROCESSED /"region_A_observations.parquet",
columns =["ID","segment_id","segment_step","latitude","longitude","ve","vn"],
)
n_cases =len (origins )

history =origins [[
"forecast_case_id","ID","segment_id","history_start_step",
"origin_latitude","origin_longitude",
]].loc [origins .index .repeat (HISTORY_HOURS )].reset_index (drop =True )
history ["history_index"]=np .tile (np .arange (HISTORY_HOURS ),n_cases )
history ["segment_step"]=history ["history_start_step"]+history ["history_index"]
history =history .merge (
observations ,
on =["ID","segment_id","segment_step"],
how ="left",
validate ="many_to_one",
).sort_values (["forecast_case_id","history_index"]).reset_index (drop =True )
history ["east_km"],history ["north_km"]=latlon_to_local_km (
history ["origin_latitude"],history ["origin_longitude"],
history ["latitude"],history ["longitude"],
)
history_values =history [["east_km","north_km","ve","vn"]].to_numpy ().reshape (
n_cases ,HISTORY_HOURS ,4 
)
X =np .column_stack ([
origins ["origin_latitude"].to_numpy (),
origins ["origin_longitude"].to_numpy (),
history_values .reshape (n_cases ,-1 ),
])

targets =origins [[
"forecast_case_id","ID","segment_id","target_start_step",
"origin_latitude","origin_longitude",
]].loc [origins .index .repeat (HORIZON_HOURS )].reset_index (drop =True )
targets ["lead_time_hours"]=np .tile (np .arange (1 ,HORIZON_HOURS +1 ),n_cases )
targets ["segment_step"]=targets ["target_start_step"]+targets ["lead_time_hours"]-1 
targets =targets .merge (
observations [["ID","segment_id","segment_step","latitude","longitude"]],
on =["ID","segment_id","segment_step"],
how ="left",
validate ="many_to_one",
).sort_values (["forecast_case_id","lead_time_hours"]).reset_index (drop =True )
targets ["east_km"],targets ["north_km"]=latlon_to_local_km (
targets ["origin_latitude"],targets ["origin_longitude"],
targets ["latitude"],targets ["longitude"],
)
Y =targets [["east_km","north_km"]].to_numpy ().reshape (n_cases ,144 )

train_idx =np .flatnonzero (origins ["partition"].eq ("training").to_numpy ())
calibration_idx =np .flatnonzero (origins ["partition"].eq ("calibration").to_numpy ())
test_idx =np .flatnonzero (origins ["partition"].eq ("test").to_numpy ())
groups =origins ["ID"].to_numpy ()

checks =pd .Series ({
"cases=5117":n_cases ==5117 ,
"X=5117×98":X .shape ==(5117 ,98 ),
"Y=5117×144":Y .shape ==(5117 ,144 ),
"training=3591":len (train_idx )==3591 ,
"calibration=769":len (calibration_idx )==769 ,
"test=757":len (test_idx )==757 ,
"\u65e0\u7f3a\u5931\u6216\u65e0\u9650\u503c":np .isfinite (X ).all ()and np .isfinite (Y ).all (),
},name ="\u901a\u8fc7")
display (checks .to_frame ())
assert checks .all ()


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


group_cv = GroupKFold(n_splits=5)
fixed_folds = [
    (train_idx[fit_local], train_idx[valid_local])
    for fit_local, valid_local in group_cv.split(
        X[train_idx], Y[train_idx], groups=groups[train_idx]
    )
]

fold_table = pd.DataFrame({
    "fold": np.arange(1, 6),
    "fit_cases": [len(a) for a, _ in fixed_folds],
    "validation_cases": [len(b) for _, b in fixed_folds],
    "validation_drifters": [origins.iloc[b]["ID"].nunique() for _, b in fixed_folds],
})
display(fold_table)

for fit_idx, valid_idx in fixed_folds:
    assert set(origins.iloc[fit_idx]["ID"]).isdisjoint(set(origins.iloc[valid_idx]["ID"]))


def endpoint_errors(indices, predictions):
    predictions = np.asarray(predictions)
    pred_east = predictions[:, -2]
    pred_north = predictions[:, -1]
    case_origins = origins.iloc[indices]
    pred_lat, pred_lon = local_km_to_latlon(
        case_origins["origin_latitude"].to_numpy(),
        case_origins["origin_longitude"].to_numpy(),
        pred_east,
        pred_north,
    )
    true_last = Y[indices][:, -2:]
    true_lat, true_lon = local_km_to_latlon(
        case_origins["origin_latitude"].to_numpy(),
        case_origins["origin_longitude"].to_numpy(),
        true_last[:, 0],
        true_last[:, 1],
    )
    return haversine_km(pred_lat, pred_lon, true_lat, true_lon)


def evaluate_configuration(params, n_estimators, random_state):
    rows = []
    for fold, (fit_idx, valid_idx) in enumerate(fixed_folds, start=1):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            bootstrap=True,
            random_state=random_state + fold,
            n_jobs=-1,
            **params,
        )
        model.fit(X[fit_idx], Y[fit_idx])
        pred = model.predict(X[valid_idx])
        errors = endpoint_errors(valid_idx, pred)
        rows.append({
            "fold": fold,
            "validation_cases": len(valid_idx),
            "rmse_72h_km": np.sqrt(np.mean(errors ** 2)),
            "median_72h_km": np.median(errors),
            "trajectory_displacement_rmse_km": np.sqrt(np.mean((pred - Y[valid_idx]) ** 2)),
        })
    return pd.DataFrame(rows)


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


parameter_space ={
"max_depth":[None ,10 ,14 ,18 ,24 ,32 ,40 ],
"min_samples_leaf":[1 ,2 ,3 ,4 ,6 ,8 ,12 ],
"min_samples_split":[2 ,4 ,8 ,12 ,20 ],
"max_features":[0.25 ,0.4 ,0.55 ,0.7 ,0.85 ,1.0 ,"sqrt"],
"max_samples":[0.6 ,0.75 ,0.9 ,None ],
}
n_candidates =3 if FAST_MODE else 20 
stage1_trees =25 if FAST_MODE else 80 
candidates =list (ParameterSampler (
parameter_space ,
n_iter =n_candidates ,
random_state =RANDOM_STATE ,
))

stage1_rows =[]
start =time .time ()
for candidate_id ,params in enumerate (candidates ,start =1 ):
    fold_result =evaluate_configuration (params ,stage1_trees ,RANDOM_STATE )
    fold_result ["candidate"]=candidate_id 
    for key ,value in params .items ():
        fold_result [key ]=value 
    stage1_rows .append (fold_result )
    print (
    f"{candidate_id :02d}/{len (candidates )} "
    f"RMSE72={fold_result ['rmse_72h_km'].mean ():.3f} km | {params }"
    )

stage1_folds =pd .concat (stage1_rows ,ignore_index =True )
stage1_summary =(
stage1_folds .groupby ("candidate",as_index =False )
.agg (
mean_cv_rmse_72h_km =("rmse_72h_km","mean"),
std_cv_rmse_72h_km =("rmse_72h_km","std"),
mean_cv_median_72h_km =("median_72h_km","mean"),
mean_trajectory_rmse_km =("trajectory_displacement_rmse_km","mean"),
)
.merge (
stage1_folds .drop_duplicates ("candidate")[[
"candidate","max_depth","min_samples_leaf","min_samples_split",
"max_features","max_samples",
]],
on ="candidate",
)
.sort_values ("mean_cv_rmse_72h_km")
.reset_index (drop =True )
)
display (stage1_summary .head (10 ).round (3 ))
print (f"\u7b2c\u4e00\u9636\u6bb5\u8017\u65f6：{time .time ()-start :.1f}\u79d2")


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


top_n =1 if FAST_MODE else 2 
tree_counts =[30 ,60 ]if FAST_MODE else [150 ,350 ,600 ]
top_candidate_ids =stage1_summary .head (top_n )["candidate"].astype (int ).tolist ()

stage2_rows =[]
start =time .time ()
for candidate_id in top_candidate_ids :
    params =candidates [candidate_id -1 ]
    for n_trees in tree_counts :
        fold_result =evaluate_configuration (params ,n_trees ,RANDOM_STATE +100 )
        fold_result ["candidate"]=candidate_id 
        fold_result ["n_estimators"]=n_trees 
        for key ,value in params .items ():
            fold_result [key ]=value 
        stage2_rows .append (fold_result )
        print (
        f"candidate={candidate_id }, trees={n_trees }, "
        f"RMSE72={fold_result ['rmse_72h_km'].mean ():.3f} km"
        )

stage2_folds =pd .concat (stage2_rows ,ignore_index =True )
stage2_summary =(
stage2_folds .groupby (["candidate","n_estimators"],as_index =False )
.agg (
mean_cv_rmse_72h_km =("rmse_72h_km","mean"),
std_cv_rmse_72h_km =("rmse_72h_km","std"),
mean_cv_median_72h_km =("median_72h_km","mean"),
mean_trajectory_rmse_km =("trajectory_displacement_rmse_km","mean"),
)
.sort_values (["mean_cv_rmse_72h_km","mean_trajectory_rmse_km"])
.reset_index (drop =True )
)
display (stage2_summary .round (3 ))
best_stage2 =stage2_summary .iloc [0 ]
best_candidate =int (best_stage2 ["candidate"])
best_n_estimators =int (best_stage2 ["n_estimators"])
best_params =candidates [best_candidate -1 ]
print ("\u6700\u7ec8\u7ed3\u6784：",best_params )
print ("\u6811\u6570\u91cf：",best_n_estimators )
print (f"\u7b2c\u4e8c\u9636\u6bb5\u8017\u65f6：{time .time ()-start :.1f}\u79d2")


# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


final_seeds =[2026 ]if FAST_MODE else [2026 ,2037 ,2048 ]
test_predictions =[]
start =time .time ()

for seed in final_seeds :
    model =RandomForestRegressor (
    n_estimators =best_n_estimators ,
    bootstrap =True ,
    random_state =seed ,
    n_jobs =-1 ,
    **best_params ,
    )
    model .fit (X [train_idx ],Y [train_idx ])
    test_predictions .append (model .predict (X [test_idx ]))
    print ("\u5b8c\u6210seed：",seed )

y_pred_test =np .mean (test_predictions ,axis =0 )
print ("\u96c6\u6210\u9884\u6d4b\u5f62\u72b6：",y_pred_test .shape )
print (f"\u6700\u7ec8\u62df\u5408\u8017\u65f6：{time .time ()-start :.1f}\u79d2")


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def evaluate_test (y_pred ):
    y_pred =np .asarray (y_pred ).reshape (len (test_idx ),HORIZON_HOURS ,2 )
    test_origins =origins .iloc [test_idx ].reset_index (drop =True )
    origin_lat =np .repeat (test_origins ["origin_latitude"].to_numpy (),HORIZON_HOURS )
    origin_lon =np .repeat (test_origins ["origin_longitude"].to_numpy (),HORIZON_HOURS )
    pred_lat ,pred_lon =local_km_to_latlon (
    origin_lat ,origin_lon ,
    y_pred [:,:,0 ].reshape (-1 ),
    y_pred [:,:,1 ].reshape (-1 ),
    )
    test_targets =(
    targets [targets ["forecast_case_id"].isin (test_origins ["forecast_case_id"])]
    .sort_values (["forecast_case_id","lead_time_hours"])
    .reset_index (drop =True )
    )
    evaluation =pd .DataFrame ({
    "forecast_case_id":np .repeat (test_origins ["forecast_case_id"].to_numpy (),HORIZON_HOURS ),
    "ID":np .repeat (test_origins ["ID"].to_numpy (),HORIZON_HOURS ),
    "lead_time_hours":np .tile (np .arange (1 ,HORIZON_HOURS +1 ),len (test_origins )),
    "pred_latitude":pred_lat ,
    "pred_longitude":pred_lon ,
    "true_latitude":test_targets ["latitude"].to_numpy (),
    "true_longitude":test_targets ["longitude"].to_numpy (),
    })
    evaluation ["separation_error_km"]=haversine_km (
    evaluation ["pred_latitude"],evaluation ["pred_longitude"],
    evaluation ["true_latitude"],evaluation ["true_longitude"],
    )
    lead_metrics =(
    evaluation .groupby ("lead_time_hours")["separation_error_km"]
    .agg (
    n ="size",
    median_error_km ="median",
    q25_error_km =lambda x :x .quantile (0.25 ),
    q75_error_km =lambda x :x .quantile (0.75 ),
    rmse_km =lambda x :np .sqrt (np .mean (np .square (x ))),
    )
    .reset_index ()
    )
    case_rows =[]
    for case_id ,frame in evaluation .groupby ("forecast_case_id",sort =False ):
        frame =frame .sort_values ("lead_time_hours")
        errors =frame ["separation_error_km"].to_numpy ()
        case_rows .append ({
        "forecast_case_id":case_id ,
        "ID":frame ["ID"].iloc [0 ],
        "final_72h_error_km":errors [-1 ],
        "integrated_trajectory_error_km_h":np .trapezoid (
        np .r_ [0.0 ,errors ],
        x =np .r_ [0 ,frame ["lead_time_hours"].to_numpy ()],
        ),
        })
    return evaluation ,lead_metrics ,pd .DataFrame (case_rows )

evaluation ,lead_metrics ,case_metrics =evaluate_test (y_pred_test )

def metric (hour ,column ):
    return float (lead_metrics .loc [lead_metrics ["lead_time_hours"].eq (hour ),column ].iloc [0 ])

summary =pd .DataFrame ({
"\u6307\u6807":[
"Training cases","Calibration cases（\u672a\u4f7f\u7528）","Test cases",
"24h median error (km)","48h median error (km)",
"72h median error (km)","72h RMSE (km)","Median ITE (km·h)",
],
"\u7ed3\u679c":[
len (train_idx ),len (calibration_idx ),len (test_idx ),
metric (24 ,"median_error_km"),metric (48 ,"median_error_km"),
metric (72 ,"median_error_km"),metric (72 ,"rmse_km"),
case_metrics ["integrated_trajectory_error_km_h"].median (),
],
})
display (summary .round (3 ))


# ## Analysis stage 8
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


comparison_sources ={
"Velocity persistence":pd .read_csv (OUTPUT_DIR /"velocity_persistence_lead_metrics.csv"),
"Spatial mean flow":pd .read_csv (OUTPUT_DIR /"spatial_mean_flow_lead_metrics.csv"),
"FNN":pd .read_csv (OUTPUT_DIR /"fnn_lead_metrics.csv"),
"Optimised Random Forest":lead_metrics ,
}

fig ,ax =plt .subplots (figsize =(10 ,5 ))
styles ={
"Velocity persistence":("#666666","--"),
"Spatial mean flow":("#999999",":"),
"FNN":("#d81b60","-"),
"Optimised Random Forest":("#0072b2","-"),
}
for name ,frame in comparison_sources .items ():
    color ,linestyle =styles [name ]
    ax .plot (
    frame ["lead_time_hours"],frame ["median_error_km"],
    color =color ,linestyle =linestyle ,
    linewidth =2.7 if name in ["FNN","Optimised Random Forest"]else 2 ,
    label =name ,
    )
ax .set (
xlabel ="Lead time（\u5c0f\u65f6）",
ylabel ="Median separation error（km）",
title ="\u4f18\u5316Random Forest\u4e0eFNN\u53ca\u7269\u7406\u57fa\u7ebf",
)
ax .set_xlim (1 ,72 )
ax .grid (alpha =0.25 )
ax .legend (frameon =False )
plt .tight_layout ()
fig .savefig (
OUTPUT_DIR /"optimised_random_forest_vs_fnn_and_baselines.png",
dpi =200 ,
bbox_inches ="tight",
)
plt .show ()

rows =[]
for name ,frame in comparison_sources .items ():
    row =frame .loc [frame ["lead_time_hours"].eq (72 )].iloc [0 ]
    rows .append ({
    "model":name ,
    "median_72h_km":row ["median_error_km"],
    "rmse_72h_km":row ["rmse_km"],
    })
comparison_72h =pd .DataFrame (rows ).sort_values ("rmse_72h_km")
display (comparison_72h .round (3 ))


# ## Analysis stage 9
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


fnn_cases = pd.read_csv(OUTPUT_DIR / "fnn_test_case_metrics.csv")[
    ["forecast_case_id", "final_72h_error_km"]
].rename(columns={"final_72h_error_km": "fnn_error_km"})

paired = case_metrics.merge(fnn_cases, on="forecast_case_id", validate="one_to_one")
paired = paired.rename(columns={"final_72h_error_km": "rf_error_km"})
unique_drifters = paired["ID"].unique()
rng = np.random.default_rng(RANDOM_STATE)
n_bootstrap = 200 if FAST_MODE else 5000
bootstrap_rows = []

for _ in range(n_bootstrap):
    sampled_ids = rng.choice(unique_drifters, size=len(unique_drifters), replace=True)
    sampled_parts = [paired[paired["ID"].eq(d)] for d in sampled_ids]
    sample = pd.concat(sampled_parts, ignore_index=True)
    rf_rmse = np.sqrt(np.mean(sample["rf_error_km"] ** 2))
    fnn_rmse = np.sqrt(np.mean(sample["fnn_error_km"] ** 2))
    bootstrap_rows.append({
        "rmse_difference_rf_minus_fnn_km": rf_rmse - fnn_rmse,
        "median_difference_rf_minus_fnn_km":
            sample["rf_error_km"].median() - sample["fnn_error_km"].median(),
    })

bootstrap = pd.DataFrame(bootstrap_rows)
point_rmse_difference = (
    np.sqrt(np.mean(paired["rf_error_km"] ** 2))
    - np.sqrt(np.mean(paired["fnn_error_km"] ** 2))
)
point_median_difference = (
    paired["rf_error_km"].median() - paired["fnn_error_km"].median()
)
bootstrap_summary = pd.DataFrame({
    "comparison": ["RMSE difference: RF − FNN", "Median difference: RF − FNN"],
    "point_difference_km": [point_rmse_difference, point_median_difference],
    "ci_2.5_km": [
        bootstrap["rmse_difference_rf_minus_fnn_km"].quantile(0.025),
        bootstrap["median_difference_rf_minus_fnn_km"].quantile(0.025),
    ],
    "ci_97.5_km": [
        bootstrap["rmse_difference_rf_minus_fnn_km"].quantile(0.975),
        bootstrap["median_difference_rf_minus_fnn_km"].quantile(0.975),
    ],
})
display(bootstrap_summary.round(3))


# ## Analysis stage 10
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


stage1_folds .to_csv (OUTPUT_DIR /"optimised_random_forest_stage1_folds.csv",index =False )
stage1_summary .to_csv (OUTPUT_DIR /"optimised_random_forest_stage1_summary.csv",index =False )
stage2_folds .to_csv (OUTPUT_DIR /"optimised_random_forest_stage2_folds.csv",index =False )
stage2_summary .to_csv (OUTPUT_DIR /"optimised_random_forest_stage2_summary.csv",index =False )
lead_metrics .to_csv (OUTPUT_DIR /"optimised_random_forest_lead_metrics.csv",index =False )
case_metrics .to_csv (OUTPUT_DIR /"optimised_random_forest_test_case_metrics.csv",index =False )
comparison_72h .to_csv (OUTPUT_DIR /"optimised_random_forest_72h_comparison.csv",index =False )
bootstrap_summary .to_csv (OUTPUT_DIR /"optimised_random_forest_vs_fnn_bootstrap.csv",index =False )

final_config ={
"selection_metric":"grouped 5-fold CV 72h Haversine RMSE",
"n_estimators":best_n_estimators ,
"seeds":final_seeds ,
**best_params ,
}
with open (
OUTPUT_DIR /"optimised_random_forest_best_parameters.json",
"w",
encoding ="utf-8",
)as f :
    json .dump (final_config ,f ,ensure_ascii =False ,indent =2 )

print ("Notebook\u5b8c\u6210。\u7ed3\u679c\u76ee\u5f55：",OUTPUT_DIR )
print (json .dumps (final_config ,ensure_ascii =False ,indent =2 ))


# ## Analysis stage 11
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.
