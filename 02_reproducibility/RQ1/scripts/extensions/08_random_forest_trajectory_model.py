#!/usr/bin/env python
# coding: utf-8

# # 08 Random Forest Trajectory Model
# 
# Fits the retained Random Forest trajectory model used in the frozen RQ1 comparison.
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
import warnings 

import matplotlib .pyplot as plt 
import numpy as np 
import pandas as pd 
from IPython .display import display 
from sklearn .model_selection import GroupKFold 

PROJECT_ROOT =next (
(p for p in [Path .cwd (),*Path .cwd ().parents ]if (p /"data"/"processed").exists ()),
None ,
)
if PROJECT_ROOT is None :
    raise FileNotFoundError ("\u627e\u4e0d\u5230 data/processed；\u8bf7\u4ece\u9879\u76ee\u76ee\u5f55\u6216\u5176\u5b50\u76ee\u5f55\u8fd0\u884c。")

PROCESSED =PROJECT_ROOT /"data"/"processed"
OUTPUT_DIR =PROJECT_ROOT /"output"/"rq1"/"zh"
OUTPUT_DIR .mkdir (parents =True ,exist_ok =True )

EARTH_RADIUS_KM =6371.0 
HISTORY_HOURS =24 
HORIZON_HOURS =72 
RANDOM_STATE =2026 
N_FOLDS =5 
FAST_MODE =os .getenv ("RQ1_FAST_MODE","0")=="1"

plt .rcParams ["font.sans-serif"]=["PingFang SC","Arial Unicode MS","DejaVu Sans"]
plt .rcParams ["axes.unicode_minus"]=False 

print ("\u9879\u76ee\u76ee\u5f55：",PROJECT_ROOT )
print ("\u8f93\u51fa\u76ee\u5f55：",OUTPUT_DIR )
print ("\u5feb\u901f\u68c0\u67e5\u6a21\u5f0f：",FAST_MODE )

from sklearn .ensemble import RandomForestRegressor 


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
columns =["ID","segment_id","segment_step","time","latitude","longitude","ve","vn"],
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
history ["relative_east_km"],history ["relative_north_km"]=latlon_to_local_km (
history ["origin_latitude"],history ["origin_longitude"],
history ["latitude"],history ["longitude"],
)
history_values =history [[
"relative_east_km","relative_north_km","ve","vn"
]].to_numpy ().reshape (n_cases ,HISTORY_HOURS ,4 )
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
targets ["target_east_km"],targets ["target_north_km"]=latlon_to_local_km (
targets ["origin_latitude"],targets ["origin_longitude"],
targets ["latitude"],targets ["longitude"],
)
Y =targets [["target_east_km","target_north_km"]].to_numpy ().reshape (n_cases ,-1 )

train_idx =np .flatnonzero (origins ["partition"].eq ("training").to_numpy ())
calibration_idx =np .flatnonzero (origins ["partition"].eq ("calibration").to_numpy ())
test_idx =np .flatnonzero (origins ["partition"].eq ("test").to_numpy ())
groups =origins ["ID"].to_numpy ()

checks =pd .Series ({
"cases = 5117":n_cases ==5117 ,
"X shape = 5117×98":X .shape ==(5117 ,98 ),
"Y shape = 5117×144":Y .shape ==(5117 ,144 ),
"training cases = 3591":len (train_idx )==3591 ,
"calibration cases = 769":len (calibration_idx )==769 ,
"test cases = 757":len (test_idx )==757 ,
"X/Y\u6ca1\u6709\u7f3a\u5931\u503c":np .isfinite (X ).all ()and np .isfinite (Y ).all (),
},name ="\u901a\u8fc7")
display (checks .to_frame ())
assert checks .all ()


# ## Analysis stage 3
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


parameter_grid =[
{"max_depth":None ,"min_samples_leaf":1 ,"max_features":1.0 },
{"max_depth":None ,"min_samples_leaf":2 ,"max_features":0.7 },
{"max_depth":None ,"min_samples_leaf":4 ,"max_features":0.7 },
{"max_depth":24 ,"min_samples_leaf":1 ,"max_features":1.0 },
{"max_depth":24 ,"min_samples_leaf":2 ,"max_features":0.7 },
{"max_depth":16 ,"min_samples_leaf":4 ,"max_features":0.5 },
]
if FAST_MODE :
    parameter_grid =parameter_grid [:2 ]

cv_trees =25 if FAST_MODE else 100 
cv =GroupKFold (n_splits =N_FOLDS )
cv_rows =[]
start =time .time ()

for candidate_id ,params in enumerate (parameter_grid ,start =1 ):
    fold_scores =[]
    for fold ,(fit_local ,valid_local )in enumerate (
    cv .split (X [train_idx ],Y [train_idx ],groups =groups [train_idx ]),start =1 
    ):
        fit_idx =train_idx [fit_local ]
        valid_idx =train_idx [valid_local ]
        model =RandomForestRegressor (
        n_estimators =cv_trees ,
        bootstrap =True ,
        random_state =RANDOM_STATE +fold ,
        n_jobs =-1 ,
        **params ,
        )
        model .fit (X [fit_idx ],Y [fit_idx ])
        pred =model .predict (X [valid_idx ])
        rmse =float (np .sqrt (np .mean ((pred -Y [valid_idx ])**2 )))
        fold_scores .append (rmse )
        cv_rows .append ({
        "candidate":candidate_id ,"fold":fold ,"validation_cases":len (valid_idx ),
        "displacement_rmse_km":rmse ,**params ,
        })
    print (f"\u5019\u9009{candidate_id }: mean RMSE={np .mean (fold_scores ):.3f} km, {params }")

cv_folds =pd .DataFrame (cv_rows )
cv_summary =(
cv_folds .groupby ("candidate",as_index =False )
.agg (mean_cv_rmse_km =("displacement_rmse_km","mean"),
std_cv_rmse_km =("displacement_rmse_km","std"))
)
cv_summary =cv_summary .merge (
cv_folds .drop_duplicates ("candidate")[["candidate","max_depth","min_samples_leaf","max_features"]],
on ="candidate",
)
best_candidate =int (cv_summary .loc [cv_summary ["mean_cv_rmse_km"].idxmin (),"candidate"])
best_params =parameter_grid [best_candidate -1 ]
display (cv_summary .sort_values ("mean_cv_rmse_km").round (3 ))
print ("\u9009\u62e9：",best_params )
print (f"CV\u8017\u65f6：{time .time ()-start :.1f}\u79d2")


# ## Analysis stage 4
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


final_trees = 60 if FAST_MODE else 400
random_forest = RandomForestRegressor(
    n_estimators=final_trees,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    **best_params,
)
random_forest.fit(X[train_idx], Y[train_idx])
y_pred_test = random_forest.predict(X[test_idx])
print("Test prediction shape:", y_pred_test.shape)


# ## Analysis stage 5
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


def evaluate_test_predictions (model_name ,y_pred_test ):
    y_pred_test =np .asarray (y_pred_test ).reshape (len (test_idx ),HORIZON_HOURS ,2 )
    test_origins =origins .iloc [test_idx ].reset_index (drop =True )
    origin_lat =np .repeat (test_origins ["origin_latitude"].to_numpy (),HORIZON_HOURS )
    origin_lon =np .repeat (test_origins ["origin_longitude"].to_numpy (),HORIZON_HOURS )
    pred_lat ,pred_lon =local_km_to_latlon (
    origin_lat ,origin_lon ,
    y_pred_test [:,:,0 ].reshape (-1 ),
    y_pred_test [:,:,1 ].reshape (-1 ),
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
        hours =frame ["lead_time_hours"].to_numpy ()
        case_rows .append ({
        "forecast_case_id":case_id ,
        "ID":frame ["ID"].iloc [0 ],
        "final_72h_error_km":errors [-1 ],
        "integrated_trajectory_error_km_h":np .trapezoid (
        np .r_ [0.0 ,errors ],x =np .r_ [0 ,hours ]
        ),
        })
    case_metrics =pd .DataFrame (case_rows )

    def at (hour ,column ):
        return float (lead_metrics .loc [lead_metrics ["lead_time_hours"].eq (hour ),column ].iloc [0 ])

    summary =pd .DataFrame ({
    "\u6307\u6807":[
    "\u6a21\u578b","Training cases","Calibration cases（\u672a\u4f7f\u7528）","Test cases",
    "24h median error (km)","48h median error (km)",
    "72h median error (km)","72h RMSE (km)","Median ITE (km·h)",
    ],
    "\u7ed3\u679c":[
    model_name ,len (train_idx ),len (calibration_idx ),len (test_idx ),
    at (24 ,"median_error_km"),at (48 ,"median_error_km"),
    at (72 ,"median_error_km"),at (72 ,"rmse_km"),
    case_metrics ["integrated_trajectory_error_km_h"].median (),
    ],
    })
    return evaluation ,lead_metrics ,case_metrics ,summary 


def compare_with_baselines (model_label ,lead_metrics ,color ,output_stem ):
    persistence =pd .read_csv (OUTPUT_DIR /"velocity_persistence_lead_metrics.csv")
    mean_flow =pd .read_csv (OUTPUT_DIR /"spatial_mean_flow_lead_metrics.csv")
    comparison =(
    lead_metrics [["lead_time_hours","median_error_km","rmse_km"]]
    .rename (columns ={"median_error_km":"model_median_km","rmse_km":"model_rmse_km"})
    .merge (
    persistence [["lead_time_hours","median_error_km","rmse_km"]]
    .rename (columns ={"median_error_km":"persistence_median_km","rmse_km":"persistence_rmse_km"}),
    on ="lead_time_hours",
    )
    .merge (
    mean_flow [["lead_time_hours","median_error_km","rmse_km"]]
    .rename (columns ={"median_error_km":"mean_flow_median_km","rmse_km":"mean_flow_rmse_km"}),
    on ="lead_time_hours",
    )
    )
    comparison ["skill_vs_persistence_pct"]=100 *(
    1 -comparison ["model_rmse_km"]/comparison ["persistence_rmse_km"]
    )
    comparison ["skill_vs_mean_flow_pct"]=100 *(
    1 -comparison ["model_rmse_km"]/comparison ["mean_flow_rmse_km"]
    )
    fig ,ax =plt .subplots (figsize =(10 ,4.8 ))
    ax .plot (comparison ["lead_time_hours"],comparison ["persistence_median_km"],
    "--",color ="#666666",lw =2 ,label ="Velocity persistence")
    ax .plot (comparison ["lead_time_hours"],comparison ["mean_flow_median_km"],
    ":",color ="#999999",lw =2.5 ,label ="Spatial mean flow")
    ax .plot (comparison ["lead_time_hours"],comparison ["model_median_km"],
    color =color ,lw =2.8 ,label =model_label )
    ax .set (
    xlabel ="Lead time（\u5c0f\u65f6）",
    ylabel ="Median separation error（km）",
    title =f"{model_label } \u4e0e\u4e24\u4e2a\u7269\u7406\u57fa\u7ebf",
    )
    ax .set_xlim (1 ,72 )
    ax .grid (alpha =0.25 )
    ax .legend (frameon =False )
    plt .tight_layout ()
    fig .savefig (OUTPUT_DIR /f"{output_stem }_vs_baselines.png",dpi =200 ,bbox_inches ="tight")
    plt .show ()
    display (
    comparison .loc [comparison ["lead_time_hours"].isin ([24 ,48 ,72 ])]
    .round (2 )
    )
    return comparison 

evaluation ,lead_metrics ,case_metrics ,summary =evaluate_test_predictions ("Random Forest",y_pred_test )
display (summary .round (2 ))


# ## Analysis stage 6
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


comparison = compare_with_baselines(
    "Random Forest", lead_metrics, "#d95f02", "random_forest"
)


# ## Analysis stage 7
# 
# This stage documents the next part of the analysis. The following code and retained outputs provide the exact computational record.

# In[ ]:


cv_folds .to_csv (OUTPUT_DIR /"random_forest_cross_validation_folds.csv",index =False )
cv_summary .to_csv (OUTPUT_DIR /"random_forest_cross_validation_summary.csv",index =False )
lead_metrics .to_csv (OUTPUT_DIR /"random_forest_lead_metrics.csv",index =False )
case_metrics .to_csv (OUTPUT_DIR /"random_forest_test_case_metrics.csv",index =False )
comparison .to_csv (OUTPUT_DIR /"random_forest_vs_baselines.csv",index =False )
with open (OUTPUT_DIR /"random_forest_best_parameters.json","w",encoding ="utf-8")as f :
    json .dump ({"n_estimators":final_trees ,"bootstrap":True ,**best_params },f ,ensure_ascii =False ,indent =2 )
print ("\u7ed3\u679c\u5df2\u4fdd\u5b58\u5230：",OUTPUT_DIR )

