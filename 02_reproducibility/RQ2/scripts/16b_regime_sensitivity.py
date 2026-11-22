"""RQ2 Step 16B: physical-regime threshold sensitivity (no model refitting).
Run from Data3001 root: python RQ2/16b_regime_sensitivity.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import label
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
CLIM = ROOT / 'RQ2/data/glorys_climatology_1994_2022.nc'
ERRORS = ROOT / 'RQ2/data/rq2_test_model_errors_long.csv'
OUT = ROOT / 'RQ2/data'
OUT.mkdir(parents=True, exist_ok=True)
SPEED_THRESHOLDS = (0.4, 0.5, 0.6)
BUFFERS_KM = (25, 50, 75)
N_BOOT = 2000
SEED = 3001
REGIMES = ['Shelf', 'Kuroshio Core', 'Transition', 'Other Offshore']
MODELS = ['Persistence', 'Mean Flow', 'Ridge', 'FNN', 'VAR(1)', 'KNN']

def xyz(lon, lat):
    lon, lat = np.deg2rad(np.asarray(lon, dtype=float)), np.deg2rad(np.asarray(lat, dtype=float))
    return np.column_stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)))

def chord_to_km(chord):
    return 2 * 6371.0088 * np.arcsin(np.clip(np.asarray(chord)/2, 0, 1))

def interval(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return (np.percentile(a, 2.5), np.percentile(a, 97.5)) if len(a) else (np.nan, np.nan)

def main():
    print('=== RQ2 STEP 16B: REGIME THRESHOLD SENSITIVITY ===')
    df = pd.read_csv(ERRORS)
    required = {'forecast_case_id','ID','model','final_72h_error_km','origin_longitude',
                'origin_latitude','depth_m','regime','glorys_match_valid'}
    assert required <= set(df), f'Missing columns: {required - set(df)}'
    assert len(df) == 757 * 6 and df.forecast_case_id.nunique() == 757
    assert not df.duplicated(['forecast_case_id','model']).any()
    assert set(df.model.unique()) == set(MODELS)
    cases = df.drop_duplicates('forecast_case_id').sort_values('forecast_case_id').copy().reset_index(drop=True)
    assert len(cases) == 757 and cases.ID.nunique() == 71
    assert (cases['regime'] != 'Unclassified').all()
    assert cases['glorys_match_valid'].fillna(False).astype(bool).all()
    assert cases.depth_m.notna().all()
    assert cases.depth_m.ge(0).all()

    with xr.open_dataset(CLIM) as ds:
        u = ds['u_mean'].squeeze(drop=True).transpose('latitude','longitude').values
        v = ds['v_mean'].squeeze(drop=True).transpose('latitude','longitude').values
        lons = ds.longitude.values
        lats = ds.latitude.values
    speed = np.hypot(u, v)
    wet = np.isfinite(speed)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    wet_tree = cKDTree(xyz(lon_grid[wet], lat_grid[wet]))
    wet_chord, wet_indices = wet_tree.query(xyz(cases.origin_longitude, cases.origin_latitude))
    match_km = chord_to_km(wet_chord)
    assert np.max(match_km) <= 15, f'Case-to-GLORYS grid distance exceeds 15km: {np.max(match_km):.2f}'
    # The wet-grid index follows NumPy boolean indexing order.
    wet_flat_indices = np.flatnonzero(wet)
    case_flat_indices = wet_flat_indices[wet_indices]
    case_rows, case_cols = np.unravel_index(case_flat_indices, wet.shape)

    wide = df.pivot(index='forecast_case_id', columns='model', values='final_72h_error_km').loc[cases.forecast_case_id]
    assert wide.notna().all().all()
    err = wide[MODELS].to_numpy(dtype=float)
    ids = cases.ID.to_numpy()
    unique_ids = np.unique(ids)
    # Shared resampling indices for all settings/models, preserving cross-regime overlap.
    rng = np.random.default_rng(SEED)
    by_id = [np.flatnonzero(ids == drifter) for drifter in unique_ids]
    boot_draws = rng.integers(0, len(unique_ids), size=(N_BOOT, len(unique_ids)))
    boot_idx = [np.concatenate([by_id[j] for j in draw]) for draw in boot_draws]

    summary_rows, contrast_rows, count_rows, label_rows, component_rows = [], [], [], [], []
    structure = np.ones((3,3), dtype=int)
    baseline_labels = cases['regime'].to_numpy()

    for threshold in SPEED_THRESHOLDS:
        labels, n_components = label(wet & (speed >= threshold), structure=structure)
        sizes = np.bincount(labels.ravel())
        if len(sizes) < 3:
            raise RuntimeError(f'Only {len(sizes)-1} components at {threshold}m/s')
        sizes[0] = 0
        top2 = np.argsort(sizes)[-2:][::-1]
        core_mask = np.isin(labels, top2)
        core_lons = lon_grid[core_mask]
        core_lats = lat_grid[core_mask]
        core_tree = cKDTree(xyz(core_lons, core_lats))
        core_dist_km = chord_to_km(core_tree.query(xyz(cases.origin_longitude, cases.origin_latitude))[0])
        is_core = core_mask[case_rows, case_cols]
        for rank, cid in enumerate(top2, start=1):
            cellmask = labels == cid
            component_rows.append({'speed_threshold_m_s': threshold,'rank': rank,'component_id': int(cid),
                                   'n_cells': int(sizes[cid]),'lon_min': float(lon_grid[cellmask].min()),
                                   'lon_max': float(lon_grid[cellmask].max()),
                                   'lat_min': float(lat_grid[cellmask].min()),'lat_max': float(lat_grid[cellmask].max())})
        print(f'Threshold {threshold:.1f}: top components {top2.tolist()}, sizes {sizes[top2].tolist()}')

        for buffer_km in BUFFERS_KM:
            # Same priority as Step 13: core > shelf > transition > other offshore.
            shelf = (~is_core) & (cases.depth_m.to_numpy() <= 200)
            transition = (~is_core) & (~shelf) & (
                (cases.depth_m.to_numpy() <= 1000) | (core_dist_km <= buffer_km))
            regimes = np.full(len(cases), 'Other Offshore', dtype=object)
            regimes[transition] = 'Transition'
            regimes[shelf] = 'Shelf'
            regimes[is_core] = 'Kuroshio Core'
            if threshold == 0.5 and buffer_km == 50:
                agreement = np.mean(regimes == baseline_labels)
                print(f'  MAIN-SETTING REPRODUCTION: {agreement:.2%} ({np.sum(regimes != baseline_labels)} mismatches)')
                if not np.array_equal(regimes, baseline_labels):
                    mismatches = cases.loc[regimes != baseline_labels, ['forecast_case_id','regime']].copy()
                    mismatches['recomputed_regime'] = regimes[regimes != baseline_labels]
                    mismatches.to_csv(OUT/'rq2_sensitivity_main_setting_mismatches.csv', index=False)
                    raise RuntimeError('0.5m/s, 50km does NOT reproduce Step 13 labels. Investigate before interpretation.')
            for k, name in enumerate(REGIMES):
                mask = regimes == name
                count_rows.append({'speed_threshold_m_s':threshold,'buffer_km':buffer_km,
                                   'regime':name,'n_cases':int(mask.sum()),'n_drifters':int(np.unique(ids[mask]).size)})
                for m, model in enumerate(MODELS):
                    x = err[mask, m]
                    if not len(x):
                        continue
                    reps_med, reps_rmse = [], []
                    for bi in boot_idx:
                        chosen = bi[mask[bi]]
                        if len(chosen):
                            e = err[chosen, m]
                            reps_med.append(np.median(e))
                            reps_rmse.append(np.sqrt(np.mean(e**2)))
                    lo_med, hi_med = interval(reps_med)
                    lo_rmse, hi_rmse = interval(reps_rmse)
                    summary_rows.append({'speed_threshold_m_s':threshold,'buffer_km':buffer_km,'regime':name,
                                         'model':model,'n_cases':len(x),'median_km':float(np.median(x)),
                                         'median_ci_lower_km':lo_med,'median_ci_upper_km':hi_med,
                                         'rmse_km':float(np.sqrt(np.mean(x**2))),
                                         'rmse_ci_lower_km':lo_rmse,'rmse_ci_upper_km':hi_rmse})
            core = regimes == 'Kuroshio Core'
            shelf = regimes == 'Shelf'
            for m, model in enumerate(MODELS):
                x, y = err[core, m], err[shelf, m]
                point_med = float(np.median(x)-np.median(y))
                point_rmse = float(np.sqrt(np.mean(x*x))-np.sqrt(np.mean(y*y)))
                reps_med, reps_rmse = [], []
                for bi in boot_idx:
                    xc, ys = err[bi[core[bi]],m], err[bi[shelf[bi]],m]
                    if len(xc) and len(ys):
                        reps_med.append(np.median(xc)-np.median(ys))
                        reps_rmse.append(np.sqrt(np.mean(xc**2))-np.sqrt(np.mean(ys**2)))
                for metric, point, reps in [('median',point_med,reps_med),('rmse',point_rmse,reps_rmse)]:
                    lo, hi = interval(reps)
                    contrast_rows.append({'speed_threshold_m_s':threshold,'buffer_km':buffer_km,'model':model,
                                          'metric':metric,'core_minus_shelf_km':point,
                                          'ci_lower_km':lo,'ci_upper_km':hi,
                                          'ci_excludes_zero':bool(lo>0 or hi<0),
                                          'n_valid_bootstrap':len(reps)})
            for i, case_id in enumerate(cases.forecast_case_id):
                label_rows.append({'forecast_case_id':case_id,'ID':ids[i],
                                   'speed_threshold_m_s':threshold,'buffer_km':buffer_km,
                                   'regime':regimes[i],'distance_to_core_km':core_dist_km[i]})
            counts = {r:int(np.sum(regimes==r)) for r in REGIMES}
            print(f'  buffer {buffer_km:>2}km: {counts}')

    pd.DataFrame(summary_rows).to_csv(OUT/'rq2_sensitivity_model_summary.csv',index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(OUT/'rq2_sensitivity_core_shelf_contrasts.csv',index=False)
    pd.DataFrame(count_rows).to_csv(OUT/'rq2_sensitivity_regime_counts.csv',index=False)
    pd.DataFrame(label_rows).to_csv(OUT/'rq2_sensitivity_case_labels.csv',index=False)
    pd.DataFrame(component_rows).to_csv(OUT/'rq2_sensitivity_core_components.csv',index=False)
    print('\n=== FNN CORE MINUS SHELF, MEDIAN ERROR (KM) ===')
    print(contrasts[(contrasts.model=='FNN') & (contrasts.metric=='median')][
        ['speed_threshold_m_s','buffer_km','core_minus_shelf_km','ci_lower_km','ci_upper_km','ci_excludes_zero']
    ].round(2).to_string(index=False))
    print('\nSaved five CSV files to:',OUT)
    print('=== STEP 16B COMPLETE ===')

if __name__ == '__main__':
    main()
