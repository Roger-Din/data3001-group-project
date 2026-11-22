# Data provenance and reconstruction

## Included frozen inputs

`processed/` contains the exact leakage-safe data used for the final analysis:

- `region_A_observations.parquet`: 562,944 quality-controlled hourly observations.
- `region_A_segments.parquet`: continuous drifter segments.
- `forecast_origins.parquet`: 5,117 non-overlapping 24-hour-history/72-hour-target cases.
- `drifter_splits.csv`: drifter-level training, calibration and test assignment.
- `qc_summary.csv`: preprocessing summary.

These files are included so a reviewer can reproduce the report without downloading or reprocessing the entire global archive.

## Upstream sources

- Drifter observations: NOAA/AOML Global Drifter Program hourly quality-controlled trajectories (GDP 1-hour product; Elipot et al., 2022). Variables used in this project are documented in `../data_dictionary/gdp1h_variable_dictionary.csv`.
- Bathymetry: GEBCO gridded bathymetry, used to assign depth and shelf categories.
- Ocean currents: Copernicus Marine/GLORYS reanalysis, aggregated to the cached climatology in `../RQ2/data/glorys_climatology_1994_2022.nc`.

The raw global GDP, GEBCO and GLORYS archives are not duplicated in this repository because of size and redistribution considerations. The processed Region-A inputs and cached environmental features needed by the submitted analyses are included.

## Leakage-safe construction

Run `../scripts/build_processed_dataset.py` only when the raw GDP source has been supplied locally. The final frozen split must retain entire drifter IDs within one partition. The automated validator confirms the expected case counts (3,591 training, 769 calibration and 757 test) and checks that no drifter ID crosses partitions.
