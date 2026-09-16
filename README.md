# DATA3001 Global Drifter Data Analysis Project

## 1. Project Overview

This is a group project for UNSW DATA3001: Data Science and Decisions in Practice.

The project uses hourly quality-controlled data from the NOAA Global Drifter Program (GDP) to investigate surface ocean circulation and sea-surface temperature patterns within a selected region.

The current work represents the initial data exploration stage. The final study region and research question will be determined after group discussion.

## 2. Research Background

Approximately 70% of Earth’s surface is covered by oceans, but there are far fewer fixed observation stations at sea than weather stations on land. Satellites provide broad ocean coverage, but in-situ observations are still required to complement and validate satellite measurements and numerical models.

The Global Drifter Program collects observations by deploying drifting buoys throughout the world’s oceans. These drifters record their position, observation time and sea-surface temperature.

A standard drifter normally carries a drogue at a depth of approximately 15 metres. The drogue helps the buoy follow near-surface ocean currents. After the drogue is lost, the buoy becomes more affected by wind and waves. The variable `drogue_status` must therefore be considered when analysing surface circulation.

## 3. Dataset

This project uses the Global Drifter Program one-hour interpolated quality-controlled dataset.

The dataset includes:

- Time coverage: October 1987 to October 2022
- Number of drifter trajectories: 19,396
- Number of hourly observations: approximately 197.2 million
- Data size: approximately 16 GB
- Number of variables: approximately 60
- Python data structure: `xarray.Dataset`

The dataset contains two main levels:

- `traj`: metadata with one value for each drifter trajectory
- `obs`: hourly observations recorded along the trajectories

## 4. Main Variables

| Variable | Meaning | Main use |
|---|---|---|
| `id` | Unique drifter identifier | Separates individual drifters and trajectories |
| `time` | Observation time | Time filtering and seasonal analysis |
| `lon` | Longitude | Identifies the drifter’s location |
| `lat` | Latitude | Identifies the drifter’s location |
| `ve` | Eastward velocity component | Describes east–west surface-current movement |
| `vn` | Northward velocity component | Describes north–south surface-current movement |
| `sst` | Fitted sea-surface temperature | Analyses ocean temperature and its variation |
| `drogue_status` | Whether the drogue is attached | Identifies observations that more closely follow ocean currents |
| `flg_sst` | Sea-surface temperature quality flag | Filters reliable temperature observations |
| `rowsize` | Number of observations in each trajectory | Connects trajectory metadata with hourly observations |

A more detailed explanation of the dataset variables is provided in the variable dictionary.

## 5. Current Practice Region

The California Current region is currently being used as a practice case for initial data exploration.

The practice region is defined as:

- Longitude: −130° to −118°
- Latitude: 30° to 45°
- Time period: 2020

The initial subset contains:

- 5,265 hourly observations
- 13 distinct drifters
- 4,787 observations available for sea-surface temperature analysis

This region is currently a practice example and may not become the group’s final study region.

## 6. Preliminary Research Directions

The group will select a final research question from possible directions such as:

1. Mean ocean-current speed and direction within a selected region
2. Seasonal changes in ocean circulation
3. Spatial and temporal patterns in sea-surface temperature
4. The relationship between sea-surface temperature and current speed
5. Differences between drifters with and without an attached drogue

## 7. Current Progress

The following tasks have been completed:

- Created a Python virtual environment and configured JupyterLab
- Connected to the Global Drifter Program dataset
- Examined the dataset structure and main variables
- Selected a California Current practice region
- Checked missing values and sea-surface temperature quality flags
- Created drifter trajectory and observation-density plots
- Produced initial spatial-grid statistics
- Created the group GitHub repository

## 8. Repository Structure

```text
data3001-group-project/
├── README.md
├── .gitignore
├── 01_data_check_clean.ipynb
└── data_dictionary/
    └── gdp1h_variable_dictionary.csv
```

File descriptions:

- `README.md`: project background, research direction, methods and task allocation
- `.gitignore`: prevents virtual environments, temporary files and large data files from being uploaded
- `01_data_check_clean.ipynb`: data access, quality checks and preliminary exploration
- `data_dictionary/gdp1h_variable_dictionary.csv`: definitions, units and analytical uses for all dataset variables

## 9. Team Responsibilities

| Team member | Current responsibility | GitHub branch | Status |
|---|---|---|---|
| Roger Ding | Environment setup, variable inspection and preliminary data exploration | To be confirmed | In progress |
| To be added | To be confirmed | To be confirmed | Not started |
| To be added | To be confirmed | To be confirmed | Not started |
| To be added | To be confirmed | To be confirmed | Not started |

The team will review and update task allocation during weekly meetings.

## 10. Next Steps

- Invite all team members and the supervisor to the GitHub repository
- Select the final study region
- Evaluate data adequacy within the selected region
- Finalise the research question
- Allocate data processing, analysis, visualisation and writing tasks
- Complete the preliminary analysis and update this README

## 11. Data Sources

- NOAA Global Drifter Program
- NOAA GDP ERDDAP one-hour interpolated quality-controlled dataset
- CloudDrift `gdp1h` dataset interface