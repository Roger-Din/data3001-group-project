# RQ2 extension: regional and current-conditioned models

This exploratory experiment compares four Ridge strategies while holding the algorithm fixed:

- **Global original:** the original 98 inputs and all 3,591 training cases.
- **Regional:** a separate Ridge model for each ocean regime.
- **Current-conditioned:** adds climatological origin current components and speed.
- **Current-residual:** predicts a correction to a trajectory obtained by holding the origin climatological current constant.

All selection uses grouped five-fold cross-validation within the training partition. Calibration cases remain unused and the fixed test partition is used only for final comparison.

Regional Ridge reduced 72-hour RMSE on Shelf (46.35 to 40.27 km), Kuroshio Core (92.08 to 86.26 km) and Other Offshore (56.53 to 54.36 km); Transition was essentially unchanged. Adding current information helped most clearly on Shelf. These findings are exploratory because the strategies were developed after viewing earlier test results.

The residual baseline is deliberately limited: it holds the forecast-origin climatological current fixed for 72 hours rather than querying the spatial current field at every predicted position. A stronger future test would use monthly climatology and out-of-region validation.
