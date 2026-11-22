# RQ1 Python scripts

This directory contains Python exports of every canonical RQ1 notebook. The
folder structure mirrors `RQ1/notebooks/`:

- `core/`: the two physical baselines and four initial learned models;
- `extensions/`: model-family extensions, optimisation and champion comparison;
- `analysis/`: final comparisons and robustness diagnostics.

The notebooks remain the main reproducible record because they include the
explanation and saved outputs. These scripts provide a plain-Python version of
the same code for inspection, version comparison and command-line execution.

Run a script from the `02_reproducibility` directory, for example:

```bash
python3 RQ1/scripts/core/01_velocity_persistence_baseline.py
```

Model-training scripts can take substantially longer than the analysis scripts.
The complete confirmatory order is documented in `docs/EXECUTION_ORDER.md`.

