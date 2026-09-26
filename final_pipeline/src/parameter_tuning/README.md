# Parameter Tuning Module

Utilities for BERTopic hyperparameter search and visual inspection.

## What this module does

- Evaluates topic model configurations against quality metrics (coherence, diversity, silhouette score, outlier percentage).
- Supports sensitivity analysis and grid search workflows.
- Produces charts to compare parameter configurations.

## Why this module exists

Topic quality depends strongly on UMAP and HDBSCAN settings. This module supports the three-phase tuning process (sensitivity analysis, grid search, manual validation) used to select era-specific parameters before final runs.

## Files

```
parameter_tuning/
├── helpers.py
├── visualizations.py
├── tuning-notebook.ipynb
├── configs/
└── results/
```

## How to use

Primary workflow is notebook-based:

- Open `tuning-notebook.ipynb`
- Choose era
- Run baseline, sensitivity analysis, then grid search
- Inspect saved plots and result tables in `results/`

## Programmatic entry points

`helpers.py` provides core tuning utilities.

`visualizations.py` provides comparison plots (metric charts, UMAP and topic distribution views).

## Outputs

Typical outputs in `parameter_tuning/results/`:

- search result CSV files
- selected and best config JSON files
- plot artifacts for inspection

## Notes

- This module is optional for routine execution.
- Final production runs use `final_pipeline/src/config.json` through the main pipeline notebook.
