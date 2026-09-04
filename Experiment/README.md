# Experiment workspace

This folder is the working area for attempts to improve the Japanese housing price model.

## Goals

1. Keep experiments reproducible.
2. Record what changed, why it was tried, and whether it improved accuracy.
3. Avoid data leakage, especially when creating price-history or market aggregate features.

## Folder layout

- `LOG.md` — human-readable running log of decisions, experiments, and outcomes.
- `experiment_registry.csv` — compact machine-readable index of experiments.
- `templates/experiment_template.md` — checklist/template for each experiment note.
- `scripts/evaluate_model.py` — repeatable evaluation script for the current Keras model.
- `results/` — generated metrics/reports from experiment runs.

## Recommended workflow

1. Define an experiment in `LOG.md` before running it.
2. Run the relevant script or notebook.
3. Save outputs under `results/`.
4. Add one row to `experiment_registry.csv`.
5. Summarise whether the result should be kept, rejected, or investigated further.

## Core metrics

Primary metrics:

- Median absolute percentage error (`median_ape`)
- Mean absolute error in yen (`mae_yen`)
- Raw-price R² (`r2_price`)
- Log-price R² (`r2_log10`)

Required breakdowns where possible:

- Property type
- Prefecture / region
- Price quantile
- Transaction year
- Condo-like vs non-condo-like
