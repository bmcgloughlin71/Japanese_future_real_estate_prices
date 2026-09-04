# Experiment log

This log tracks work to improve the Japanese housing price model.

## 2026-09-04 — Experiment workspace created

### Objective
Create a reproducible, human-readable workspace for model-improvement experiments.

### Changes
- Added an `Experiment/` folder.
- Added a README describing the workflow and evaluation metrics.
- Added a registry CSV for compact experiment tracking.
- Added an experiment-note template.
- Added a baseline evaluation script for the current Keras model.

### Outcome
No model change yet. This is infrastructure for future experiments.

### Next planned steps
1. Run the baseline evaluation script and save results.
2. Use the baseline as the comparison point for future feature/model experiments.
3. Research external Japanese datasets that may improve location, macroeconomic, and market context features.

## 2026-09-04 — EXP-001 baseline test evaluation

### Objective
Establish the current committed Keras model's held-out test performance before trying model-improvement experiments.

### Command
```bash
python3 Experiment/scripts/evaluate_model.py \
  --split test \
  --output Experiment/results/baseline_test_metrics.json \
  --markdown Experiment/results/baseline_test_report.md
```

### Outcome
Baseline test metrics were written to:
- `Experiment/results/baseline_test_metrics.json`
- `Experiment/results/baseline_test_report.md`

Key metrics:
- `median_ape`: 0.1904
- `mae_yen`: ¥7,645,337
- `r2_price`: 0.5486
- `r2_log10`: 0.7620
- `within_25pct`: 0.6096

### Decision
Keep as the baseline for future experiments.

## 2026-09-04 — EXP-002 HistGradientBoosting baseline on existing 32 features

### Objective
Test whether a tabular tree-based model can outperform the current dense neural network without adding new data or changing production app code.

### Hypothesis
Gradient-boosted decision trees should handle tabular feature interactions and non-linear thresholds better than the current dense network, especially for raw-yen error and high-price outliers.

### Change
Added and ran:
```bash
python3 Experiment/scripts/train_hist_gradient_boosting.py \
  --experiment-id EXP-002 \
  --max-train-rows 500000 \
  --max-iter 300 \
  --learning-rate 0.05
```

The model used the same 32 production features from `artifacts/preprocess_params.json` and the same target transform: `log10(TotalTransactionValue + 1)`.

### Leakage check
No new features were added. The experiment reused the existing train/dev/test CSVs and production preprocessing. Therefore it has the same leakage profile as the current baseline: no obvious preprocessing leakage, but the random row split may still allow nearby/repeated-location information to cross splits.

### Outcome
Results were written to:
- `Experiment/results/EXP-002_hist_gradient_boosting_metrics.json`
- `Experiment/results/EXP-002_hist_gradient_boosting_report.md`

Test-set comparison against the current Keras baseline:

| Metric | Keras baseline | EXP-002 HGB | Change |
|---|---:|---:|---:|
| median_ape | 0.190421 | 0.187094 | improved slightly |
| mae_yen | 7,645,337 | 6,863,022 | improved ~10.2% |
| rmse_yen | 35,165,333 | 28,077,291 | improved ~20.2% |
| r2_price | 0.548629 | 0.712251 | improved strongly |
| r2_log10 | 0.761957 | 0.781411 | improved modestly |
| within_25pct | 0.609572 | 0.621101 | improved slightly |
| p95_ape | 1.120944 | 0.889660 | improved strongly |

### Decision
Promising. Keep this experiment as evidence that tree/boosting models are worth pursuing. It improves raw-yen error and tail robustness substantially, while only slightly improving typical percentage error.

### Follow-up ideas
1. Run HGB on all training rows instead of a 500k sample.
2. Tune `max_leaf_nodes`, `learning_rate`, `max_iter`, and `l2_regularization` against dev.
3. Compare against LightGBM/CatBoost if dependencies can be added in the experiment environment.
4. Evaluate grouped/time/location split performance to test whether random-split performance is optimistic.

## 2026-09-04 — EXP-003 HistGradientBoosting on all training rows

### Objective
Check whether EXP-002 improves further when trained on the full training split rather than a 500k-row sample.

### Command
```bash
python3 Experiment/scripts/train_hist_gradient_boosting.py \
  --experiment-id EXP-003 \
  --max-train-rows 0 \
  --max-iter 300 \
  --learning-rate 0.05
```

### Change
Same model family and 32-feature input as EXP-002, but uses all 1,720,073 training rows.

### Leakage check
Same leakage profile as EXP-002: no new features, existing random split only.

### Outcome
Results were written to:
- `Experiment/results/EXP-003_hist_gradient_boosting_metrics.json`
- `Experiment/results/EXP-003_hist_gradient_boosting_report.md`

Test-set comparison:

| Metric | Keras baseline | EXP-002 HGB 500k | EXP-003 HGB all rows |
|---|---:|---:|---:|
| median_ape | 0.190421 | 0.187094 | 0.186485 |
| mae_yen | 7,645,337 | 6,863,022 | 6,812,060 |
| rmse_yen | 35,165,333 | 28,077,291 | 27,922,769 |
| r2_price | 0.548629 | 0.712251 | 0.715409 |
| r2_log10 | 0.761957 | 0.781411 | 0.783091 |
| within_25pct | 0.609572 | 0.621101 | 0.622729 |
| p95_ape | 1.120944 | 0.889660 | 0.883646 |

### Decision
Keep. Full-data HGB is the strongest result so far, though the gain over the 500k-row HGB is modest. The main takeaway remains that a tree/boosting model is a better fit for this tabular problem than the current dense NN.

## 2026-09-04 — EXP-004 dev-set calibration of existing Keras predictions

### Objective
Test whether the existing Keras model can be improved by a simple post-processing calibration from predicted log-price to true log-price, fitted only on the dev split and evaluated on test.

### Hypothesis
If the neural network predictions are systematically biased or compressed, an affine or isotonic calibration layer may improve yen-space and tail metrics without retraining the model.

### Change
Added and ran:
```bash
python3 Experiment/scripts/calibrate_keras_predictions.py --experiment-id EXP-004
```

The script fits two calibrators on dev predictions:
- affine linear map: `true_log = a * pred_log + b`
- isotonic monotonic map: flexible non-linear calibration

### Leakage check
Calibration is trained only on the dev split and evaluated on test. This is acceptable for an experiment, but if adopted, the test split should remain final-only and calibration choices should be selected on dev.

### Outcome
Results were written to:
- `Experiment/results/EXP-004_keras_calibration_metrics.json`
- `Experiment/results/EXP-004_keras_calibration_report.md`

Test-set comparison:

| Metric | Keras baseline | Affine calibration | Isotonic calibration | EXP-003 HGB all rows |
|---|---:|---:|---:|---:|
| median_ape | 0.190421 | 0.187094 | 0.187692 | 0.186485 |
| mae_yen | 7,645,337 | 7,000,537 | 6,848,414 | 6,812,060 |
| rmse_yen | 35,165,333 | 29,728,012 | 26,259,939 | 27,922,769 |
| r2_price | 0.548629 | 0.677421 | 0.748295 | 0.715409 |
| r2_log10 | 0.761957 | 0.779449 | 0.780042 | 0.783091 |
| within_25pct | 0.609572 | 0.616176 | 0.619027 | 0.622729 |
| p95_ape | 1.120944 | 0.893268 | 0.903968 | 0.883646 |

### Decision
Keep as a useful finding but do not prefer it over EXP-003 for typical accuracy. Calibration substantially improves the Keras model, especially RMSE/raw R², but the all-row HGB model remains better on median APE, log R², within-25%, and p95 APE. Isotonic calibration has the best RMSE/raw R² seen so far, so it may be useful in an ensemble or for reducing extreme yen errors.

## 2026-09-04 — EXP-005 small HGB hyperparameter sweep on 500k rows

### Objective
Tune the HistGradientBoostingRegressor using the dev split while keeping the same 32 production features.

### Command
```bash
python3 Experiment/scripts/tune_hist_gradient_boosting.py \
  --experiment-id EXP-005 \
  --max-train-rows 500000
```

### Change
Added and ran a small grid over:
- learning rate: 0.03, 0.05, 0.08
- max leaf nodes: 31, 63
- L2 regularization: 0.0, 0.01
- max iterations: 300
- min samples per leaf: 20

Model selection used dev `median_ape`. Only the best dev configuration was evaluated on test.

### Leakage check
The test split was not used for selecting the hyperparameters; selection used dev median APE. The experiment still inherits the existing random row split caveat.

### Outcome
Best config:
- learning_rate: 0.08
- max_leaf_nodes: 63
- l2_regularization: 0.0
- min_samples_leaf: 20
- max_iter: 300

Results were written to:
- `Experiment/results/EXP-005_hgb_tuning_metrics.json`
- `Experiment/results/EXP-005_hgb_tuning_report.md`

Test-set metrics:
- median_ape: 0.177810
- mae_yen: ¥6,484,904
- rmse_yen: ¥25,680,793
- r2_price: 0.759275
- r2_log10: 0.793693
- within_25pct: 0.639412
- p95_ape: 0.852457

### Decision
Keep. Tuning gives a meaningful gain over EXP-003 even on the 500k training sample.

## 2026-09-04 — EXP-006 tuned HGB on all training rows

### Objective
Train the best EXP-005 HGB configuration on all 1,720,073 training rows.

### Command
```bash
python3 Experiment/scripts/train_hist_gradient_boosting.py \
  --experiment-id EXP-006 \
  --max-train-rows 0 \
  --max-iter 300 \
  --learning-rate 0.08 \
  --max-leaf-nodes 63
```

### Change
Same best hyperparameters from EXP-005, but using the full training split.

### Leakage check
Hyperparameters came from dev tuning. Test was used only for final evaluation of this candidate. Existing random row split caveat remains.

### Outcome
Results were written to:
- `Experiment/results/EXP-006_hist_gradient_boosting_metrics.json`
- `Experiment/results/EXP-006_hist_gradient_boosting_report.md`

Test-set comparison:

| Metric | Keras baseline | EXP-003 HGB all rows | EXP-006 tuned HGB all rows |
|---|---:|---:|---:|
| median_ape | 0.190421 | 0.186485 | 0.175551 |
| mae_yen | 7,645,337 | 6,812,060 | 6,385,012 |
| rmse_yen | 35,165,333 | 27,922,769 | 24,553,924 |
| r2_price | 0.548629 | 0.715409 | 0.779938 |
| r2_log10 | 0.761957 | 0.783091 | 0.797283 |
| within_25pct | 0.609572 | 0.622729 | 0.644324 |
| p95_ape | 1.120944 | 0.883646 | 0.842847 |

### Decision
Strong keep. This is the best model experiment so far across all tracked metrics. It improves median APE from 19.0% to 17.6%, MAE by about 16.5%, RMSE by about 30.2%, and raw R² from 0.55 to 0.78 versus the current Keras baseline.

## 2026-09-04 — Diagnostic plot: true vs predicted prices for baseline and experiments

### Objective
Create a visual comparison of true versus predicted prices on the held-out test split for all completed model experiments.

### Command
```bash
python3 Experiment/scripts/plot_true_vs_predicted.py \
  --output Experiment/results/true_vs_predicted_all_experiments.png \
  --sample 80000
```

### Output
- `Experiment/results/true_vs_predicted_all_experiments.png`

### Notes
The plot uses `log10(price + 1)` on both axes and includes a red dashed perfect-prediction line. Each panel title includes the test-set median absolute percentage error for that model/correction. The scatter is sampled to 80,000 test points per panel for readability.

## 2026-09-04 — EXP-007 cheap-property classifier + specialist regressor

### Objective
Address the very poor performance for ultra-cheap properties by adding a two-stage pipeline:
1. classify whether a property is likely to sell for `<= ¥1,000,000`,
2. use a specialist cheap-property regressor for classifier-positive cases,
3. otherwise use the tuned general HGB regressor.

### Command
```bash
python3 Experiment/scripts/cheap_property_two_stage.py \
  --experiment-id EXP-007 \
  --cheap-threshold-yen 1000000
```

### Change
Added and ran `Experiment/scripts/cheap_property_two_stage.py`.

Models trained:
- `HistGradientBoostingClassifier` with `class_weight="balanced"` for price `<= ¥1m`.
- Tuned general HGB regressor matching EXP-006 settings.
- Cheap-tail HGB regressor trained only on training rows with price `<= ¥1m`.

The classifier threshold was selected on dev to minimize overall median APE, with cheap-tail APE and MAE as tie-breakers.

### Leakage check
Classifier, general regressor, and cheap regressor were trained only on the training split. Threshold selection used dev only. Test was used only for final evaluation. Existing caveat remains: the underlying splits are random row splits, not time/location-grouped splits.

### Output
- `Experiment/results/EXP-007_cheap_two_stage_metrics.json`
- `Experiment/results/EXP-007_cheap_two_stage_report.md`
- `Experiment/results/EXP-007_cheap_two_stage_plot.png`

### Classifier result on test
At the dev-selected threshold `0.976371`:
- prevalence: 0.81% cheap properties
- ROC AUC: 0.9502
- average precision: 0.1659
- precision: 0.2703
- recall: 0.0230
- F1: 0.0424
- confusion matrix: TP=40, FP=108, FN=1700, TN=213162

Interpretation: the probability ranking is informative (`ROC AUC` is high), but because cheap properties are very rare, the conservative threshold catches very few of them.

### Regression result on test

| Metric | EXP-006 general HGB | EXP-007 two-stage |
|---|---:|---:|
| median_ape | 0.175551 | 0.175600 |
| mae_yen | 6,385,012 | 6,387,144 |
| rmse_yen | 24,553,924 | 24,555,741 |
| r2_price | 0.779938 | 0.779905 |
| r2_log10 | 0.797283 | 0.793806 |
| within_25pct | 0.644324 | 0.644184 |
| p95_ape | 0.842847 | 0.843209 |

Cheap true-price tail only (`<= ¥1m`):

| Metric | EXP-006 general HGB | EXP-007 two-stage |
|---|---:|---:|
| median_ape | 4.833708 | 4.766134 |
| mae_yen | 3,876,588 | 3,868,106 |
| within_25pct | 0.009195 | 0.010920 |

### Decision
Reject as a replacement for EXP-006. The two-stage pipeline only marginally improves the true cheap tail and slightly worsens the overall metrics. The classifier can rank cheap cases, but the trade-off is poor: thresholds that catch many cheap cases create many false positives and damage normal-property predictions.

### Follow-up ideas
1. Revisit this after adding external rural/location anchors such as official land price and station/ridership features.
2. Try a softer blend between general and cheap regressors instead of hard classifier switching.
3. Consider treating `<= ¥1m` sales as a distinct anomaly/distress-sale class rather than ordinary market-price regression.

## 2026-09-04 — EXP-008 cheap-property classifier threshold sweep

### Objective
Investigate whether EXP-007 failed because the cheap-property classifier threshold was too conservative, despite strong ROC AUC.

### Command
```bash
python3 Experiment/scripts/cheap_threshold_sweep.py \
  --experiment-id EXP-008 \
  --cheap-threshold-yen 1000000
```

### Change
Added and ran `Experiment/scripts/cheap_threshold_sweep.py`. The script retrains the EXP-007 classifier/general-regressor/cheap-regressor setup, then evaluates fixed classifier thresholds from 0.10 to 0.99 on dev and test.

### Leakage check
Same as EXP-007: models train on training split only; threshold analysis uses held-out dev/test for diagnosis. If a threshold is adopted, it should be selected on dev and final test should be used once.

### Output
- `Experiment/results/EXP-008_cheap_threshold_sweep.csv`
- `Experiment/results/EXP-008_cheap_threshold_sweep.json`
- `Experiment/results/EXP-008_cheap_threshold_sweep_report.md`
- `Experiment/results/EXP-008_cheap_threshold_sweep_plot.png`

### Key test results

| Threshold | Precision | Recall | TP | FP | Overall median APE | Cheap-tail median APE | Overall within 25% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.027 | 0.980 | 1706 | 60612 | 0.2321 | 0.4129 | 0.5229 |
| 0.50 | 0.058 | 0.884 | 1538 | 25165 | 0.1906 | 0.4383 | 0.6026 |
| 0.60 | 0.067 | 0.846 | 1472 | 20444 | 0.1870 | 0.4534 | 0.6115 |
| 0.80 | 0.102 | 0.701 | 1220 | 10694 | 0.1810 | 0.5251 | 0.6281 |
| 0.85 | 0.124 | 0.632 | 1100 | 7750 | 0.1792 | 0.6352 | 0.6329 |
| 0.90 | 0.160 | 0.501 | 872 | 4573 | 0.1776 | 2.5895 | 0.6380 |
| 0.95 | 0.255 | 0.232 | 404 | 1181 | 0.1760 | 4.2950 | 0.6430 |
| 0.976 | 0.270 | 0.023 | 40 | 108 | 0.1756 | 4.7661 | 0.6442 |
| 0.99 | 0.000 | 0.000 | 0 | 10 | 0.1756 | 4.8337 | 0.6443 |

### Interpretation
The high ROC AUC was meaningful: lowering the threshold catches many cheap properties and massively improves the true cheap-tail median APE. However, because cheap properties are rare, recall-focused thresholds create many false positives, routing normal properties through the cheap specialist and damaging overall performance.

### Decision
Do not adopt a hard two-stage switch as the main model yet. If the product goal is to avoid severe overpricing of likely distressed/ultra-cheap homes, threshold 0.80–0.85 is a defensible warning/safety setting. If the goal is best overall point prediction, keep EXP-006.

### Follow-up ideas
1. Use the classifier as a warning/uncertainty flag rather than hard switching.
2. Try soft blending: `combined_log = p_cheap * cheap_pred + (1 - p_cheap) * general_pred`, possibly with a calibrated/sharpened probability.
3. Add official land-price and station features, then repeat the threshold sweep.
