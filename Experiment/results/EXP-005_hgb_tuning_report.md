# EXP-005 — HGB tuning

Selection metric: lowest dev `median_ape`. Only the best dev config was evaluated on test.

## Best config

```json
{
  "learning_rate": 0.08,
  "max_iter": 300,
  "max_leaf_nodes": 63,
  "l2_regularization": 0.0,
  "min_samples_leaf": 20,
  "random_seed": 42,
  "actual_iter_refit": 300
}
```

## Best metrics

| Split | median_ape | mae_yen | rmse_yen | r2_price | r2_log10 | within_25pct | p95_ape |
|---|---:|---:|---:|---:|---:|---:|---:|
| dev | 0.179620 | 6,440,058 | 27,790,293 | 0.701981 | 0.788619 | 0.637201 | 0.854784 |
| test | 0.177810 | 6,484,904 | 25,680,793 | 0.759275 | 0.793693 | 0.639412 | 0.852457 |

## Dev sweep

| rank | median_ape | mae_yen | r2_log10 | actual_iter | config |
|---:|---:|---:|---:|---:|---|
| 1 | 0.179620 | 6,440,058 | 0.788619 | 300 | `{"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 63, "l2_regularization": 0.0, "min_samples_leaf": 20}` |
| 2 | 0.179697 | 6,438,601 | 0.788791 | 300 | `{"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 63, "l2_regularization": 0.01, "min_samples_leaf": 20}` |
| 3 | 0.181818 | 6,523,523 | 0.784997 | 300 | `{"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 63, "l2_regularization": 0.0, "min_samples_leaf": 20}` |
| 4 | 0.181819 | 6,525,238 | 0.784889 | 300 | `{"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 63, "l2_regularization": 0.01, "min_samples_leaf": 20}` |
| 5 | 0.184846 | 6,641,113 | 0.781628 | 300 | `{"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.0, "min_samples_leaf": 20}` |
| 6 | 0.184928 | 6,638,776 | 0.781509 | 300 | `{"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.01, "min_samples_leaf": 20}` |
| 7 | 0.186055 | 6,734,952 | 0.777119 | 300 | `{"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 63, "l2_regularization": 0.01, "min_samples_leaf": 20}` |
| 8 | 0.186055 | 6,735,003 | 0.777118 | 300 | `{"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 63, "l2_regularization": 0.0, "min_samples_leaf": 20}` |
| 9 | 0.188033 | 6,794,230 | 0.775639 | 300 | `{"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.0, "min_samples_leaf": 20}` |
| 10 | 0.188214 | 6,790,308 | 0.775932 | 300 | `{"learning_rate": 0.05, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.01, "min_samples_leaf": 20}` |
| 11 | 0.193643 | 7,044,169 | 0.766049 | 300 | `{"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.01, "min_samples_leaf": 20}` |
| 12 | 0.193643 | 7,044,163 | 0.766050 | 300 | `{"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.0, "min_samples_leaf": 20}` |
