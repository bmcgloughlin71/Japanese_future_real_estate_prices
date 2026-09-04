# EXP-006 — HistGradientBoostingRegressor

## Configuration

```json
{
  "train_rows": 1720073,
  "max_train_rows": 0,
  "max_iter": 300,
  "actual_iter": 300,
  "learning_rate": 0.08,
  "max_leaf_nodes": 63,
  "l2_regularization": 0.0,
  "target": "log10(TotalTransactionValue + 1)"
}
```

## Metrics

| Split | n | median_ape | mae_yen | r2_price | r2_log10 | within_25pct |
|---|---:|---:|---:|---:|---:|---:|
| dev | 215,009 | 0.177653 | 6,358,258 | 0.700952 | 0.791704 | 0.641355 |
| test | 215,010 | 0.175551 | 6,385,012 | 0.779938 | 0.797283 | 0.644324 |
