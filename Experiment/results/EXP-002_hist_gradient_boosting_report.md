# EXP-002 — HistGradientBoostingRegressor

## Configuration

```json
{
  "train_rows": 500000,
  "max_train_rows": 500000,
  "max_iter": 300,
  "actual_iter": 300,
  "learning_rate": 0.05,
  "max_leaf_nodes": 31,
  "l2_regularization": 0.0,
  "target": "log10(TotalTransactionValue + 1)"
}
```

## Metrics

| Split | n | median_ape | mae_yen | r2_price | r2_log10 | within_25pct |
|---|---:|---:|---:|---:|---:|---:|
| dev | 215,009 | 0.188033 | 6,794,230 | 0.647040 | 0.775639 | 0.617360 |
| test | 215,010 | 0.187094 | 6,863,022 | 0.712251 | 0.781411 | 0.621101 |
