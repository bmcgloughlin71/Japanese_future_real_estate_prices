# EXP-003 — HistGradientBoostingRegressor

## Configuration

```json
{
  "train_rows": 1720073,
  "max_train_rows": 0,
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
| dev | 215,009 | 0.187362 | 6,741,698 | 0.648657 | 0.777272 | 0.619002 |
| test | 215,010 | 0.186485 | 6,812,060 | 0.715409 | 0.783091 | 0.622729 |
