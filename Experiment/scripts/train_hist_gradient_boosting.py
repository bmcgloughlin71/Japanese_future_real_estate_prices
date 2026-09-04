"""Train a scikit-learn HistGradientBoostingRegressor experiment.

This is intentionally kept under Experiment/ so it does not change the production
FastAPI app or current Keras model.

Examples:
    python3 Experiment/scripts/train_hist_gradient_boosting.py --max-train-rows 500000
    python3 Experiment/scripts/train_hist_gradient_boosting.py --max-iter 300 --learning-rate 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from app.preprocess import PARAMS, build_feature_vector


SPLIT_PATHS = {
    "train": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_training_data.csv",
    "dev": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_dev_data.csv",
    "test": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_test_data.csv",
}


def load_xy(split: str, max_rows: int | None = None, random_seed: int = 42) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(SPLIT_PATHS[split])
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_seed).sort_index()
    x = np.asarray([build_feature_vector(row) for row in df.to_dict("records")], dtype=np.float32)
    y = df["TotalTransactionValue"].to_numpy(dtype=float)
    return df, x, y


def metrics(y_true: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
    y_pred = np.power(10, pred_log) - 1
    abs_err = np.abs(y_pred - y_true)
    ape = abs_err / y_true
    y_log = np.log10(y_true + 1)
    return {
        "n": int(len(y_true)),
        "mae_yen": float(np.mean(abs_err)),
        "median_abs_error_yen": float(np.median(abs_err)),
        "rmse_yen": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "r2_price": float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "r2_log10": float(1 - np.sum((y_log - pred_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_25pct": float(np.mean(ape <= 0.25)),
        "within_50pct": float(np.mean(ape <= 0.50)),
        "p90_ape": float(np.quantile(ape, 0.90)),
        "p95_ape": float(np.quantile(ape, 0.95)),
        "p99_ape": float(np.quantile(ape, 0.99)),
    }


def markdown_report(payload: dict) -> str:
    lines = [
        f"# {payload['experiment_id']} — HistGradientBoostingRegressor",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(payload["config"], indent=2),
        "```",
        "",
        "## Metrics",
        "",
        "| Split | n | median_ape | mae_yen | r2_price | r2_log10 | within_25pct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, vals in payload["metrics"].items():
        lines.append(
            f"| {split} | {vals['n']:,} | {vals['median_ape']:.6f} | {vals['mae_yen']:,.0f} | {vals['r2_price']:.6f} | {vals['r2_log10']:.6f} | {vals['within_25pct']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate a tree-based sklearn baseline.")
    parser.add_argument("--experiment-id", default="EXP-002")
    parser.add_argument("--max-train-rows", type=int, default=500000, help="Sample training rows for speed; 0 means all rows.")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    max_train_rows = None if args.max_train_rows == 0 else args.max_train_rows

    train_df, x_train, y_train = load_xy("train", max_train_rows, args.random_seed)
    _, x_dev, y_dev = load_xy("dev")
    _, x_test, y_test = load_xy("test")

    y_train_log = np.log10(y_train + 1)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=args.random_seed,
        verbose=0,
    )
    model.fit(x_train, y_train_log)

    payload = {
        "experiment_id": args.experiment_id,
        "model": "sklearn.ensemble.HistGradientBoostingRegressor",
        "feature_order": PARAMS["feature_order"],
        "config": {
            "train_rows": int(len(train_df)),
            "max_train_rows": args.max_train_rows,
            "max_iter": args.max_iter,
            "actual_iter": int(model.n_iter_),
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "target": "log10(TotalTransactionValue + 1)",
        },
        "metrics": {
            "dev": metrics(y_dev, model.predict(x_dev)),
            "test": metrics(y_test, model.predict(x_test)),
        },
    }

    results_dir = ROOT / "Experiment/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{args.experiment_id}_hist_gradient_boosting_metrics.json"
    report_path = results_dir / f"{args.experiment_id}_hist_gradient_boosting_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(markdown_report(payload), encoding="utf-8")

    if args.save_model:
        model_path = results_dir / f"{args.experiment_id}_hist_gradient_boosting.joblib"
        joblib.dump(model, model_path)
        payload["saved_model"] = str(model_path.relative_to(ROOT))
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
