"""Evaluate the current Keras housing price model on an existing split.

Example:
    python3 Experiment/scripts/evaluate_model.py \
        --split test \
        --output Experiment/results/baseline_test_metrics.json \
        --markdown Experiment/results/baseline_test_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

from app.preprocess import build_feature_vector


SPLIT_PATHS = {
    "train": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_training_data.csv",
    "dev": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_dev_data.csv",
    "test": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_test_data.csv",
}
MODEL_PATH = ROOT / "Regression_Analysis/Model_and_Weights/Japanese_Housing_Price_Model.keras"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
    abs_err = np.abs(y_pred - y_true)
    ape = abs_err / y_true
    y_log = np.log10(y_true + 1)
    return {
        "n": int(len(y_true)),
        "mean_actual": float(np.mean(y_true)),
        "median_actual": float(np.median(y_true)),
        "mae_yen": float(np.mean(abs_err)),
        "median_abs_error_yen": float(np.median(abs_err)),
        "rmse_yen": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "mape": float(np.mean(ape)),
        "median_ape": float(np.median(ape)),
        "r2_price": float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)),
        "r2_log10": float(1 - np.sum((y_log - pred_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)),
        "mae_log10": float(np.mean(np.abs(y_log - pred_log))),
        "within_10pct": float(np.mean(ape <= 0.10)),
        "within_25pct": float(np.mean(ape <= 0.25)),
        "within_50pct": float(np.mean(ape <= 0.50)),
        "within_100pct": float(np.mean(ape <= 1.00)),
        "p90_ape": float(np.quantile(ape, 0.90)),
        "p95_ape": float(np.quantile(ape, 0.95)),
        "p99_ape": float(np.quantile(ape, 0.99)),
    }


def _markdown_report(split: str, metrics: Dict[str, float], by_type: Dict[str, Dict[str, float]]) -> str:
    lines = [
        f"# Model evaluation — {split} split",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"| `{key}` | {value:,.6g} |")
        else:
            lines.append(f"| `{key}` | {value} |")
    lines += ["", "## By property type", "", "| Type | n | median_ape | mae_yen | within_25pct |", "|---|---:|---:|---:|---:|"]
    for type_name, vals in by_type.items():
        lines.append(
            f"| {type_name} | {int(vals['n']):,} | {vals['median_ape']:.4f} | {vals['mae_yen']:,.0f} | {vals['within_25pct']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the current housing price model.")
    parser.add_argument("--split", choices=sorted(SPLIT_PATHS), default="test")
    parser.add_argument("--output", help="Optional JSON metrics output path.")
    parser.add_argument("--markdown", help="Optional Markdown report output path.")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()

    os.chdir(ROOT)
    df = pd.read_csv(SPLIT_PATHS[args.split])
    x = np.asarray([build_feature_vector(row) for row in df.to_dict("records")], dtype=np.float32)
    y = df["TotalTransactionValue"].to_numpy(dtype=float)

    model = tf.keras.models.load_model(MODEL_PATH)
    pred_log = model.predict(x, batch_size=args.batch_size, verbose=0).reshape(-1)
    pred = np.power(10, pred_log) - 1

    overall = _metrics(y, pred, pred_log)
    by_type = {}
    for type_name, group in df.groupby("Type"):
        idx = group.index.to_numpy()
        by_type[type_name] = _metrics(y[idx], pred[idx], pred_log[idx])

    payload = {"split": args.split, "model_path": str(MODEL_PATH.relative_to(ROOT)), "overall": overall, "by_type": by_type}
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.markdown:
        out = ROOT / args.markdown
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown_report(args.split, overall, by_type), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
