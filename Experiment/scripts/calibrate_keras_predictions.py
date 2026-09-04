"""Calibrate existing Keras log-price predictions using the dev split.

Fits simple post-processing maps from predicted log10(price+1) to true
log10(price+1), then evaluates on test. This does not retrain the production
model or change app code.

Example:
    python3 Experiment/scripts/calibrate_keras_predictions.py --experiment-id EXP-004
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

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from app.preprocess import build_feature_vector


SPLIT_PATHS = {
    "dev": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_dev_data.csv",
    "test": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_test_data.csv",
}
MODEL_PATH = ROOT / "Regression_Analysis/Model_and_Weights/Japanese_Housing_Price_Model.keras"


def load_xy(split: str):
    df = pd.read_csv(SPLIT_PATHS[split])
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


def report(payload: dict) -> str:
    lines = [
        f"# {payload['experiment_id']} — Keras prediction calibration",
        "",
        "## Metrics",
        "",
        "| Method | Split | median_ape | mae_yen | rmse_yen | r2_price | r2_log10 | within_25pct | p95_ape |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, splits in payload["metrics"].items():
        for split, vals in splits.items():
            lines.append(
                f"| {method} | {split} | {vals['median_ape']:.6f} | {vals['mae_yen']:,.0f} | {vals['rmse_yen']:,.0f} | {vals['r2_price']:.6f} | {vals['r2_log10']:.6f} | {vals['within_25pct']:.6f} | {vals['p95_ape']:.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate fixed Keras model predictions using dev split.")
    parser.add_argument("--experiment-id", default="EXP-004")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--save-calibrators", action="store_true")
    args = parser.parse_args()

    os.chdir(ROOT)
    _, x_dev, y_dev = load_xy("dev")
    _, x_test, y_test = load_xy("test")
    true_dev_log = np.log10(y_dev + 1)

    model = tf.keras.models.load_model(MODEL_PATH)
    dev_pred_log = model.predict(x_dev, batch_size=args.batch_size, verbose=0).reshape(-1)
    test_pred_log = model.predict(x_test, batch_size=args.batch_size, verbose=0).reshape(-1)

    affine = LinearRegression()
    affine.fit(dev_pred_log.reshape(-1, 1), true_dev_log)

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(dev_pred_log, true_dev_log)

    calibrated = {
        "uncalibrated": {
            "dev": dev_pred_log,
            "test": test_pred_log,
        },
        "affine": {
            "dev": affine.predict(dev_pred_log.reshape(-1, 1)),
            "test": affine.predict(test_pred_log.reshape(-1, 1)),
        },
        "isotonic": {
            "dev": isotonic.predict(dev_pred_log),
            "test": isotonic.predict(test_pred_log),
        },
    }

    payload = {
        "experiment_id": args.experiment_id,
        "model_path": str(MODEL_PATH.relative_to(ROOT)),
        "calibration_train_split": "dev",
        "calibrators": {
            "affine": {
                "slope": float(affine.coef_[0]),
                "intercept": float(affine.intercept_),
            },
            "isotonic": {
                "n_thresholds": int(len(isotonic.X_thresholds_)),
            },
        },
        "metrics": {},
    }

    for method, split_preds in calibrated.items():
        payload["metrics"][method] = {
            "dev": metrics(y_dev, split_preds["dev"]),
            "test": metrics(y_test, split_preds["test"]),
        }

    results_dir = ROOT / "Experiment/results"
    json_path = results_dir / f"{args.experiment_id}_keras_calibration_metrics.json"
    report_path = results_dir / f"{args.experiment_id}_keras_calibration_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(report(payload), encoding="utf-8")

    if args.save_calibrators:
        joblib.dump(affine, results_dir / f"{args.experiment_id}_affine_calibrator.joblib")
        joblib.dump(isotonic, results_dir / f"{args.experiment_id}_isotonic_calibrator.joblib")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
