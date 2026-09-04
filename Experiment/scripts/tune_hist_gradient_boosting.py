"""Tune HistGradientBoostingRegressor on the dev split.

Runs a small fixed grid using the existing 32 production features. Selection is
based on dev median APE; only the best dev configuration is evaluated on test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from app.preprocess import PARAMS, build_feature_vector

SPLIT_PATHS = {
    "train": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_training_data.csv",
    "dev": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_dev_data.csv",
    "test": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_test_data.csv",
}


def load_xy(split: str, max_rows: int | None = None, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(SPLIT_PATHS[split])
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_seed).sort_index()
    x = np.asarray([build_feature_vector(row) for row in df.to_dict("records")], dtype=np.float32)
    y = df["TotalTransactionValue"].to_numpy(dtype=float)
    return x, y


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


def fit_model(config: dict, x_train: np.ndarray, y_train: np.ndarray) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=config["learning_rate"],
        max_iter=config["max_iter"],
        max_leaf_nodes=config["max_leaf_nodes"],
        l2_regularization=config["l2_regularization"],
        min_samples_leaf=config["min_samples_leaf"],
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=config["random_seed"],
    )
    model.fit(x_train, np.log10(y_train + 1))
    return model


def markdown(payload: dict) -> str:
    lines = [
        f"# {payload['experiment_id']} — HGB tuning",
        "",
        "Selection metric: lowest dev `median_ape`. Only the best dev config was evaluated on test.",
        "",
        "## Best config",
        "",
        "```json",
        json.dumps(payload["best_config"], indent=2),
        "```",
        "",
        "## Best metrics",
        "",
        "| Split | median_ape | mae_yen | rmse_yen | r2_price | r2_log10 | within_25pct | p95_ape |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, vals in payload["best_metrics"].items():
        lines.append(
            f"| {split} | {vals['median_ape']:.6f} | {vals['mae_yen']:,.0f} | {vals['rmse_yen']:,.0f} | {vals['r2_price']:.6f} | {vals['r2_log10']:.6f} | {vals['within_25pct']:.6f} | {vals['p95_ape']:.6f} |"
        )
    lines += ["", "## Dev sweep", "", "| rank | median_ape | mae_yen | r2_log10 | actual_iter | config |", "|---:|---:|---:|---:|---:|---|"]
    for i, row in enumerate(payload["dev_results"], 1):
        cfg = {k: row["config"][k] for k in ["learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization", "min_samples_leaf"]}
        lines.append(f"| {i} | {row['metrics']['median_ape']:.6f} | {row['metrics']['mae_yen']:,.0f} | {row['metrics']['r2_log10']:.6f} | {row['actual_iter']} | `{json.dumps(cfg)}` |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune HGB on dev and test only best config.")
    parser.add_argument("--experiment-id", default="EXP-005")
    parser.add_argument("--max-train-rows", type=int, default=500000, help="0 means all training rows.")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(ROOT)
    max_train_rows = None if args.max_train_rows == 0 else args.max_train_rows
    x_train, y_train = load_xy("train", max_train_rows, args.random_seed)
    x_dev, y_dev = load_xy("dev")
    x_test, y_test = load_xy("test")

    grid = []
    for learning_rate, max_iter, max_leaf_nodes, l2_regularization, min_samples_leaf in product(
        [0.03, 0.05, 0.08],
        [300],
        [31, 63],
        [0.0, 0.01],
        [20],
    ):
        grid.append({
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "max_leaf_nodes": max_leaf_nodes,
            "l2_regularization": l2_regularization,
            "min_samples_leaf": min_samples_leaf,
            "random_seed": args.random_seed,
        })

    dev_results = []
    for n, config in enumerate(grid, 1):
        model = fit_model(config, x_train, y_train)
        dev_metrics = metrics(y_dev, model.predict(x_dev))
        row = {"config": config, "actual_iter": int(model.n_iter_), "metrics": dev_metrics}
        dev_results.append(row)
        print(f"{n}/{len(grid)} dev median_ape={dev_metrics['median_ape']:.6f} config={config}", flush=True)

    dev_results.sort(key=lambda r: (r["metrics"]["median_ape"], r["metrics"]["mae_yen"]))
    best_config = dev_results[0]["config"]
    best_model = fit_model(best_config, x_train, y_train)

    payload = {
        "experiment_id": args.experiment_id,
        "model": "sklearn.ensemble.HistGradientBoostingRegressor",
        "feature_order": PARAMS["feature_order"],
        "selection_metric": "dev.median_ape",
        "train_rows": int(len(y_train)),
        "best_config": {**best_config, "actual_iter_refit": int(best_model.n_iter_)},
        "best_metrics": {
            "dev": metrics(y_dev, best_model.predict(x_dev)),
            "test": metrics(y_test, best_model.predict(x_test)),
        },
        "dev_results": dev_results,
    }

    results_dir = ROOT / "Experiment/results"
    json_path = results_dir / f"{args.experiment_id}_hgb_tuning_metrics.json"
    report_path = results_dir / f"{args.experiment_id}_hgb_tuning_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload["best_metrics"], indent=2))
    print(f"Wrote {json_path.relative_to(ROOT)} and {report_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
