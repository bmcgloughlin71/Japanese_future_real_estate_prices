"""Two-stage cheap-property experiment.

1. Train a classifier for ultra-cheap properties: price <= threshold yen.
2. Train a specialist regressor on only those cheap training rows.
3. Train the best current general HGB regressor.
4. Select a classifier probability threshold on dev.
5. Evaluate classifier and combined two-stage predictions on test.
6. Write JSON/Markdown reports and diagnostic plots.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from app.preprocess import PARAMS, build_feature_vector

SPLIT_PATHS = {
    "train": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_training_data.csv",
    "dev": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_dev_data.csv",
    "test": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_test_data.csv",
}


def load_xy(split: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(SPLIT_PATHS[split])
    x = np.asarray([build_feature_vector(row) for row in df.to_dict("records")], dtype=np.float32)
    y = df["TotalTransactionValue"].to_numpy(dtype=float)
    return df, x, y


def fit_regressor(x: np.ndarray, y: np.ndarray, *, learning_rate: float, max_leaf_nodes: int, max_iter: int, random_seed: int):
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        l2_regularization=0.0,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=random_seed,
    )
    model.fit(x, np.log10(y + 1))
    return model


def regression_metrics(y_true: np.ndarray, pred_log: np.ndarray) -> Dict[str, float]:
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


def classifier_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float | int]:
    pred = prob >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    return {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "average_precision": float(average_precision_score(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def choose_threshold(y_dev: np.ndarray, prob_dev: np.ndarray, general_log: np.ndarray, cheap_log: np.ndarray) -> Dict[str, float]:
    y_class = y_dev <= choose_threshold.cheap_price_threshold
    precision, recall, pr_thresholds = precision_recall_curve(y_class, prob_dev)
    candidates = sorted(set([0.05, 0.10, 0.20, 0.30, 0.50, *pr_thresholds[:: max(1, len(pr_thresholds) // 200)].tolist()]))
    rows = []
    for threshold in candidates:
        use_cheap = prob_dev >= threshold
        combined_log = np.where(use_cheap, cheap_log, general_log)
        reg = regression_metrics(y_dev, combined_log)
        clf = classifier_metrics(y_class, prob_dev, threshold)
        cheap_mask = y_dev <= choose_threshold.cheap_price_threshold
        cheap_reg = regression_metrics(y_dev[cheap_mask], combined_log[cheap_mask])
        rows.append({
            "threshold": float(threshold),
            "median_ape": reg["median_ape"],
            "mae_yen": reg["mae_yen"],
            "within_25pct": reg["within_25pct"],
            "cheap_median_ape": cheap_reg["median_ape"],
            "cheap_within_25pct": cheap_reg["within_25pct"],
            "precision": clf["precision"],
            "recall": clf["recall"],
            "f1": clf["f1"],
        })
    # Prioritize overall median APE, then cheap-tail median APE, then overall MAE.
    rows.sort(key=lambda r: (r["median_ape"], r["cheap_median_ape"], r["mae_yen"]))
    return {"best": rows[0], "candidates": rows[:25]}


def make_plots(y_test, general_log, combined_log, prob_test, threshold, out_path: Path, sample: int, random_seed: int):
    rng = np.random.default_rng(random_seed)
    true_log = np.log10(y_test + 1)
    if sample and sample < len(y_test):
        idx = rng.choice(len(y_test), size=sample, replace=False)
    else:
        idx = np.arange(len(y_test))

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.reshape(-1)
    panels = [
        (axes[0], "EXP-006 general HGB", general_log),
        (axes[1], "EXP-007 two-stage combined", combined_log),
    ]
    lo = min(true_log.min(), general_log.min(), combined_log.min())
    hi = max(true_log.max(), general_log.max(), combined_log.max())
    for ax, title, pred_log in panels:
        ape = np.abs((np.power(10, pred_log) - 1 - y_test) / y_test)
        ax.scatter(true_log[idx], pred_log[idx], s=2, alpha=0.12, linewidths=0)
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_title(f"{title}\nmedian APE={np.median(ape):.3f}")
        ax.set_xlabel("True log10(price + 1)")
        ax.set_ylabel("Predicted log10(price + 1)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)

    cheap_mask = y_test <= choose_threshold.cheap_price_threshold
    axes[2].scatter(true_log[cheap_mask], general_log[cheap_mask], s=8, alpha=0.25, label="general")
    axes[2].scatter(true_log[cheap_mask], combined_log[cheap_mask], s=8, alpha=0.25, label="two-stage")
    axes[2].plot([true_log[cheap_mask].min(), true_log[cheap_mask].max()], [true_log[cheap_mask].min(), true_log[cheap_mask].max()], "r--", lw=1.2)
    axes[2].set_title("Cheap true-price tail only (<= ¥1m)")
    axes[2].set_xlabel("True log10(price + 1)")
    axes[2].set_ylabel("Predicted log10(price + 1)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    axes[3].hist(prob_test[y_test > choose_threshold.cheap_price_threshold], bins=60, alpha=0.65, label="> ¥1m", density=True)
    axes[3].hist(prob_test[cheap_mask], bins=60, alpha=0.65, label="<= ¥1m", density=True)
    axes[3].axvline(threshold, color="red", linestyle="--", label=f"threshold={threshold:.3f}")
    axes[3].set_title("Classifier cheap-property probabilities")
    axes[3].set_xlabel("Predicted probability price <= ¥1m")
    axes[3].set_ylabel("Density")
    axes[3].legend()
    axes[3].grid(alpha=0.25)

    fig.suptitle("EXP-007 cheap-property classifier + specialist regressor", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=180)


def markdown(payload: dict) -> str:
    lines = [
        f"# {payload['experiment_id']} — cheap-property two-stage model",
        "",
        "## Classifier metrics",
        "",
        "| Split | threshold | prevalence | precision | recall | f1 | ROC AUC | average precision | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, vals in payload["classifier_metrics"].items():
        lines.append(
            f"| {split} | {vals['threshold']:.6f} | {vals['prevalence']:.4f} | {vals['precision']:.4f} | {vals['recall']:.4f} | {vals['f1']:.4f} | {vals['roc_auc']:.4f} | {vals['average_precision']:.4f} | {vals['tp']} | {vals['fp']} | {vals['fn']} | {vals['tn']} |"
        )
    lines += ["", "## Regression metrics", "", "| Model | Split | median_ape | mae_yen | rmse_yen | r2_price | r2_log10 | within_25pct | p95_ape |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for model_name, splits in payload["regression_metrics"].items():
        for split, vals in splits.items():
            lines.append(
                f"| {model_name} | {split} | {vals['median_ape']:.6f} | {vals['mae_yen']:,.0f} | {vals['rmse_yen']:,.0f} | {vals['r2_price']:.6f} | {vals['r2_log10']:.6f} | {vals['within_25pct']:.6f} | {vals['p95_ape']:.6f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Cheap-property classifier + specialist regressor experiment.")
    parser.add_argument("--experiment-id", default="EXP-007")
    parser.add_argument("--cheap-threshold-yen", type=float, default=1_000_000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--plot-sample", type=int, default=80000)
    args = parser.parse_args()

    choose_threshold.cheap_price_threshold = args.cheap_threshold_yen
    os.chdir(ROOT)

    print("Loading features...")
    train_df, x_train, y_train = load_xy("train")
    _, x_dev, y_dev = load_xy("dev")
    _, x_test, y_test = load_xy("test")

    y_train_class = y_train <= args.cheap_threshold_yen
    y_dev_class = y_dev <= args.cheap_threshold_yen
    y_test_class = y_test <= args.cheap_threshold_yen

    print("Training cheap-property classifier...")
    classifier = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.08,
        max_iter=300,
        max_leaf_nodes=63,
        l2_regularization=0.0,
        min_samples_leaf=20,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=args.random_seed,
        class_weight="balanced",
    )
    classifier.fit(x_train, y_train_class)
    dev_prob = classifier.predict_proba(x_dev)[:, 1]
    test_prob = classifier.predict_proba(x_test)[:, 1]

    print("Training general tuned HGB regressor...")
    general = fit_regressor(x_train, y_train, learning_rate=0.08, max_leaf_nodes=63, max_iter=300, random_seed=args.random_seed)
    dev_general_log = general.predict(x_dev)
    test_general_log = general.predict(x_test)

    print("Training cheap specialist regressor...")
    cheap_mask_train = y_train <= args.cheap_threshold_yen
    cheap = fit_regressor(
        x_train[cheap_mask_train],
        y_train[cheap_mask_train],
        learning_rate=0.05,
        max_leaf_nodes=31,
        max_iter=300,
        random_seed=args.random_seed,
    )
    dev_cheap_log = cheap.predict(x_dev)
    test_cheap_log = cheap.predict(x_test)

    threshold_result = choose_threshold(y_dev, dev_prob, dev_general_log, dev_cheap_log)
    threshold = threshold_result["best"]["threshold"]
    print(f"Selected threshold={threshold:.6f}")

    dev_combined_log = np.where(dev_prob >= threshold, dev_cheap_log, dev_general_log)
    test_combined_log = np.where(test_prob >= threshold, test_cheap_log, test_general_log)

    cheap_dev_mask = y_dev <= args.cheap_threshold_yen
    cheap_test_mask = y_test <= args.cheap_threshold_yen

    payload = {
        "experiment_id": args.experiment_id,
        "cheap_threshold_yen": args.cheap_threshold_yen,
        "feature_order": PARAMS["feature_order"],
        "train_counts": {
            "rows": int(len(y_train)),
            "cheap_rows": int(np.sum(y_train_class)),
            "cheap_prevalence": float(np.mean(y_train_class)),
        },
        "classifier_config": {
            "model": "HistGradientBoostingClassifier",
            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "class_weight": "balanced",
            "actual_iter": int(classifier.n_iter_),
        },
        "general_regressor_config": {
            "model": "HistGradientBoostingRegressor",
            "learning_rate": 0.08,
            "max_iter": 300,
            "max_leaf_nodes": 63,
            "actual_iter": int(general.n_iter_),
        },
        "cheap_regressor_config": {
            "model": "HistGradientBoostingRegressor",
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "actual_iter": int(cheap.n_iter_),
        },
        "threshold_selection": threshold_result,
        "classifier_metrics": {
            "dev": classifier_metrics(y_dev_class, dev_prob, threshold),
            "test": classifier_metrics(y_test_class, test_prob, threshold),
        },
        "regression_metrics": {
            "general_hgb": {
                "dev": regression_metrics(y_dev, dev_general_log),
                "test": regression_metrics(y_test, test_general_log),
            },
            "two_stage_combined": {
                "dev": regression_metrics(y_dev, dev_combined_log),
                "test": regression_metrics(y_test, test_combined_log),
            },
            "general_hgb_on_true_cheap_only": {
                "dev": regression_metrics(y_dev[cheap_dev_mask], dev_general_log[cheap_dev_mask]),
                "test": regression_metrics(y_test[cheap_test_mask], test_general_log[cheap_test_mask]),
            },
            "two_stage_on_true_cheap_only": {
                "dev": regression_metrics(y_dev[cheap_dev_mask], dev_combined_log[cheap_dev_mask]),
                "test": regression_metrics(y_test[cheap_test_mask], test_combined_log[cheap_test_mask]),
            },
        },
    }

    results_dir = ROOT / "Experiment/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    json_path = results_dir / f"{args.experiment_id}_cheap_two_stage_metrics.json"
    report_path = results_dir / f"{args.experiment_id}_cheap_two_stage_report.md"
    plot_path = results_dir / f"{args.experiment_id}_cheap_two_stage_plot.png"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_path.write_text(markdown(payload), encoding="utf-8")
    make_plots(y_test, test_general_log, test_combined_log, test_prob, threshold, plot_path, args.plot_sample, args.random_seed)

    print(json.dumps({
        "classifier_test": payload["classifier_metrics"]["test"],
        "general_test": payload["regression_metrics"]["general_hgb"]["test"],
        "combined_test": payload["regression_metrics"]["two_stage_combined"]["test"],
        "general_cheap_test": payload["regression_metrics"]["general_hgb_on_true_cheap_only"]["test"],
        "combined_cheap_test": payload["regression_metrics"]["two_stage_on_true_cheap_only"]["test"],
    }, indent=2))
    print(f"Wrote {json_path.relative_to(ROOT)}, {report_path.relative_to(ROOT)}, {plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
