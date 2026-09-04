"""Threshold sweep for the cheap-property two-stage model.

Re-trains the EXP-007 components, then evaluates a fixed list of classifier
thresholds to show the trade-off between cheap-tail recall and overall accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from cheap_property_two_stage import classifier_metrics, fit_regressor, load_xy, regression_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep cheap-classifier thresholds.")
    parser.add_argument("--experiment-id", default="EXP-008")
    parser.add_argument("--cheap-threshold-yen", type=float, default=1_000_000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(ROOT)
    print("Loading features...")
    _, x_train, y_train = load_xy("train")
    _, x_dev, y_dev = load_xy("dev")
    _, x_test, y_test = load_xy("test")

    y_train_class = y_train <= args.cheap_threshold_yen
    y_dev_class = y_dev <= args.cheap_threshold_yen
    y_test_class = y_test <= args.cheap_threshold_yen

    print("Training classifier...")
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

    print("Training regressors...")
    general = fit_regressor(x_train, y_train, learning_rate=0.08, max_leaf_nodes=63, max_iter=300, random_seed=args.random_seed)
    cheap_mask_train = y_train <= args.cheap_threshold_yen
    cheap = fit_regressor(x_train[cheap_mask_train], y_train[cheap_mask_train], learning_rate=0.05, max_leaf_nodes=31, max_iter=300, random_seed=args.random_seed)

    dev_general_log = general.predict(x_dev)
    test_general_log = general.predict(x_test)
    dev_cheap_log = cheap.predict(x_dev)
    test_cheap_log = cheap.predict(x_test)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.9763705702305402, 0.99]
    rows = []
    for threshold in thresholds:
        for split, y, y_class, prob, general_log, cheap_log in [
            ("dev", y_dev, y_dev_class, dev_prob, dev_general_log, dev_cheap_log),
            ("test", y_test, y_test_class, test_prob, test_general_log, test_cheap_log),
        ]:
            combined = np.where(prob >= threshold, cheap_log, general_log)
            overall = regression_metrics(y, combined)
            cheap_true = y <= args.cheap_threshold_yen
            cheap_metrics = regression_metrics(y[cheap_true], combined[cheap_true])
            normal_true = y > args.cheap_threshold_yen
            normal_metrics = regression_metrics(y[normal_true], combined[normal_true])
            clf = classifier_metrics(y_class, prob, threshold)
            rows.append({
                "threshold": threshold,
                "split": split,
                "precision": clf["precision"],
                "recall": clf["recall"],
                "f1": clf["f1"],
                "tp": clf["tp"],
                "fp": clf["fp"],
                "fn": clf["fn"],
                "tn": clf["tn"],
                "overall_median_ape": overall["median_ape"],
                "overall_mae_yen": overall["mae_yen"],
                "overall_within_25pct": overall["within_25pct"],
                "cheap_median_ape": cheap_metrics["median_ape"],
                "cheap_mae_yen": cheap_metrics["mae_yen"],
                "cheap_within_25pct": cheap_metrics["within_25pct"],
                "normal_median_ape": normal_metrics["median_ape"],
                "normal_mae_yen": normal_metrics["mae_yen"],
                "normal_within_25pct": normal_metrics["within_25pct"],
            })

    results_dir = ROOT / "Experiment/results"
    csv_path = results_dir / f"{args.experiment_id}_cheap_threshold_sweep.csv"
    json_path = results_dir / f"{args.experiment_id}_cheap_threshold_sweep.json"
    md_path = results_dir / f"{args.experiment_id}_cheap_threshold_sweep_report.md"
    plot_path = results_dir / f"{args.experiment_id}_cheap_threshold_sweep_plot.png"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"experiment_id": args.experiment_id, "rows": rows}, indent=2), encoding="utf-8")

    test_rows = [r for r in rows if r["split"] == "test"]
    lines = [
        f"# {args.experiment_id} — cheap-property threshold sweep",
        "",
        "All rows below are test-split results.",
        "",
        "| threshold | precision | recall | f1 | TP | FP | overall median APE | cheap median APE | normal median APE | overall within 25% |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in test_rows:
        lines.append(
            f"| {r['threshold']:.6f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['tp']} | {r['fp']} | {r['overall_median_ape']:.4f} | {r['cheap_median_ape']:.4f} | {r['normal_median_ape']:.4f} | {r['overall_within_25pct']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    xs = np.array([r["threshold"] for r in test_rows])
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.reshape(-1)
    axes[0].plot(xs, [r["recall"] for r in test_rows], marker="o", label="recall")
    axes[0].plot(xs, [r["precision"] for r in test_rows], marker="o", label="precision")
    axes[0].plot(xs, [r["f1"] for r in test_rows], marker="o", label="f1")
    axes[0].set_title("Classifier threshold metrics")
    axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].plot(xs, [r["overall_median_ape"] for r in test_rows], marker="o", label="overall")
    axes[1].plot(xs, [r["normal_median_ape"] for r in test_rows], marker="o", label="normal > ¥1m")
    axes[1].set_title("Overall/normal median APE")
    axes[1].legend(); axes[1].grid(alpha=.25)
    axes[2].plot(xs, [r["cheap_median_ape"] for r in test_rows], marker="o")
    axes[2].set_title("True cheap-tail median APE")
    axes[2].grid(alpha=.25)
    axes[3].plot(xs, [r["fp"] for r in test_rows], marker="o", label="false positives")
    axes[3].plot(xs, [r["tp"] for r in test_rows], marker="o", label="true positives")
    axes[3].set_title("Cheap classifier counts")
    axes[3].legend(); axes[3].grid(alpha=.25)
    for ax in axes:
        ax.set_xlabel("classifier probability threshold")
    fig.suptitle("Cheap-property threshold sweep", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(plot_path, dpi=170)

    print(md_path.read_text(encoding="utf-8"))
    print(f"Wrote {csv_path.relative_to(ROOT)}, {json_path.relative_to(ROOT)}, {md_path.relative_to(ROOT)}, {plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
