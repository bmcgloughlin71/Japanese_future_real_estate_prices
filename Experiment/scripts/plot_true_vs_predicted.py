"""Create true-vs-predicted log-price plots for baseline and experiments.

This regenerates predictions for the saved experiment configurations and writes a
single multi-panel PNG under Experiment/results/.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

from app.preprocess import build_feature_vector

SPLIT_PATHS = {
    "train": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_training_data.csv",
    "dev": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_dev_data.csv",
    "test": ROOT / "Data_processing/Split_Data_Sets/All_prefecture_Housing_with_migration_location_and_pop_data_test_data.csv",
}
MODEL_PATH = ROOT / "Regression_Analysis/Model_and_Weights/Japanese_Housing_Price_Model.keras"


def load_xy(split: str, max_rows: int | None = None, random_seed: int = 42):
    df = pd.read_csv(SPLIT_PATHS[split])
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_seed).sort_index()
    x = np.asarray([build_feature_vector(row) for row in df.to_dict("records")], dtype=np.float32)
    y = df["TotalTransactionValue"].to_numpy(dtype=float)
    return x, y


def train_hgb(x_train, y_train, *, learning_rate, max_leaf_nodes, max_iter=300, random_seed=42):
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
    model.fit(x_train, np.log10(y_train + 1))
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot true vs predicted log prices for experiments.")
    parser.add_argument("--output", default="Experiment/results/true_vs_predicted_all_experiments.png")
    parser.add_argument("--sample", type=int, default=80000, help="Points sampled per panel for scatter plotting; 0 means all.")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    os.chdir(ROOT)
    rng = np.random.default_rng(args.random_seed)

    print("Loading dev/test features...")
    x_dev, y_dev = load_xy("dev")
    x_test, y_test = load_xy("test")
    y_test_log = np.log10(y_test + 1)

    print("Predicting Keras baseline/calibrations...")
    keras_model = tf.keras.models.load_model(MODEL_PATH)
    dev_keras_log = keras_model.predict(x_dev, batch_size=8192, verbose=0).reshape(-1)
    test_keras_log = keras_model.predict(x_test, batch_size=8192, verbose=0).reshape(-1)

    affine = LinearRegression().fit(dev_keras_log.reshape(-1, 1), np.log10(y_dev + 1))
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(dev_keras_log, np.log10(y_dev + 1))

    predictions = [
        ("Baseline Keras", test_keras_log),
        ("EXP-004 affine calibration", affine.predict(test_keras_log.reshape(-1, 1))),
        ("EXP-004 isotonic calibration", isotonic.predict(test_keras_log)),
    ]

    hgb_runs = [
        ("EXP-002 HGB 500k", 500000, 0.05, 31),
        ("EXP-003 HGB all rows", None, 0.05, 31),
        ("EXP-005 tuned HGB 500k", 500000, 0.08, 63),
        ("EXP-006 tuned HGB all rows", None, 0.08, 63),
    ]
    for name, max_rows, lr, leaves in hgb_runs:
        print(f"Training/predicting {name}...")
        x_train, y_train = load_xy("train", max_rows=max_rows, random_seed=args.random_seed)
        model = train_hgb(x_train, y_train, learning_rate=lr, max_leaf_nodes=leaves, random_seed=args.random_seed)
        predictions.append((name, model.predict(x_test)))
        del x_train, y_train, model

    if args.sample and args.sample < len(y_test_log):
        idx = rng.choice(len(y_test_log), size=args.sample, replace=False)
    else:
        idx = np.arange(len(y_test_log))

    n = len(predictions)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 4.8 * rows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    lo = min(y_test_log.min(), min(np.min(pred) for _, pred in predictions))
    hi = max(y_test_log.max(), max(np.max(pred) for _, pred in predictions))
    pad = 0.05 * (hi - lo)
    lo -= pad
    hi += pad

    for ax, (name, pred_log) in zip(axes, predictions):
        ax.scatter(y_test_log[idx], pred_log[idx], s=2, alpha=0.12, linewidths=0)
        ax.plot([lo, hi], [lo, hi], color="crimson", linestyle="--", linewidth=1.2, label="perfect")
        err = np.median(np.abs((np.power(10, pred_log) - 1 - y_test) / y_test))
        ax.set_title(f"{name}\nmedian APE={err:.3f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
        ax.set_xlabel("True log10(price + 1)")
        ax.set_ylabel("Predicted log10(price + 1)")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("True vs predicted Japanese real-estate prices — test split", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print(f"Wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
