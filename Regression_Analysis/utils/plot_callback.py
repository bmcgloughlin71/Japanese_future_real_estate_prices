import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class PlotPredictionsCallback_log(tf.keras.callbacks.Callback):
    def __init__(self, dev_features, dev_target_log, cols, out_dir="epoch_plots_true_v_pred"):
        super().__init__()
        self.dev_features = dev_features
        self.dev_target_log = dev_target_log
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.feature_names = list(cols)[1:]

    def on_epoch_end(self, epoch, logs=None):
        preds_log = self.model.predict(self.dev_features).flatten()
        preds = np.power(10, preds_log) - 1
        true_vals = np.power(10, self.dev_target_log) - 1

        output = np.column_stack((self.dev_features, true_vals, preds))
        header = "#" + "\t".join(self.feature_names + ["real", "pred"])

        np.savetxt(
            f"{self.out_dir}/epoch_{epoch+1:03d}.txt",
            output,
            header=header,
            fmt="%.6f",
            delimiter="\t",
            comments=''
        )

        plt.figure(figsize=(6, 5))
        plt.scatter(true_vals, preds, alpha=0.3)
        plt.plot([true_vals.min(), true_vals.max()], [true_vals.min(), true_vals.max()], 'r--')
        plt.xlabel("True Price")
        plt.ylabel("Predicted Price")
        plt.title(f"True vs Predicted (Epoch {epoch+1})")
        plt.grid(True)
        plt.xscale("log")
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/epoch_{epoch+1:03d}.png")
        plt.close()
