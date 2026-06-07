# utils/plot_metrics.py
"""
Offline accuracy plot generator.
Reads per-client CSV metrics and saves a chart to results/.
Run: python plot_metrics.py
"""
import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

files       = sorted(glob.glob("data/metrics_client_*.csv"))
data_frames = []

for f in files:
    try:
        df  = pd.read_csv(f)
        cid = os.path.splitext(os.path.basename(f))[0].split("_")[-1]
        df["client"]      = cid
        df["round_index"] = range(1, len(df) + 1)
        data_frames.append(df)
    except Exception as e:
        print(f"[WARN] Could not read {f}: {e}")

if not data_frames:
    print("No metrics files found (data/metrics_client_*.csv). "
          "Run clients with evaluate() first.")
else:
    combined = pd.concat(data_frames, ignore_index=True)

    # Check required columns exist
    if "accuracy" not in combined.columns:
        print("[ERROR] 'accuracy' column not found in metrics CSV files.")
    else:
        pivot = combined.pivot_table(index="round_index", columns="client", values="accuracy")

        fig, ax = plt.subplots(figsize=(9, 5))
        for col in pivot.columns:
            ax.plot(pivot.index, pivot[col], marker="o", linewidth=2, label=f"Client {col}")

        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Client Evaluation Accuracy per FL Round", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()

        out_path = "results/accuracy_per_client.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")

        # BUG FIX #14: Show plot and close to free memory
        plt.show()
        plt.close(fig)
