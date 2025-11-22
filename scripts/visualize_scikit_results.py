import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(results_dir):
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    all_data = []

    for file in json_files:
        with open(file, "r") as f:
            data = json.load(f)
            for entry in data:
                all_data.append(entry)

    return pd.DataFrame(all_data)


def plot_metrics(df, output_dir):
    sns.set_theme(style="whitegrid")

    # MSE vs MAE loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df, x="model", y="MSE", ax=axes[0], capsize=0.1, errorbar=("ci", 95))
    axes[0].set_title("Mean Squared Error (MSE)")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=df, x="model", y="MAE", ax=axes[1], capsize=0.1, errorbar=("ci", 95))
    axes[1].set_title("Mean Absolute Error (MAE)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_mae_comparison.png"))
    plt.close()

    # Direction Accuracy and Explained Variance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(data=df, x="model", y="Direction Accuracy", ax=axes[0], capsize=0.1, errorbar=("ci", 95))
    axes[0].set_title("Direction Accuracy (Cosine Similarity)")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(data=df, x="model", y="Explained Variance", ax=axes[1], capsize=0.1, errorbar=("ci", 95))
    axes[1].set_title("Explained Variance Score")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_variance_comparison.png"))
    plt.close()

    # per-torque MSE
    torque_data = []
    for _, row in df.iterrows():
        mse_torques = row["MSE per Torque"]
        for i, val in enumerate(mse_torques):
            torque_data.append({"model": row["model"], "Torque": f"Torque {i + 1}", "MSE": val})

    df_torque = pd.DataFrame(torque_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_torque, x="model", y="MSE", hue="Torque", capsize=0.1)
    plt.title("MSE per Torque Dimension")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_per_torque.png"))
    plt.close()


def main():
    if not os.path.exists("results"):
        print("Error: Results directory does not exist.")
        return

    df = load_results("results")

    if df.empty:
        print("No data found.")
        return

    print(f"Found {len(df)} entries from models: {df['model'].unique()}")

    print("Generating plots...")

    plot_metrics(df, "results/plots/")
    print("Done.")


if __name__ == "__main__":
    main()
