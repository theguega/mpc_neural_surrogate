import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from tqdm import tqdm


def load_data(filepath):
    episodes = []
    try:
        with h5py.File(filepath, "r") as f:
            keys = np.array(sorted(f["episodes"].keys()))
            for ep in keys:
                grp = f["episodes"][ep]
                states = grp["states"][:]
                targets = grp["targets"][:]
                actions = grp["actions"][:]
                episodes.append((states, targets, actions))
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None, None

    X_list = []
    Y_list = []

    for states, targets, actions in episodes:
        combined_input = np.hstack([states, targets])
        X_list.append(combined_input)
        Y_list.append(actions)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds)


def sub_dataset(X, Y, ratio=0.5, random_state_index=0):
    # slice the first N% of data
    limit = int(len(X) * ratio)
    X_sub = X[:limit]
    Y_sub = Y[:limit]

    X_train, X_test, y_train, y_test = train_test_split(X_sub, Y_sub, test_size=0.2, random_state=42 + random_state_index, shuffle=True)
    train_len = int(len(X_train))
    return X_train, X_test, y_train, y_test, train_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data/robot_mpc_dataset.h5", help="Path to input file")
    parser.add_argument("--output_dir", type=str, default="results/dataset_scale/", help="Directory to save results")
    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")
    X, Y = load_data(args.input_file)
    if X is None:
        return

    print(f"Data loaded. X shape: {X.shape}, Y shape: {Y.shape}")

    n_trials = 5
    # ratios = [0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
    ratios = [0.05, 0.1, 0.2, 0.5, 0.75]

    results = {"MLP": {"means": [], "stds": [], "sizes": []}, "SVR": {"means": [], "stds": [], "sizes": []}}

    print(f"\nRunning evaluation with {n_trials} trials per ratio...")

    for ratio in ratios:
        scores_mlp = []
        scores_svr = []

        train_len = 0
        for i in tqdm(range(n_trials), desc=f"Ratio {ratio}"):
            X_train, X_test, y_train, y_test, train_len = sub_dataset(X, Y, ratio=ratio, random_state_index=i)

            start_time = time.time()
            mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=i)
            mse_mlp = evaluate_model(mlp, X_train, X_test, y_train, y_test)
            scores_mlp.append(mse_mlp)
            train_time = time.time() - start_time
            tqdm.write(f"Training MLP with {train_len} samples took {train_time:.2f} seconds, score {mse_mlp:.2f}")

            start_time = time.time()
            svr = MultiOutputRegressor(SVR())
            mse_svr = evaluate_model(svr, X_train, X_test, y_train, y_test)
            scores_svr.append(mse_svr)
            train_time = time.time() - start_time
            tqdm.write(f"Training SVR with {train_len} samples took {train_time:.2f} seconds, score {mse_svr:.2f}")

        results["MLP"]["sizes"].append(train_len)
        results["MLP"]["means"].append(np.mean(scores_mlp))
        results["MLP"]["stds"].append(np.std(scores_mlp))

        # fill NaN if skipped
        results["SVR"]["sizes"].append(train_len)
        if scores_svr:
            results["SVR"]["means"].append(np.mean(scores_svr))
            results["SVR"]["stds"].append(np.std(scores_svr))
        else:
            results["SVR"]["means"].append(np.nan)
            results["SVR"]["stds"].append(np.nan)

    plt.figure(figsize=(10, 6))

    for name, data in results.items():
        sizes = np.array(data["sizes"])
        means = np.array(data["means"])
        stds = np.array(data["stds"])

        mask = ~np.isnan(means)
        if not np.any(mask):
            continue

        plt.plot(sizes[mask], means[mask], "o-", label=name)
        plt.fill_between(sizes[mask], means[mask] - stds[mask], means[mask] + stds[mask], alpha=0.2)

    plt.title("Model Scaling: MSE vs Dataset Size")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True)
    output = os.path.join(args.output_dir, "dataset_scale_plot.png")
    plt.savefig(output)
    plt.show()


if __name__ == "__main__":
    main()
