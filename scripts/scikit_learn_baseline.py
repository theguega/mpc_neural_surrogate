import argparse
import json
import os
from datetime import datetime

import h5py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
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


def compute_direction_accuracy(y_true, y_pred):
    """
    Cosine similarity between predicted and true torque vectors
    Returns mean cosine similarity
    # y_true: (N, 3), y_pred: (N, 3)
    """

    norm_true = np.linalg.norm(y_true, axis=1, keepdims=True)
    norm_pred = np.linalg.norm(y_pred, axis=1, keepdims=True)

    norm_true[norm_true == 0] = 1e-8
    norm_pred[norm_pred == 0] = 1e-8

    y_true_norm = y_true / norm_true
    y_pred_norm = y_pred / norm_pred

    # dot product of normalized vectors is cosine similarity
    cos_sim = np.sum(y_true_norm * y_pred_norm, axis=1)
    return np.mean(cos_sim)


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # global Metrics
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    ev = explained_variance_score(y_test, preds)
    da = compute_direction_accuracy(y_test, preds)

    # per-torque metrics
    mse_per_torque = mean_squared_error(y_test, preds, multioutput="raw_values")
    mae_per_torque = mean_absolute_error(y_test, preds, multioutput="raw_values")

    return {
        "MSE": mse,
        "MAE": mae,
        "Explained Variance": ev,
        "Direction Accuracy": da,
        "MSE per Torque": mse_per_torque,
        "MAE per Torque": mae_per_torque,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run on a small subset of data for testing")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model to run (or 'all'). Options: 'Linear Regression', 'Random Forest', 'MLP Regressor', 'Gradient Boosting', 'KNN Regressor'",
    )
    parser.add_argument("--input_file", type=str, default="data/robot_mpc_dataset.h5", help="Path to input file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    print(f"Loading data from {args.input_file}...")
    X, Y = load_data(args.input_file)

    print(f"Data loaded. X shape: {X.shape}, Y shape: {Y.shape}")

    if args.test:
        print("TEST MODE: Using only 1000 samples.")
        X = X[:1000]
        Y = Y[:1000]

    models_def = {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=50, n_jobs=-1, max_depth=10),
        "MLP Regressor": lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500),
        "Gradient Boosting": lambda: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100)),
        "KNN Regressor": lambda: KNeighborsRegressor(n_neighbors=5),
    }

    if args.model != "all":
        if args.model in models_def:
            models_def = {args.model: models_def[args.model]}
        else:
            print(f"Error: Model '{args.model}' not found. Available: {list(models_def.keys())}")
            return

    n_trials = 5
    results_agg = {name: [] for name in models_def.keys()}

    print(f"\nRunning evaluation with {n_trials} trials per model...")

    # models loop
    for name, model_factory in tqdm(models_def.items(), desc="Models"):
        trial_metrics = []

        # trials loop
        for i in tqdm(range(n_trials), desc=f"Trials ({name})", leave=False):
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42 + i, shuffle=True)

            model = model_factory()
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

            metrics["trial"] = i
            metrics["model"] = name
            metrics["timestamp"] = datetime.now().isoformat()

            trial_metrics.append(metrics)

        results_agg[name] = trial_metrics

        # export results
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name.replace(' ', '_')}_{timestamp_str}.json"
        filepath = os.path.join(args.output_dir, filename)

        serializable_metrics = []
        for m in trial_metrics:
            m_copy = m.copy()
            for k, v in m_copy.items():
                if isinstance(v, np.ndarray):
                    m_copy[k] = v.tolist()
                elif isinstance(v, (np.float32, np.float64)):
                    m_copy[k] = float(v)
            serializable_metrics.append(m_copy)

        with open(filepath, "w") as f:
            json.dump(serializable_metrics, f, indent=4)

        print(f"\nor {name} to {filepath}")

    print("\n" + "=" * 140)
    header = f"{'Model':<20} | {'MSE (Mean ± CI)':<25} | {'MAE (Mean ± CI)':<25} | {'Expl. Var.':<12} | {'Dir. Acc.':<12} | {'MSE/Torque':<30}"
    print(header)
    print("-" * 140)

    for name, metrics_list in results_agg.items():
        mses = [m["MSE"] for m in metrics_list]
        maes = [m["MAE"] for m in metrics_list]
        evs = [m["Explained Variance"] for m in metrics_list]
        das = [m["Direction Accuracy"] for m in metrics_list]

        mse_torques = np.array([m["MSE per Torque"] for m in metrics_list])

        # Compute Mean and 95% CI (1.96 * std / sqrt(n))
        def get_stats(data):
            mean = np.mean(data)
            ci = 1.96 * np.std(data) / np.sqrt(len(data))
            return mean, ci

        mse_mean, mse_ci = get_stats(mses)
        mae_mean, mae_ci = get_stats(maes)
        ev_mean, _ = get_stats(evs)
        da_mean, _ = get_stats(das)

        mse_torque_mean = np.mean(mse_torques, axis=0)

        mse_str = f"{mse_mean:.5f} ± {mse_ci:.5f}"
        mae_str = f"{mae_mean:.5f} ± {mae_ci:.5f}"
        ev_str = f"{ev_mean:.4f}"
        da_str = f"{da_mean:.4f}"
        mse_torque_str = f"[{mse_torque_mean[0]:.4f}, {mse_torque_mean[1]:.4f}, {mse_torque_mean[2]:.4f}]"

        print(f"{name:<20} | {mse_str:<25} | {mae_str:<25} | {ev_str:<12} | {da_str:<12} | {mse_torque_str:<30}")

    print("=" * 140)


if __name__ == "__main__":
    main()
