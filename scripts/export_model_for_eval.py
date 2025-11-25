"""
Export trained scikit-learn models for closed-loop evaluation.

This script trains a scikit-learn model on the dataset and exports it as a pickle file
for use in closed-loop evaluation.
"""

import argparse
import os
import pickle
from datetime import datetime

import h5py
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


def load_data(filepath):
    """Load dataset from HDF5 file."""
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
        # Input: [state (6D), target (3D)] -> 9D
        combined_input = np.hstack([states, targets])
        X_list.append(combined_input)
        Y_list.append(actions)

    X = np.vstack(X_list)
    Y = np.vstack(Y_list)
    return X, Y


def get_model(model_name):
    """Get model instance by name."""
    models = {
        "linear": Ridge(),
        "random_forest": RandomForestRegressor(n_estimators=50, n_jobs=-1, max_depth=10),
        "mlp": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
        "gradient_boosting": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42)),
        "knn": KNeighborsRegressor(n_neighbors=5),
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]


def main():
    parser = argparse.ArgumentParser(
        description="Train and export a scikit-learn model for closed-loop evaluation"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['linear', 'random_forest', 'mlp', 'gradient_boosting', 'knn'],
        help="Model type to train"
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default='data/robot_mpc_dataset.h5',
        help="Path to input HDF5 dataset (default: data/robot_mpc_dataset.h5)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/trained',
        help="Directory to save exported model (default: models/trained)"
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Model Export for Closed-Loop Evaluation")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data from: {args.input_file}")
    X, Y = load_data(args.input_file)
    
    if X is None or Y is None:
        print("Error: Failed to load data")
        return
    
    print(f"Data loaded: X shape = {X.shape}, Y shape = {Y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Train model
    print(f"\nTraining {args.model} model...")
    model = get_model(args.model)
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Train R² score: {train_score:.4f}")
    print(f"Test R² score: {test_score:.4f}")
    
    # Export model
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{args.model}_{timestamp_str}.pkl"
    filepath = os.path.join(args.output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModel exported to: {filepath}")
    print("=" * 80)
    print("\nTo use this model in closed-loop evaluation:")
    print(f"  python scripts/closed_loop_eval.py --controller-type sklearn --model-path {filepath}")
    print("=" * 80)


if __name__ == "__main__":
    main()
