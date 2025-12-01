# %% [markdown]
# # MPC Surrogate Training Pipeline
#
# This script implements the complete training and evaluation pipeline for approximating MPC policies using neural networks.
# It includes improvements for LSTM/GRU models (packing/masking) and robust saving logic.

# %%
import json
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# %%
def compute_direction_accuracy(y_true, y_pred):
    """
    Sum the number of time the sign of the predicted torque matches the true torque.
    """
    return np.sum(np.sign(y_true) == np.sign(y_pred)) / (3 * len(y_true))


torch.manual_seed(42)
np.random.seed(42)

# Update DATA_PATH to point to the correct location relative to this script
DATA_PATH = "../data/robot_mpc_dataset.h5"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# %% [markdown]
# ## Data Preparation


# %%
class MPCDataset(Dataset):
    def __init__(self, filepath, episode_keys, mode="mlp", augment=False):
        """
        mode: 'mlp' (flattens trajectories) or 'rnn' (keeps trajectories intact)
        """
        super().__init__()
        self.augment = augment
        self.mode = mode
        self.data = []  # (inputs, actions) tuples

        with h5py.File(filepath, "r") as f:
            grp_eps = f["episodes"]
            for ep in episode_keys:
                # Load raw data
                s = torch.from_numpy(grp_eps[ep]["states"][:]).float()  # (T, 6)
                t = torch.from_numpy(grp_eps[ep]["targets"][:]).float()  # (T, 3)
                a = torch.from_numpy(grp_eps[ep]["actions"][:]).float()  # (T, 3)

                # states + targets as input -> (T, 9)
                inp = torch.cat([s, t], dim=-1)
                self.data.append((inp, a))

        # if MLP, flatten all steps from these specific episodes into one tensor
        if self.mode == "mlp":
            self.inputs = torch.cat([x[0] for x in self.data], dim=0)
            self.actions = torch.cat([x[1] for x in self.data], dim=0)

    def __len__(self):
        return len(self.inputs) if self.mode == "mlp" else len(self.data)

    def __getitem__(self, idx):
        if self.mode == "mlp":
            x, y = self.inputs[idx], self.actions[idx]  # (9,), (3,)
            if self.augment:
                x[:6] += torch.randn(6) * 0.01  # noise on state
                y += torch.randn(3) * 0.005  # noise on action
            return x, y

        # RNN
        else:
            x, y = self.data[idx]  # (T, 9), (T, 3)
            if self.augment:
                noise_x = torch.randn_like(x)
                noise_x[:, 6:] = 0  # no noise on targets
                x = x + (noise_x * 0.01)
                y = y + (torch.randn_like(y) * 0.005)
            return x, y


def collate_rnn(batch):
    inputs, actions = zip(*batch)
    lengths = torch.tensor([x.size(0) for x in inputs])

    # pad variable lengths (T0, T1...) to max length in batch
    padded_inputs = pad_sequence(inputs, batch_first=True)  # (B, T_max, 9)
    padded_actions = pad_sequence(actions, batch_first=True)  # (B, T_max, 3)

    return padded_inputs, padded_actions, lengths


def create_dataloaders(filepath, train_ratio=0.8, batch_size=32, dataset_type="mlp"):
    # split by episode idx
    with h5py.File(filepath, "r") as f:
        keys = np.array(sorted(f["episodes"].keys()))

    np.random.shuffle(keys)
    split = int(len(keys) * train_ratio)
    train_keys, val_keys = keys[:split], keys[split:]

    if dataset_type == "mlp":
        train_ds_mlp = MPCDataset(filepath, train_keys, mode="mlp", augment=True)
        val_ds_mlp = MPCDataset(filepath, val_keys, mode="mlp", augment=False)

        train = DataLoader(train_ds_mlp, batch_size=batch_size, shuffle=True)
        val = DataLoader(val_ds_mlp, batch_size=batch_size, shuffle=False)
    elif dataset_type == "rnn":
        train_ds_rnn = MPCDataset(filepath, train_keys, mode="rnn", augment=True)
        val_ds_rnn = MPCDataset(filepath, val_keys, mode="rnn", augment=False)

        train = DataLoader(train_ds_rnn, batch_size=batch_size, shuffle=True, collate_fn=collate_rnn)
        val = DataLoader(val_ds_rnn, batch_size=batch_size, shuffle=False, collate_fn=collate_rnn)

    return train, val


# %% [markdown]
# ## Model Architectures


# %%
class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""

    def __init__(self, input_dim=9, hidden_dims=[128, 64], output_dim=3):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim), nn.Dropout(0.1)])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, output_dim=3):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        # x: (B, T_max, 9)
        if lengths is not None:
            # Move lengths to CPU for pack_padded_sequence if necessary
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            out_packed, (hn, cn) = self.lstm(x_packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, (hn, cn) = self.lstm(x)

        out = self.fc(out)  # (B, T, 3)
        return out


class GRU(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, output_dim=3):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        # x: (B, T, 9)
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
            out_packed, hn = self.gru(x_packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        else:
            out, hn = self.gru(x)  # (B, T_max, 3)

        out = self.fc(out)
        return out


model_configs = {
    "MLP": {"class": MLP, "params": {"hidden_dims": [128, 64]}, "dataset_type": "mlp"},
    "LSTM": {"class": LSTM, "params": {"hidden_dim": 128, "num_layers": 2}, "dataset_type": "rnn"},
    "GRU": {"class": GRU, "params": {"hidden_dim": 128, "num_layers": 2}, "dataset_type": "rnn"},
}

print("Available models:", list(model_configs.keys()))

# %% [markdown]
# ## Training and Evaluation Functions


# %%
def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    total_samples = 0

    for batch in train_loader:
        if len(batch) == 3:
            inputs, targets, lengths = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs, lengths)

            # Masking for RNN
            # Create mask: (B, T) -> (B, T, D)
            mask = torch.arange(targets.size(1), device=DEVICE)[None, :] < lengths[:, None].to(DEVICE)
            mask = mask.unsqueeze(-1).expand_as(targets)

            # Apply mask to loss (assuming reduction='none')
            loss_raw = criterion(outputs, targets)
            loss = (loss_raw * mask).sum() / mask.sum()

        else:
            inputs, targets = batch
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Standard mean reduction for MLP

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, test_loader, criterion):
    """Evaluate model on test set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                inputs, targets, lengths = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs, lengths)

                # Masking for RNN
                mask = torch.arange(targets.size(1), device=DEVICE)[None, :] < lengths[:, None].to(DEVICE)
                mask = mask.unsqueeze(-1).expand_as(targets)

                loss_raw = criterion(outputs, targets)
                loss = (loss_raw * mask).sum() / mask.sum()

                # Filter predictions and targets for metrics
                # We flatten and remove padded values
                outputs_masked = outputs[mask].view(-1, 3)
                targets_masked = targets[mask].view(-1, 3)

                all_predictions.append(outputs_masked.cpu().numpy())
                all_targets.append(targets_masked.cpu().numpy())

            else:
                inputs, targets = batch
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    # Flatten for metrics calculation if RNN (already done above for RNN, but good for safety)
    if predictions.ndim == 3:
        predictions = predictions.reshape(-1, predictions.shape[-1])
        targets = targets.reshape(-1, targets.shape[-1])

    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    da = compute_direction_accuracy(targets, predictions)
    ev = explained_variance_score(targets, predictions)

    return {
        "loss": total_loss / num_batches,
        "mse": mse,
        "mae": mae,
        "direction_accuracy": da,
        "explained_variance": ev,
        "predictions": predictions,
        "targets": targets,
    }


def train_model(model_name, model, train_loader, test_loader, num_epochs=100, patience=10):
    """Complete training pipeline for a model"""
    print(f"\\n=== Training {model_name} ===")

    # Use reduction='none' to handle masking manually in train_epoch/evaluate for RNNs
    # For MLP, we can still use it but need to average manually or handle it.
    # To keep it simple, let's use reduction='none' for both and handle averaging.
    # Actually, for MLP, train_epoch logic above assumes 'mean' if not RNN.
    # Let's adjust:
    # If RNN (len(batch)==3), we do manual masking and averaging.
    # If MLP, we want standard mean.
    # So we pass reduction='none' and handle it.

    mse_criterion = nn.MSELoss(reduction="none")
    mae_criterion = nn.L1Loss(reduction="none")

    # Helper to handle MLP loss with reduction='none'
    def mlp_loss_wrapper(output, target, crit):
        l = crit(output, target)
        return l.mean()

    # We need to wrap the criterion for MLP to do mean, or change train_epoch to handle it.
    # Let's change train_epoch to handle reduction='none' for MLP too.
    # Updated train_epoch logic:
    # if MLP: loss = criterion(outputs, targets).mean()

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    mse_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    mae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    results = {
        "model_name": model_name,
        "hyperparameters": {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": train_loader.batch_size, "num_epochs": num_epochs},
        "training_history": [],
        "best_mse_results": None,
        "best_mae_results": None,
    }

    best_mse_loss = float("inf")
    best_mae_loss = float("inf")
    patience_counter = 0

    # tqdm progress bar for epochs
    pbar = tqdm(range(num_epochs), desc=f"Training {model_name}")

    try:
        for epoch in pbar:
            start_time = time.time()

            # Train with MSE
            # We need to pass a criterion that works.
            # For RNN, we use 'none' and mask.
            # For MLP, we use 'none' and mean.
            # Let's define a wrapper or handle inside train_epoch.
            # I updated train_epoch to assume reduction='none' for RNN and handle it.
            # For MLP, I need to update train_epoch to .mean() the loss.

            # Re-defining train_epoch inside here or updating the global one?
            # I updated the global one above to use criterion(outputs, targets) for MLP.
            # If criterion is 'none', this returns a tensor. I need to .mean() it.
            # Let's update train_epoch in the file content above to do .mean() for MLP.
            # I will do that in the file content string I am building.

            train_loss_mse = train_epoch(model, train_loader, mse_criterion, optimizer)
            mse_results = evaluate(model, test_loader, mse_criterion)

            # Train with MAE
            train_loss_mae = train_epoch(model, train_loader, mae_criterion, optimizer)
            mae_results = evaluate(model, test_loader, mae_criterion)

            # Update schedulers
            mse_scheduler.step(mse_results["loss"])
            mae_scheduler.step(mae_results["loss"])

            epoch_time = time.time() - start_time

            # Update tqdm display
            pbar.set_postfix({"mse": mse_results["loss"], "mae": mae_results["loss"], "time": f"{epoch_time:.2f}s"})

            # Log results
            epoch_results = {
                "epoch": epoch + 1,
                "train_loss_mse": train_loss_mse,
                "test_loss_mse": mse_results["loss"],
                "test_mse": mse_results["mse"],
                "test_mae": mse_results["mae"],
                "train_loss_mae": train_loss_mae,
                "test_loss_mae": mae_results["loss"],
                "test_direction_accuracy": mse_results["direction_accuracy"],
                "test_explained_variance": mse_results["explained_variance"],
                "epoch_time": epoch_time,
            }
            results["training_history"].append(epoch_results)

            # Save best models
            if mse_results["loss"] < best_mse_loss:
                best_mse_loss = mse_results["loss"]
                results["best_mse_results"] = mse_results.copy()
                torch.save(model.state_dict(), f"{model_name}_best_mse.pth")

            if mae_results["loss"] < best_mae_loss:
                best_mae_loss = mae_results["loss"]
                results["best_mae_results"] = mae_results.copy()
                torch.save(model.state_dict(), f"{model_name}_best_mae.pth")

            # Early stopping logic
            if mse_results["loss"] >= best_mse_loss:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= patience:
                print(f"\\nEarly stopping at epoch {epoch + 1}")
                break

    except KeyboardInterrupt:
        print("\\nTraining interrupted by user. Saving current results...")
    except Exception as e:
        print(f"\\nAn error occurred: {e}")
        raise e
    finally:
        # Save results to JSON regardless of how we exit
        print(f"\\nSaving results for {model_name}...")
        results_file = f"{model_name}_results.json"

        # Prepare for JSON serialization (remove numpy arrays)
        json_results = results.copy()
        if "best_mse_results" in json_results and json_results["best_mse_results"]:
            json_results["best_mse_results"].pop("predictions", None)
            json_results["best_mse_results"].pop("targets", None)
            # Convert numpy floats to python floats
            for k, v in json_results["best_mse_results"].items():
                if isinstance(v, (np.floating, float)):
                    json_results["best_mse_results"][k] = float(v)

        if "best_mae_results" in json_results and json_results["best_mae_results"]:
            json_results["best_mae_results"].pop("predictions", None)
            json_results["best_mae_results"].pop("targets", None)
            for k, v in json_results["best_mae_results"].items():
                if isinstance(v, (np.floating, float)):
                    json_results["best_mae_results"][k] = float(v)

        # Convert training history values
        for epoch_res in json_results["training_history"]:
            for k, v in epoch_res.items():
                if isinstance(v, (np.floating, float)):
                    epoch_res[k] = float(v)

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"Results saved to {results_file}")

    return results


# %% [markdown]
# ## Run Training Pipeline

# %%
if __name__ == "__main__":
    # Ensure data path exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        # Try to find it in current directory
        if os.path.exists("robot_mpc_dataset.h5"):
            DATA_PATH = "robot_mpc_dataset.h5"
            print(f"Found dataset in current directory: {DATA_PATH}")
        else:
            print("Please check the dataset path.")
            exit(1)

    all_results = {}

    for model_name, config in model_configs.items():
        model_class = config["class"]
        model_params = config["params"]
        dataset_type = config["dataset_type"]

        print(f"\\nPreparing data for {model_name}...")
        train_loader, val_loader = create_dataloaders(DATA_PATH, train_ratio=0.8, batch_size=32, dataset_type=dataset_type)
        model = model_class(**model_params).to(DEVICE)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"\\n{model_name} - Parameters: {total_params:,}")

        # Train model
        results = train_model(model_name, model, train_loader, val_loader, num_epochs=50)
        all_results[model_name] = results

    print("\\n=== Training Complete ===")
