import h5py
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- Load dataset ---
with h5py.File("data/robot_mpc_dataset.h5", "r") as f:
    X = np.hstack([f["states"][:], f["targets"][:]])
    y = f["actions"][:]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale features and outputs ---
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)
X_train_scaled = scaler_X.transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# --- Define MLP ---
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 128),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    verbose=True,
)

# --- Train the model ---
mlp.fit(X_train_scaled, y_train_scaled)

# --- Predictions ---
y_train_pred_scaled = mlp.predict(X_train_scaled)
y_test_pred_scaled = mlp.predict(X_test_scaled)

# --- Inverse scale predictions ---
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled)

# --- Compute metrics ---
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train MSE: {train_mse:.5f}, Train R2: {train_r2:.5f}")
print(f"Test  MSE: {test_mse:.5f}, Test  R2: {test_r2:.5f}")

train_mse_scaled = mean_squared_error(y_train_scaled, y_train_pred_scaled)
test_mse_scaled = mean_squared_error(y_test_scaled, y_test_pred_scaled)
print(f"Train MSE (scaled): {train_mse_scaled:.5f}")
print(f"Test MSE  (scaled): {test_mse_scaled:.5f}")
