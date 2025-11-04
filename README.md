# MPC Surrogate

Approximating Model Predictive Control policies using neural networks for a 3DOF robotic arm.

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Usage

### Generate Dataset

Run the data generation script:

```bash
python src/mpc_surrogate/data_generator.py
```

This generates `data/robot_mpc_dataset.h5` with MPC data.

### Tests

To ensure MPC and inverse kinematics are working we implemented some unit tests :

```bash
python tests/ik_test.py

# need mjpython to run interactive mujuco simulation on macos
mjpython tests/mpc_mujoco_sim.py
```

This launches the Mujoco viewer to replay a sample trajectory.

## MPC Dataset Generation

We can generate a dataset for learning to imitate a Model Predictive Controller (MPC) for a 3-DoF robotic arm simulated in MuJoCo. The goal is to create input-output pairs suitable for training a neural network surrogate for the MPC.

### How it works

1. **Environment Setup**

   * The MuJoCo simulation is loaded with a 3-DoF robotic arm.
   * An MPC controller is instantiated with a specified timestep (`dt=0.05`) and prediction horizon (`N=20`).

2. **Target Sampling**

   * At each step, a new end-effector (EE) target is sampled with low probability (`5%`) to ensure smooth transitions between targets.
   * Targets are generated within the reachable workspace, avoiding singularities near the base.
   * `solve_inverse_kinematics()` is used to convert EE targets into corresponding joint positions for the MPC controller.

3. **Data Collection Loop**

   * For each episode (default 100 episodes) and each timestep (default 150 steps per episode):

     * The current state `[q1, q2, q3, q̇1, q̇2, q̇3]` is recorded.
     * MPC computes the torque `τ_mpc` to move toward the target joint positions.
     * We are taking the torque `τ_mpc` for our dataset.
     * Total torque is calculated as `τ_total = τ_mpc + qfrc_bias` to include static dynamics effects such as gravity and friction.
     * The simulation is stepped forward using `τ_total` for the appropriate number of simulation steps per MPC step.
   * If the MPC fails to solve, the step is skipped.

4. **Dataset Storage**

   * Data is stored in an HDF5 file with gzip compression:

     * `states`: joint positions and velocities, shape `(num_samples, 6)`
     * `targets`: end-effector target positions, shape `(num_samples, 3)`
     * `actions`: MPC torques applied, shape `(num_samples, 3)`

---

### Example

```python
import h5py
import numpy as np

with h5py.File("data/robot_mpc_dataset.h5", "r") as f:
    states = f["states"][:]
    targets = f["targets"][:]
    actions = f["actions"][:]

print(states.shape)  # e.g., (15000, 6)
print(targets.shape) # e.g., (15000, 3)
print(actions.shape) # e.g., (15000, 3)
```

This dataset can then be used to train a neural network to approximate the MPC controller in a supervised learning setting.
