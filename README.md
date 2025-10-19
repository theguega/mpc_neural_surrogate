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
python src/mpc_surrogate/generate_data.py
```

This generates `data/mpc_3dof_dataset.h5` with MPC data.

### Visualize Sample

To visualize a random sample, use mjpython:

```bash
mjpython src/mpc_surrogate/visualize_sample.py
```

This launches the Mujoco viewer to replay a sample trajectory.
