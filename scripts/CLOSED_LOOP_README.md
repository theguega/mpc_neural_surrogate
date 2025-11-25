# Closed-Loop Evaluation

This directory contains scripts for closed-loop evaluation of MPC controllers and learned surrogate models.

## Overview

The closed-loop evaluation framework allows you to:
- Test MPC controller performance on randomly generated targets
- Evaluate trained surrogate models (PyTorch or scikit-learn) in a closed-loop setting
- Compare online performance metrics between different controllers
- Generate comprehensive evaluation reports

## Scripts

### 1. `closed_loop_eval.py`

Main evaluation script that runs multiple episodes with different target positions.

**Usage:**

```bash
# Test MPC controller (ground truth)
python scripts/closed_loop_eval.py \
    --controller-type mpc \
    --num-episodes 10 \
    --max-steps 200 \
    --tolerance 0.02

# Test scikit-learn model
python scripts/closed_loop_eval.py \
    --controller-type sklearn \
    --model-path models/trained/mlp_model.pkl \
    --num-episodes 10 \
    --render

# Test PyTorch model
python scripts/closed_loop_eval.py \
    --controller-type pytorch \
    --model-path models/trained/neural_net.pt \
    --num-episodes 20 \
    --max-steps 150
```

**Arguments:**

- `--controller-type`: Type of controller (`mpc`, `sklearn`, or `pytorch`)
- `--model-path`: Path to trained model file (required for `sklearn` and `pytorch`)
- `--num-episodes`: Number of evaluation episodes (default: 10)
- `--max-steps`: Maximum steps per episode (default: 200)
- `--tolerance`: Distance threshold to consider target reached in meters (default: 0.02)
- `--render`: Enable MuJoCo visualization
- `--output-dir`: Directory to save results (default: `results/closed_loop`)
- `--seed`: Random seed for reproducibility (default: 42)

**Output:**

The script generates a JSON file with:
- Aggregate metrics (success rate, mean errors, control effort, solve times)
- Per-episode detailed metrics
- Configuration parameters

### 2. `export_model_for_eval.py`

Helper script to train and export scikit-learn models in pickle format.

**Usage:**

```bash
# Train and export an MLP model
python scripts/export_model_for_eval.py \
    --model mlp \
    --input-file data/robot_mpc_dataset.h5 \
    --output-dir models/trained

# Train other model types
python scripts/export_model_for_eval.py --model random_forest
python scripts/export_model_for_eval.py --model linear
python scripts/export_model_for_eval.py --model gradient_boosting
python scripts/export_model_for_eval.py --model knn
```

**Arguments:**

- `--model`: Model type (`linear`, `random_forest`, `mlp`, `gradient_boosting`, `knn`)
- `--input-file`: Path to HDF5 dataset (default: `data/robot_mpc_dataset.h5`)
- `--output-dir`: Directory to save model (default: `models/trained`)
- `--test-size`: Fraction for test split (default: 0.2)
- `--seed`: Random seed (default: 42)

## Metrics

The evaluation computes the following metrics for each episode:

### Success Metrics
- `reached_target`: Whether the target was reached within tolerance
- `steps_to_target`: Number of steps to reach target (if successful)

### Error Metrics
- `final_ee_error`: End-effector error at episode end (meters)
- `final_joint_error`: Joint angle error at episode end (radians)
- `mean_ee_error`: Average end-effector error during episode
- `std_ee_error`: Standard deviation of end-effector error
- `max_ee_error`: Maximum end-effector error during episode
- `mean_joint_error`: Average joint error during episode

### Control Metrics
- `mean_control_effort`: Average torque magnitude
- `std_control_effort`: Standard deviation of torque magnitude
- `total_control_effort`: Cumulative torque magnitude

### Timing Metrics
- `mean_solve_time`: Average time to compute control action
- `std_solve_time`: Standard deviation of solve time
- `total_time`: Total computation time for episode

## Example Workflow

### 1. Generate Dataset

First, generate the MPC dataset if you haven't already:

```bash
python scripts/data_generator.py
```

### 2. Export a Trained Model

Train and export a model for evaluation:

```bash
python scripts/export_model_for_eval.py --model mlp
```

### 3. Evaluate MPC Baseline

Run closed-loop evaluation with the MPC controller:

```bash
python scripts/closed_loop_eval.py \
    --controller-type mpc \
    --num-episodes 50 \
    --output-dir results/closed_loop
```

### 4. Evaluate Learned Model

Run closed-loop evaluation with the trained model:

```bash
python scripts/closed_loop_eval.py \
    --controller-type sklearn \
    --model-path models/trained/mlp_20251125_045459.pkl \
    --num-episodes 50 \
    --output-dir results/closed_loop
```

### 5. Compare Results

Load and compare the JSON results files to analyze performance differences.

## Notes

- **Model Input Format**: Models expect 9D input: `[q (3), q_dot (3), target_joints (3)]`
- **Model Output Format**: Models should output 3D torque: `[tau1, tau2, tau3]`
- **Workspace Bounds**: Targets are randomly generated within the robot's reachable workspace
- **IK Solver**: Inverse kinematics is used to convert Cartesian targets to joint targets
- **Gravity Compensation**: Control outputs are automatically combined with gravity compensation terms

## Troubleshooting

### IK Convergence Warnings

If you see many "IK did not converge" warnings, the randomly generated targets may be at the edge of the workspace. This is normal for difficult targets, and the approximate solution is still used.

### Model Loading Errors

- For scikit-learn: Ensure the pickle file was created with a compatible scikit-learn version
- For PyTorch: Ensure PyTorch is installed and the model architecture matches

### Poor Model Performance

If a learned model performs poorly in closed-loop:
- Check that the model was trained on sufficient data
- Verify the model's offline metrics (MSE, MAE) are good
- Consider that offline performance doesn't always translate to online performance
- The model may need to be trained with different features or architecture
