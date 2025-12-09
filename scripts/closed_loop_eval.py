"""
Closed-loop evaluation script for comparing MPC controller with learned surrogate models.

This script generates test episodes on-the-fly where the robot tracks randomly sampled targets using either:
- The MPC controller (ground truth)
- A trained surrogate model (PyTorch .pt/.pth or scikit-learn .pkl)

Models are automatically discovered from:
- results/scikit_learn_baseline/models/ (scikit-learn .pkl files)
- results/pytorch_comparison/results_sliding_window/models/ (PyTorch .pt/.pth files)

For each evaluation run:
1. Random target positions are sampled in 3D Cartesian space
2. Robot attempts to reach targets within 150 timesteps
3. Metrics are computed and aggregated (success rate, tracking error, control effort, etc.)
4. Results are saved to JSON files in results/closed_loop/

When evaluating multiple models, test episodes are pre-generated once to ensure fair comparison
across all models within the same run. Each new execution generates fresh random episodes.
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime

import h5py
import mujoco
import mujoco.viewer
import numpy as np
import psutil
from mpc_surrogate.mpc_controller import MPCController
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics
from data_generator import sample_3d_cartesian_target

# PyTorch model architectures
try:
    import torch
    import torch.nn as nn
    
    class MLP(nn.Module):
        """Simple Multi-Layer Perceptron (matches notebook architecture)"""
        def __init__(self, input_dim=9, hidden_dims=[128, 64], output_dim=3):
            super(MLP, self).__init__()
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    class GRU(nn.Module):
        """GRU-based recurrent neural network for sequence-to-sequence prediction"""
        def __init__(self, input_dim=9, hidden_dim=128, num_layers=2, output_dim=3):
            super(GRU, self).__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            # GRU expects (batch, seq_len, features)
            if x.dim() == 2:  # If input is (batch, features), add sequence dimension
                x = x.unsqueeze(1)
            gru_out, _ = self.gru(x)
            # Use last output for prediction
            out = self.fc(gru_out[:, -1, :])
            return out
except ImportError:
    pass  # PyTorch not available, only sklearn will work


def load_model(model_path, model_type):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the model file (.pt/.pth for PyTorch, .pkl for scikit-learn)
        model_type: 'pytorch' or 'sklearn'
    
    Returns:
        Loaded model object
    """
    if model_type == 'pytorch':
        import torch
        
        # Load the file
        loaded_obj = torch.load(model_path, map_location='cpu')
        
        # Check if it's a full model or a state_dict
        if isinstance(loaded_obj, nn.Module):
            # Full model was saved
            model = loaded_obj
            print("Loaded full PyTorch model")
        elif isinstance(loaded_obj, dict):
            # State dict was saved - detect architecture from filename
            filename = os.path.basename(model_path).lower()
            print(f"Detected state_dict, detecting architecture from filename...")
            
            # Detect architecture based on filename
            if 'gru' in filename:
                print("Instantiating GRU architecture...")
                # Detect number of GRU layers from state dict keys
                num_layers = 0
                hidden_dim = 128  # default
                for key in loaded_obj.keys():
                    if key.startswith('gru.weight_ih_l'):
                        layer_idx = int(key.split('_l')[1])
                        num_layers = max(num_layers, layer_idx + 1)
                        # Extract hidden_dim from weight shape
                        # weight_ih shape is (3*hidden_dim, input_dim) for GRU
                        weight_shape = loaded_obj[key].shape
                        hidden_dim = weight_shape[0] // 3
                num_layers = max(num_layers, 1)  # At least 1 layer
                print(f"Detected {num_layers} GRU layers with hidden_dim={hidden_dim}")
                model = GRU(input_dim=9, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=3)
            elif 'mlp' in filename:
                # Detect input dimension from filename (sliding window)
                input_dim = 9  # default
                window_size = 1  # default
                if 'win5' in filename or 'window5' in filename:
                    input_dim = 33  # 5*6 + 3
                    window_size = 5
                    print(f"Detected sliding window size 5 from filename")
                elif 'win' in filename or 'window' in filename:
                    # Try to extract window size from filename
                    import re
                    match = re.search(r'win(?:dow)?_?(\d+)', filename, re.IGNORECASE)
                    if match:
                        window_size = int(match.group(1))
                        input_dim = window_size * 6 + 3
                        print(f"Detected sliding window size {window_size} from filename")
                
                # If still default, try to infer from first layer weight shape
                if input_dim == 9 and 'network.0.weight' in loaded_obj:
                    detected_input_dim = loaded_obj['network.0.weight'].shape[1]
                    if detected_input_dim != 9:
                        input_dim = detected_input_dim
                        window_size = (input_dim - 3) // 6
                        print(f"Detected input_dim={input_dim} from weight shape (window_size={window_size})")
                
                print(f"Using input_dim={input_dim}")
                
                print(f"Using input_dim={input_dim}")
                print("Instantiating MLP architecture...")
                # Detect all hidden dimensions from state dict
                # Look for LINEAR layers only (indices 0, 2, 4, 6, 8 if using BN/Dropout, or 0, 1, 2... if not)
                hidden_dims = []
                linear_indices = []
                try:
                    for key in loaded_obj.keys():
                        if key.startswith('network.') and '.weight' in key:
                            # Extract layer index
                            idx = int(key.split('.')[1])
                            if idx not in linear_indices:
                                linear_indices.append(idx)
                    
                    linear_indices = sorted(set(linear_indices))
                    
                    # Check if BatchNorm layers are present (even indices would be BN if using BN/ReLU/Dropout pattern)
                    has_batchnorm = any(f'network.{i}.running_mean' in loaded_obj for i in range(1, 20))
                    
                    # If no BN, linear indices are consecutive (0, 1, 2, ...)
                    # If BN, linear indices skip by 4 (0, 4, 8, ...) or by 2 (0, 2, 4, ...)
                    if not has_batchnorm:
                        # Simple linear network: consecutive indices
                        for idx in linear_indices[:-1]:  # All but the last (output) layer
                            weight = loaded_obj[f'network.{idx}.weight']
                            hidden_dims.append(weight.shape[0])
                    else:
                        # Network with BN/ReLU/Dropout: indices are 0, 2, 4, 6, 8... (or 0, 4, 8...)
                        # Hidden dims are the output dims of hidden layers (all but last)
                        for idx in linear_indices[:-1]:
                            weight = loaded_obj[f'network.{idx}.weight']
                            hidden_dims.append(weight.shape[0])
                    
                    if not hidden_dims:
                        hidden_dims = [256, 128, 64, 32]  # fallback default
                except Exception as e:
                    print(f"Error detecting hidden dims: {e}")
                    hidden_dims = [256, 128, 64, 32]  # fallback default
                
                print(f"Detected hidden_dims={hidden_dims}, has_batchnorm={has_batchnorm}")
                
                # Create MLP WITHOUT batchnorm/dropout if the saved model doesn't have them
                if has_batchnorm:
                    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=3)
                else:
                    # Create a version without BN and dropout
                    class SimpleMLP(nn.Module):
                        def __init__(self, input_dim=9, hidden_dims=[128, 64], output_dim=3):
                            super(SimpleMLP, self).__init__()
                            layers = []
                            prev_dim = input_dim
                            for hidden_dim in hidden_dims:
                                layers.extend([
                                    nn.Linear(prev_dim, hidden_dim),
                                    nn.ReLU()
                                ])
                                prev_dim = hidden_dim
                            layers.append(nn.Linear(prev_dim, output_dim))
                            self.network = nn.Sequential(*layers)
                        
                        def forward(self, x):
                            return self.network(x)
                    
                    model = SimpleMLP(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=3)
                
                # Attach window_size as model attribute for later use
                model.window_size = window_size
            else:
                # Default to MLP if architecture cannot be determined
                print("Architecture not detected in filename, defaulting to MLP...")
                # Try to infer input_dim from first layer weight
                input_dim = 9
                if 'network.0.weight' in loaded_obj:
                    input_dim = loaded_obj['network.0.weight'].shape[1]
                    print(f"Detected input_dim={input_dim} from weight shape")
                model = MLP(input_dim=input_dim, hidden_dims=[128, 64], output_dim=3)
            
            model.load_state_dict(loaded_obj)
            print(f"Loaded state_dict")
        else:
            raise ValueError(f"Unknown PyTorch save format. Expected nn.Module or dict, got {type(loaded_obj)}")
        
        model.eval()
        return model
    elif model_type == 'sklearn':
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_action(model, state, target, model_type, state_history=None, window_size=1):
    """
    Predict control action using the loaded model.
    
    Args:
        model: Loaded model object
        state: Current robot state (6D: [q, q_dot])
        target: Target joint positions (3D)
        model_type: 'pytorch' or 'sklearn'
        state_history: List of past states for windowed models (optional)
        window_size: Number of timesteps in sliding window (default=1)
    
    Returns:
        Predicted control action (3D torque vector)
    """
    # For windowed models, construct input from history
    if window_size > 1 and state_history is not None:
        # Use last window_size states
        history_window = state_history[-window_size:]
        # Pad if needed (at episode start)
        while len(history_window) < window_size:
            history_window.insert(0, state)  # Repeat first state
        # Flatten: [state_t-w+1, ..., state_t, target]
        input_data = np.concatenate([np.concatenate(history_window), target])
    else:
        # Standard input: [state, target]
        input_data = np.concatenate([state, target])
    
    if model_type == 'pytorch':
        import torch
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0)
            output = model(input_tensor)
            return output.squeeze(0).cpu().numpy()
    elif model_type == 'sklearn':
        input_data = input_data.reshape(1, -1)
        return model.predict(input_data)[0]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


SKLEARN_MODELS_DIR = 'results/scikit_learn_baseline/models'


def list_available_models(models_dir: str = SKLEARN_MODELS_DIR):
    """
    List all available model files:
    - results/scikit_learn_baseline/models/ for .pkl (scikit-learn)
    - results/pytorch_comparison/results_sliding_window/models/ for .pt/.pth (PyTorch comparisons)
    
    Returns:
        List of tuples (model_name, filepath, model_type)
    """
    import glob
    
    models = []
    
    # Find scikit-learn models
    if os.path.exists(models_dir):
        pkl_files = glob.glob(os.path.join(models_dir, '*.pkl'))
        models.extend([(os.path.basename(f), f, 'sklearn') for f in sorted(pkl_files)])
    
    # Find PyTorch models
    pytorch_dir = 'results/pytorch_comparison/results_sliding_window/models'
    if os.path.exists(pytorch_dir):
        pt_files = glob.glob(os.path.join(pytorch_dir, '*.pt'))
        models.extend([(os.path.basename(f), f, 'pytorch') for f in sorted(pt_files)])
        pth_files = glob.glob(os.path.join(pytorch_dir, '*.pth'))
        models.extend([(os.path.basename(f), f, 'pytorch') for f in sorted(pth_files)])
    
    return sorted(models)


def run_episode(env, controller, model, model_type, target_xyz,
                max_steps=150, tolerance=0.03, render=False, viewer=None,
                window_size=1):
    """
    Run a single episode where the robot tracks targets from the dataset.
    
    Args:
        env: MuJoCoEnvironment instance
        controller: MPCController instance (can be None if using model)
        model: Trained model (can be None if using MPC)
        model_type: 'pytorch', 'sklearn', or 'mpc'
        target_xyz: 3D target position sampled for this test episode
        max_steps: Maximum steps per episode (default: 150)
        tolerance: Distance threshold to consider target reached
        render: Whether to render the episode
        viewer: MuJoCo viewer instance (if rendering)
    
    Returns:
        Dictionary containing episode metrics
    """
    n_sim_steps_per_mpc_step = int(0.05 / env.model.opt.timestep)
    
    # Reset environment
    obs = env.reset()
    mujoco.mj_forward(env.model, env.data)
    
    # Setup target visualization
    if render and viewer is not None:
        target_body_id = env.model.body("target").id
        env.model.body_mocapid[target_body_id] = 0
    
    # Episode tracking
    ee_errors = []
    joint_errors = []
    control_efforts = []
    solve_times = []
    action_differences = []  # Compare with MPC actions
    cpu_percentages = []  # Track CPU utilization
    state_history = []  # For sliding window models
    
    # Get process for CPU tracking
    process = psutil.Process()
    
    # Solve IK to get target joint positions once
    _, target_joints = solve_inverse_kinematics(env, target_xyz)
    for step in range(max_steps):
        current_state = obs[:6]
        current_ee_pos = env.get_ee_position()
        
        # Update target visualization
        if render and viewer is not None:
            env.data.mocap_pos[0] = target_xyz
        
        ee_error = np.linalg.norm(current_ee_pos - target_xyz)
        joint_error = np.linalg.norm(current_state[:3] - target_joints)
        
        ee_errors.append(ee_error)
        joint_errors.append(joint_error)
        
        # Early termination: if target is already reached within 0.03m, end episode
        if ee_error < 0.03:
            break
        # Compute control action
        start_time = time.time()
        cpu_before = process.cpu_percent()
        
        if model_type == 'mpc':
            tau, solved = controller.solve(current_state, target_joints)
            if not solved:
                tau = np.zeros(3)
        else:
            tau = predict_action(model, current_state, target_joints, model_type, 
                               state_history=state_history, window_size=window_size)
        
        # Update state history for windowed models
        state_history.append(current_state.copy())
        if len(state_history) > window_size:
            state_history.pop(0)
        
        solve_time = time.time() - start_time
        solve_time = time.time() - start_time
        cpu_after = process.cpu_percent()
        
        solve_times.append(solve_time)
        cpu_percentages.append((cpu_before + cpu_after) / 2)
        
        control_effort = np.linalg.norm(tau)
        control_efforts.append(control_effort)
        
        # If running MPC, action difference is zero baseline; for models, we skip
        if model_type == 'mpc':
            action_differences.append(0.0)
        
        # Apply control with gravity compensation
        total_tau = tau + env.data.qfrc_bias
        
        # Step simulation
        for _ in range(n_sim_steps_per_mpc_step):
            obs, _, _, _ = env.step(total_tau)
        
        # Render if requested
        if render and viewer is not None:
            env.render(viewer)
            time.sleep(0.001)
    
    # Compute episode metrics
    final_ee_error = ee_errors[-1] if ee_errors else 0
    reached_target = bool(final_ee_error < tolerance)
    
    metrics = {
        'episode_name': 'generated_test_episode',
        'num_steps': len(ee_errors),
        'reached_target': reached_target,
        'final_ee_error': final_ee_error,
        'final_joint_error': joint_errors[-1] if joint_errors else 0,
        'mean_ee_error': np.mean(ee_errors) if ee_errors else 0,
        'std_ee_error': np.std(ee_errors) if ee_errors else 0,
        'max_ee_error': np.max(ee_errors) if ee_errors else 0,
        'mean_joint_error': np.mean(joint_errors) if joint_errors else 0,
        'std_joint_error': np.std(joint_errors) if joint_errors else 0,
        'mean_control_effort': np.mean(control_efforts) if control_efforts else 0,
        'std_control_effort': np.std(control_efforts) if control_efforts else 0,
        'total_control_effort': np.sum(control_efforts) if control_efforts else 0,
        'mean_solve_time': np.mean(solve_times) if solve_times else 0,
        'std_solve_time': np.std(solve_times) if solve_times else 0,
        'total_time': np.sum(solve_times) if solve_times else 0,
        'mean_action_diff_from_mpc': np.mean(action_differences) if action_differences else 0,
        'std_action_diff_from_mpc': np.std(action_differences) if action_differences else 0,
        'mean_cpu_percent': np.mean(cpu_percentages) if cpu_percentages else 0,
        'std_cpu_percent': np.std(cpu_percentages) if cpu_percentages else 0
    }
    
    return metrics


def run_evaluation(args, test_episodes):
    """
    Main evaluation loop: run multiple episodes and aggregate results.
    
    Args:
        args: Command-line arguments
        test_episodes: List of pre-generated target positions (generated once per script execution)
    """
    print("=" * 80)
    print("Closed-Loop Evaluation (Generated Test Episodes)")
    print("=" * 80)
    
    # Initialize environment
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    print(f"Environment initialized: {env.model.nq} joints")
    
    # Use the provided test episodes
    total_episodes = len(test_episodes)
    print(f"Evaluating on {total_episodes} pre-generated test episodes")
    
    # Load model or controller
    model = None
    controller = None
    model_type = args.controller_type
    
    if model_type == 'mpc':
        controller = MPCController(dt=0.05, prediction_horizon=20)
        print("Using MPC controller")
    else:
        print(f"Loading {model_type} model from: {args.model_path}")
        model = load_model(args.model_path, model_type)
        print(f"Model loaded successfully")
    
    # Setup rendering if requested
    viewer = None
    if args.render:
        print("Rendering enabled")
    
    # Run episodes
    print(f"\nRunning {total_episodes} episodes with {model_type} controller...")
    print("-" * 80)
    
    all_metrics = []
    
    if args.render:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for ep_idx in range(total_episodes):
                print(f"\nEpisode {ep_idx + 1}/{total_episodes}")
                target_xyz = test_episodes[ep_idx]
                metrics = run_episode(
                    env, controller, model, model_type,
                    target_xyz,
                    max_steps=args.max_steps or 150,
                    tolerance=args.tolerance,
                    render=True,
                    viewer=viewer,
                    window_size=getattr(model, 'window_size', 1)
                )
                metrics['episode'] = ep_idx
                all_metrics.append(metrics)
                print(f"  Steps: {metrics['num_steps']}, "
                      f"Reached: {metrics['reached_target']} (final: {metrics['final_ee_error']:.4f}m, tol: {args.tolerance:.3f}m), "
                      f"Mean EE error: {metrics['mean_ee_error']:.4f}m")
    else:
        for ep_idx in range(total_episodes):
            print(f"\nEpisode {ep_idx + 1}/{total_episodes}")
            target_xyz = test_episodes[ep_idx]
            metrics = run_episode(
                env, controller, model, model_type,
                target_xyz,
                max_steps=args.max_steps or 150,
                tolerance=args.tolerance,
                render=False,
                window_size=getattr(model, 'window_size', 1)
            )
            metrics['episode'] = ep_idx
            all_metrics.append(metrics)
            print(f"  Steps: {metrics['num_steps']}, "
                  f"Reached: {metrics['reached_target']} (final: {metrics['final_ee_error']:.4f}m, tol: {args.tolerance:.3f}m), "
                  f"Mean EE error: {metrics['mean_ee_error']:.4f}m")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("Aggregated Results")
    print("=" * 80)
    
    success_rate = np.mean([m['reached_target'] for m in all_metrics])
    mean_final_error = np.mean([m['final_ee_error'] for m in all_metrics])
    mean_tracking_error = np.mean([m['mean_ee_error'] for m in all_metrics])
    mean_control_effort = np.mean([m['mean_control_effort'] for m in all_metrics])
    mean_solve_time = np.mean([m['mean_solve_time'] for m in all_metrics])
    mean_action_diff = np.mean([m['mean_action_diff_from_mpc'] for m in all_metrics])
    mean_cpu_percent = np.mean([m['mean_cpu_percent'] for m in all_metrics])
    
    # Success-specific metrics
    successful_metrics = [m for m in all_metrics if m['reached_target']]
    num_successful = len(successful_metrics)
    steps_to_success = [m['num_steps'] for m in successful_metrics] if successful_metrics else []
    mean_steps_to_success = np.mean(steps_to_success) if steps_to_success else 0
    std_steps_to_success = np.std(steps_to_success) if steps_to_success else 0
    mean_steps_all_episodes = np.mean([m['num_steps'] for m in all_metrics]) if all_metrics else 0
    
    print(f"Episodes Evaluated: {len(all_metrics)}")
    if len(all_metrics) > 0:
        print(f"Success Rate: {success_rate * 100:.2f}% ({int(success_rate * len(all_metrics))}/{len(all_metrics)} reached target within {args.tolerance:.3f}m tolerance)")
        if num_successful > 0:
            print(f"Mean Steps to Success: {mean_steps_to_success:.1f} ± {std_steps_to_success:.1f} (n={num_successful} successful episodes)")
        print(f"Mean Steps (All Episodes): {mean_steps_all_episodes:.1f}")
        print(f"Mean Final EE Error: {mean_final_error:.4f}m")
        print(f"Mean Tracking Error (Average Position Error): {mean_tracking_error:.4f}m")
        print(f"Mean Control Effort: {mean_control_effort:.4f}")
        print(f"Mean Solve Time (Computational Efficiency): {mean_solve_time * 1000:.4f}ms")
        print(f"Mean Action Difference from MPC: {mean_action_diff:.4f}")
        print(f"Mean CPU Utilization (Computational Cost): {mean_cpu_percent:.2f}%")
    else:
        print("No episodes were evaluated.")
    
    # Prepare summary
    summary = {
        'controller_type': model_type,
        'model_path': args.model_path if model_type != 'mpc' else None,
        'dataset_path': None,
        'num_episodes': len(all_metrics),
        'max_steps': args.max_steps,
        'tolerance': args.tolerance,
        'timestamp': datetime.now().isoformat(),
        'aggregate_metrics': {
            'success_rate': float(success_rate),
            'num_successful_episodes': int(num_successful),
            'mean_steps_to_success': float(mean_steps_to_success),
            'std_steps_to_success': float(std_steps_to_success),
            'mean_steps_all_episodes': float(mean_steps_all_episodes),
            'mean_final_ee_error': float(mean_final_error),
            'std_final_ee_error': float(np.std([m['final_ee_error'] for m in all_metrics])),
            'mean_tracking_error': float(mean_tracking_error),
            'std_tracking_error': float(np.std([m['mean_ee_error'] for m in all_metrics])),
            'mean_control_effort': float(mean_control_effort),
            'std_control_effort': float(np.std([m['mean_control_effort'] for m in all_metrics])),
            'mean_solve_time': float(mean_solve_time),
            'std_solve_time': float(np.std([m['mean_solve_time'] for m in all_metrics])),
            'mean_action_diff_from_mpc': float(mean_action_diff),
            'std_action_diff_from_mpc': float(np.std([m['mean_action_diff_from_mpc'] for m in all_metrics])),
            'mean_cpu_percent': float(mean_cpu_percent),
            'std_cpu_percent': float(np.std([m['mean_cpu_percent'] for m in all_metrics]))
        },
        'episode_metrics': all_metrics
    }
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    controller_name = model_type if model_type == 'mpc' else os.path.splitext(os.path.basename(args.model_path))[0]
    filename = f"closed_loop_{controller_name}_{timestamp_str}.json"
    filepath = os.path.join(args.output_dir, filename)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types"""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        else:
            return obj
    
    summary = convert_to_serializable(summary)
    
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nResults saved to: {filepath}")
    print("=" * 80)
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop evaluation using dataset episodes"
    )
    
    # Controller selection
    parser.add_argument(
        '--controller-type',
        type=str,
        required=False,
        choices=['mpc', 'pytorch', 'sklearn'],
        help="Controller type. If not specified, auto-detected from model-path or defaults to MPC."
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to model file. If not specified and not using MPC, auto-selects from results/scikit_learn_baseline/models/."
    )
    # Dataset path removed: we generate test episodes on-the-fly
    
    # Episode configuration
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=1000,
        help="Number of test episodes to generate (default: 1000)"
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help="Max steps per episode (default: None, uses full episode length)"
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.03,
        help="Distance threshold for target reached in meters (default: 0.03m = 20cm, same as dataset generation)"
    )
    
    # Visualization
    parser.add_argument(
        '--render',
        action='store_true',
        default=True,
        help="Render episodes in MuJoCo viewer (default: True)"
    )
    parser.add_argument(
        '--no-render',
        action='store_false',
        dest='render',
        help="Disable MuJoCo viewer rendering"
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/closed_loop',
        help="Directory to save results (default: results/closed_loop)"
    )
    
    # Random seed
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for episode selection (default: 42)"
    )
    
    # Utility
    parser.add_argument(
        '--list-models',
        action='store_true',
        help="List all available models in results/scikit_learn_baseline/models/ and exit"
    )
    parser.add_argument(
        '--single-model',
        action='store_true',
        help="Evaluate only a single model (use with --model-path or --controller-type)"
    )
    
    args = parser.parse_args()
    
    # Interactive mode: prompt user if no arguments provided
    if len(sys.argv) == 1:  # Only script name, no arguments
        while True:  # Main menu loop
            print("\n" + "="*80)
            print("CLOSED-LOOP EVALUATION - Interactive Mode")
            print("="*80)
            print("\nWhat would you like to evaluate?")
            print("  1) MPC Controller (from src/mpc_surrogate/mpc_controller.py)")
            print("  2) Specific model(s) (select count and choose from sklearn/pytorch)")
            print("  3) All sklearn models in results/scikit_learn_baseline/models/")
            print("  4) All pytorch models in results/pytorch_comparison/results_sliding_window/models/")
            print("  5) All sklearn + pytorch models")
            print("  6) MPC + ALL learned models (for fair comparison)")
            print("  7) Exit")
            print("-"*80)
            
            choice = input("Enter choice (1-7): ").strip()
            
            if choice == '1':
                args.controller_type = 'mpc'
                args.single_model = True
                # Validate episode count
                while True:
                    num_ep = input("Number of episodes to generate (default=1000): ").strip()
                    try:
                        args.num_episodes = int(num_ep) if num_ep else 1000
                        if args.num_episodes <= 0:
                            print("Please enter a positive number of episodes.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue
                # Validate rendering input
                while True:
                    render = input("Enable rendering? (y/n, default=n): ").strip().lower()
                    if render in ['y', 'n', '']:
                        args.render = render == 'y'
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
                print(f"\n→ Running MPC evaluation with {args.num_episodes} episodes, rendering={'ON' if args.render else 'OFF'}")
                # Generate test episodes for this evaluation
                env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
                test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
                run_evaluation(args, test_episodes)
                return  # Exit after choice 1 evaluation
                
            elif choice == '2':
                # Specific model(s) selection with mixed types allowed
                while True:
                    num_models_input = input("How many specific models do you want to evaluate? (default=1, or 'back'): ").strip()
                    if num_models_input.lower() == 'back':
                        break  # Break to return to main menu
                    try:
                        num_models = int(num_models_input) if num_models_input else 1
                        if num_models < 1:
                            print("Please enter at least 1 model.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number or 'back'.")
                        continue
                
                if num_models_input.lower() == 'back':
                    continue  # Return to main menu
                
                all_available = list_available_models()  # Scans both directories
                if not all_available:
                    print(f"No models found")
                    return
                
                selected_models = []
                go_back = False
                i = 0
                while i < num_models:
                    print(f"\n--- Selecting Model {i+1}/{num_models} ---")
                    print("Available models:")
                    for idx, (name, path, mtype) in enumerate(all_available, 1):
                        # Mark already selected models
                        is_selected = any(m[1] == path for m in selected_models)
                        marker = " (already selected)" if is_selected else ""
                        print(f"  {idx}) [{mtype:8}] {name}{marker}")
                    print(f"  0) Back to main menu")
                    
                    model_choice = input(f"Select model {i+1} (1-{len(all_available)}, or 0 to go back): ").strip()
                    if model_choice == '0':
                        print("Returning to main menu...")
                        go_back = True
                        break
                    
                    try:
                        model_choice = int(model_choice)
                        if 1 <= model_choice <= len(all_available):
                            selected_model = all_available[model_choice - 1]
                            # Check if already selected
                            if any(m[1] == selected_model[1] for m in selected_models):
                                print("This model is already selected. Please choose a different one.")
                                continue  # Re-prompt for this model without incrementing i
                            else:
                                selected_models.append(selected_model)
                                i += 1  # Move to next model only if successfully selected
                        else:
                            print("Invalid selection")
                            continue  # Re-prompt without incrementing i
                    except ValueError:
                        print("Invalid input")
                        continue  # Re-prompt without incrementing i
                
                if go_back:
                    continue  # Return to main menu
                
                # Validate episodes input
                while True:
                    num_ep = input("Number of episodes per model (default=1000): ").strip()
                    try:
                        args.num_episodes = int(num_ep) if num_ep else 1000
                        if args.num_episodes <= 0:
                            print("Please enter a positive number of episodes.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue
                
                # Validate rendering input
                while True:
                    render = input("Enable rendering? (y/n, default=n): ").strip().lower()
                    if render in ['y', 'n', '']:
                        args.render = render == 'y'
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
                
                print(f"\n→ Evaluating {num_models} specific model(s) with {args.num_episodes} episodes each")
                
                # Generate test episodes once for all models
                print(f"Pre-generating {args.num_episodes} test episodes for fair comparison...")
                env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
                test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
                print(f"Episodes generated. All models will be evaluated on identical episodes.\n")
                
                for name, path, mtype in selected_models:
                    print(f"\n{'='*80}")
                    print(f"Evaluating: {name} ({mtype})")
                    print(f"{'='*80}")
                    args.model_path = path
                    args.controller_type = mtype
                    args.single_model = True
                    run_evaluation(args, test_episodes=test_episodes)
                return  # Exit after choice 2 evaluation
            
            elif choice == '3':
                # All sklearn models
                args.controller_type = 'sklearn'
                args.single_model = False
                args.model_path = None
                # Validate episodes input
                while True:
                    num_ep = input("Number of episodes per model (default=1000): ").strip()
                    try:
                        args.num_episodes = int(num_ep) if num_ep else 1000
                        if args.num_episodes <= 0:
                            print("Please enter a positive number of episodes.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue
                # Validate rendering input
                while True:
                    render = input("Enable rendering? (y/n, default=n): ").strip().lower()
                    if render in ['y', 'n', '']:
                        args.render = render == 'y'
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
                print(f"\n→ Evaluating all sklearn models with {args.num_episodes} episodes each")
                
                # Get all sklearn models and evaluate each
                available = list_available_models()  # Scans both directories
                sklearn_models = [m for m in available if m[2] == 'sklearn']
                if not sklearn_models:
                    print("No sklearn models found.")
                    continue
                
                # Generate test episodes once for all models
                print(f"Pre-generating {args.num_episodes} test episodes for fair comparison...")
                env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
                test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
                print(f"Episodes generated. All models will be evaluated on identical episodes.\n")
                
                for idx, (name, path, mtype) in enumerate(sklearn_models, 1):
                    print(f"\n{'='*80}")
                    print(f"[{idx}/{len(sklearn_models)}] Evaluating: {name} ({mtype})")
                    print(f"{'='*80}")
                    args.model_path = path
                    args.controller_type = mtype
                    args.single_model = True
                    run_evaluation(args, test_episodes=test_episodes)
                return  # Exit after choice 3 evaluation
            
            elif choice == '4':
                # All pytorch models
                args.controller_type = 'pytorch'
                args.single_model = False
                args.model_path = None
                # Validate episodes input
                while True:
                    num_ep = input("Number of episodes per model (default=1000): ").strip()
                    try:
                        args.num_episodes = int(num_ep) if num_ep else 1000
                        if args.num_episodes <= 0:
                            print("Please enter a positive number of episodes.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue
                # Validate rendering input
                while True:
                    render = input("Enable rendering? (y/n, default=n): ").strip().lower()
                    if render in ['y', 'n', '']:
                        args.render = render == 'y'
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
                print(f"\n→ Evaluating all pytorch models with {args.num_episodes} episodes each")
                
                # Get all pytorch models and evaluate each
                available = list_available_models()  # Scans both directories
                pytorch_models = [m for m in available if m[2] == 'pytorch']
                if not pytorch_models:
                    print("No pytorch models found.")
                    continue
                
                # Generate test episodes once for all models
                print(f"Pre-generating {args.num_episodes} test episodes for fair comparison...")
                env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
                test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
                print(f"Episodes generated. All models will be evaluated on identical episodes.\n")
                
                for idx, (name, path, mtype) in enumerate(pytorch_models, 1):
                    print(f"\n{'='*80}")
                    print(f"[{idx}/{len(pytorch_models)}] Evaluating: {name} ({mtype})")
                    print(f"{'='*80}")
                    args.model_path = path
                    args.controller_type = mtype
                    args.single_model = True
                    run_evaluation(args, test_episodes=test_episodes)
                return  # Exit after choice 4 evaluation
                
            elif choice == '5':
                # All sklearn + pytorch models
                args.controller_type = None
                args.single_model = False
                args.model_path = None
                # Validate episodes input
                while True:
                    num_ep = input("Number of episodes per model (default=1000): ").strip()
                    try:
                        args.num_episodes = int(num_ep) if num_ep else 1000
                        if args.num_episodes <= 0:
                            print("Please enter a positive number of episodes.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue
                # Validate rendering input
                while True:
                    render = input("Enable rendering? (y/n, default=n): ").strip().lower()
                    if render in ['y', 'n', '']:
                        args.render = render == 'y'
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
                print(f"\n→ Evaluating all sklearn + pytorch models with {args.num_episodes} episodes each")
                
                # Get all models and evaluate each
                available = list_available_models(SKLEARN_MODELS_DIR)
                if not available:
                    print("No models found.")
                    continue
                
                # Generate test episodes once for all models
                print(f"Pre-generating {args.num_episodes} test episodes for fair comparison...")
                env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
                test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
                print(f"Episodes generated. All models will be evaluated on identical episodes.\n")
                
                for idx, (name, path, mtype) in enumerate(available, 1):
                    print(f"\n{'='*80}")
                    print(f"[{idx}/{len(available)}] Evaluating: {name} ({mtype})")
                    print(f"{'='*80}")
                    args.model_path = path
                    args.controller_type = mtype
                    args.single_model = True
                    run_evaluation(args, test_episodes=test_episodes)
                return  # Exit after choice 5 evaluation
                
            elif choice == '6':
                # MPC + ALL learned models
                args.controller_type = None
                args.single_model = False
                args.model_path = None
                # Validate episodes input
                while True:
                    num_ep = input("Number of episodes per model (default=1000): ").strip()
                    try:
                        args.num_episodes = int(num_ep) if num_ep else 1000
                        if args.num_episodes <= 0:
                            print("Please enter a positive number of episodes.")
                            continue
                        break
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                        continue
                # Validate rendering input
                while True:
                    render = input("Enable rendering? (y/n, default=n): ").strip().lower()
                    if render in ['y', 'n', '']:
                        args.render = render == 'y'
                        break
                    else:
                        print("Please enter 'y' or 'n'.")
                        continue
                print(f"\n→ Evaluating MPC + all sklearn + pytorch models with {args.num_episodes} episodes each")
                
                # Get all models
                available = list_available_models(SKLEARN_MODELS_DIR)
                if not available:
                    print("No learned models found.")
                    print("Will evaluate MPC only.")
                    all_controllers = [('MPC', None, 'mpc')]
                else:
                    all_controllers = [('MPC', None, 'mpc')] + available
                
                print(f"\nTotal controllers to evaluate: {len(all_controllers)}")
                print(f"  - MPC (baseline)")
                print(f"  - {len(available)} learned models")
                
                # Generate test episodes once for all models
                print(f"\nPre-generating {args.num_episodes} test episodes for fair comparison...")
                env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
                test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
                print(f"Episodes generated. All controllers will be evaluated on identical episodes.\n")
                
                for idx, (name, path, mtype) in enumerate(all_controllers, 1):
                    print(f"\n{'='*80}")
                    print(f"[{idx}/{len(all_controllers)}] Evaluating: {name} ({mtype})")
                    print(f"{'='*80}")
                    args.model_path = path
                    args.controller_type = mtype
                    args.single_model = True
                    run_evaluation(args, test_episodes=test_episodes)
                
                print(f"\n{'='*80}")
                print(f"Completed evaluation of MPC + all {len(available)} learned models")
                print(f"Results saved to: {args.output_dir}")
                print(f"{'='*80}")
                return  # Exit after choice 6 evaluation
                
            elif choice == '7':
                print("Exiting...")
                return
            else:
                print("Invalid choice. Please select 1-7.")
                continue
    
    # Handle --list-models flag
    if args.list_models:
        available = list_available_models(SKLEARN_MODELS_DIR)
        if available:
            print("\nAvailable models (in results/scikit_learn_baseline/models/):")
            print("-" * 80)
            for name, path, mtype in available:
                print(f"  [{mtype:8}] {name:<45} → {path}")
            print("-" * 80)
            print(f"\nTotal: {len(available)} models")
            print(f"\nUsage examples:")
            print(f"  python scripts/closed_loop_eval.py  # Evaluates all models by default")
            print(f"  python scripts/closed_loop_eval.py --model-path \"{available[0][1]}\" --single-model")
        else:
            print("No models found in results/models/")
            print("Expected file types: .pkl (scikit-learn), .pt/.pth (PyTorch)")
        return
    
    # Default behavior: evaluate ALL models unless --single-model is specified
    if not args.single_model and args.model_path is None and args.controller_type is None:
        available = list_available_models(SKLEARN_MODELS_DIR)
        if not available:
            print("No models found in results/scikit_learn_baseline/models/")
            print("Expected file types: .pkl (scikit-learn), .pt/.pth (PyTorch)")
            return
        
        print(f"\n{'='*80}")
        print(f"Evaluating ALL {len(available)} models in results/scikit_learn_baseline/models/")
        print(f"{'='*80}")
        print(f"Episodes per model: {args.num_episodes if args.num_episodes else 'ALL (1715)'}")
        print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
        print(f"{'='*80}\n")
        
        # Generate test episodes once for all models
        print(f"Pre-generating {args.num_episodes} test episodes for fair comparison...")
        env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
        test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
        print(f"Episodes generated. All models will be evaluated on identical episodes.\n")
        
        for idx, (name, path, mtype) in enumerate(available, 1):
            print(f"\n{'='*80}")
            print(f"[{idx}/{len(available)}] Evaluating: {name} ({mtype})")
            print(f"{'='*80}")
            args.model_path = path
            args.controller_type = mtype
            run_evaluation(args, test_episodes=test_episodes)
        
        print(f"\n{'='*80}")
        print(f"Completed evaluation of all {len(available)} models")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*80}")
        return
    
    # Single model evaluation
    if args.controller_type is None and args.model_path is None:
        print("ERROR: No model specified.")
        print("  - Remove --single-model flag to evaluate all models (default behavior)")
        print("  - Or specify --model-path or --controller-type for single model evaluation")
        return
    elif args.controller_type is None and args.model_path is not None:
        # Auto-detect from file extension
        if args.model_path.endswith('.pkl'):
            args.controller_type = 'sklearn'
        elif args.model_path.endswith('.pt') or args.model_path.endswith('.pth'):
            args.controller_type = 'pytorch'
        else:
            parser.error("Cannot auto-detect model type from extension. Please specify --controller-type")
        print(f"Auto-detected model type: {args.controller_type}")
    elif args.model_path is None and args.controller_type is not None:
        # MPC doesn't need a model file - it's in src/mpc_surrogate
        if args.controller_type == 'mpc':
            # MPC controller is loaded from src/mpc_surrogate/mpc_controller.py
            print(f"Using MPC controller from src/mpc_surrogate/mpc_controller.py")
            # args.model_path stays None, which is fine for MPC
        else:
            # Look for learned models in results/scikit_learn_baseline/models/
            available = list_available_models(SKLEARN_MODELS_DIR)
            matching = [m for m in available if m[2] == args.controller_type]
            if matching:
                if args.single_model:
                    args.model_path = matching[0][1]
                    print(f"Auto-selected model: {matching[0][0]}")
                else:
                    # Evaluate all models of this type
                    print(f"\nEvaluating all {len(matching)} {args.controller_type} models...")
                    
                    # Generate test episodes once for all models
                    print(f"Pre-generating {args.num_episodes} test episodes for fair comparison...")
                    env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
                    test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
                    print(f"Episodes generated. All models will be evaluated on identical episodes.\n")
                    
                    for name, path, mtype in matching:
                        print(f"\n{'='*80}")
                        print(f"Evaluating: {name} ({mtype})")
                        print(f"{'='*80}")
                        args.model_path = path
                        run_evaluation(args, test_episodes=test_episodes)
                    return
            else:
                print(f"ERROR: No {args.controller_type} models found in results/scikit_learn_baseline/models/.")
                print(f"Use --list-models to see available models.")
                return
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate test episodes for single model evaluation
    print(f"Generating {args.num_episodes} test episodes...")
    env_temp = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    test_episodes = [sample_3d_cartesian_target(env_temp) for _ in range(args.num_episodes)]
    
    # Run evaluation on generated episodes
    run_evaluation(args, test_episodes)


if __name__ == "__main__":
    main()
