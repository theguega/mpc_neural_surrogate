"""
Closed-loop evaluation script for comparing MPC controller with learned surrogate models.

This script runs episodes from the HDF5 dataset where the robot tracks targets using either:
- The MPC controller (ground truth)
- A trained surrogate model (PyTorch .pt/.pth or scikit-learn .pkl)

Models are automatically discovered from results/models/ directory.
Episodes are loaded from data/robot_mpc_dataset.h5.

Metrics are computed for each episode and aggregated at the end, including
comparison with the original MPC actions from the dataset.
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

# PyTorch model architecture (matches notebook MLP)
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
            # State dict was saved - instantiate MLP and load weights
            print("Detected state_dict, instantiating MLP architecture...")
            model = MLP(input_dim=9, hidden_dims=[128, 64], output_dim=3)
            model.load_state_dict(loaded_obj)
            print("Loaded state_dict into MLP")
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


def predict_action(model, state, target, model_type):
    """
    Predict control action using the loaded model.
    
    Args:
        model: Loaded model object
        state: Current robot state (6D: [q, q_dot])
        target: Target joint positions (3D)
        model_type: 'pytorch' or 'sklearn'
    
    Returns:
        Predicted control action (3D torque vector)
    """
    # Construct input: [state, target]
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


def list_available_models(models_dir='results/models'):
    """
    List all available model files in a directory (.pkl, .pt, .pth).
    
    Args:
        models_dir: Directory to search for model files
    
    Returns:
        List of tuples (model_name, filepath, model_type)
    """
    import glob
    
    if not os.path.exists(models_dir):
        return []
    
    models = []
    
    # Find pickle files
    pkl_files = glob.glob(os.path.join(models_dir, '*.pkl'))
    models.extend([(os.path.basename(f), f, 'sklearn') for f in sorted(pkl_files)])
    
    # Find PyTorch files
    pt_files = glob.glob(os.path.join(models_dir, '*.pt'))
    models.extend([(os.path.basename(f), f, 'pytorch') for f in sorted(pt_files)])
    
    pth_files = glob.glob(os.path.join(models_dir, '*.pth'))
    models.extend([(os.path.basename(f), f, 'pytorch') for f in sorted(pth_files)])
    
    return sorted(models)


def load_dataset_episodes(dataset_path='data/robot_mpc_dataset.h5', num_episodes=None, seed=42):
    """
    Load episodes from the HDF5 dataset.
    
    Args:
        dataset_path: Path to HDF5 dataset
        num_episodes: Number of episodes to load (None = all)
        seed: Random seed for episode selection
    
    Returns:
        List of episode data dicts with 'name', 'states', 'targets', 'actions'
    """
    np.random.seed(seed)
    
    with h5py.File(dataset_path, 'r') as f:
        episodes = f['episodes']
        ep_keys = sorted(list(episodes.keys()))
        
        # Select episodes
        if num_episodes is None or num_episodes >= len(ep_keys):
            selected_keys = ep_keys
        else:
            selected_keys = sorted(np.random.choice(ep_keys, size=num_episodes, replace=False))
        
        # Load episode data
        episode_data = []
        for ep_key in selected_keys:
            ep = episodes[ep_key]
            episode_data.append({
                'name': ep_key,
                'states': ep['states'][:],
                'targets': ep['targets'][:],
                'actions': ep['actions'][:]
            })
    
    return episode_data


def run_episode(env, controller, model, model_type, episode_data, 
                max_steps=None, tolerance=0.2, render=False, viewer=None):
    """
    Run a single episode where the robot tracks targets from the dataset.
    
    Args:
        env: MuJoCoEnvironment instance
        controller: MPCController instance (can be None if using model)
        model: Trained model (can be None if using MPC)
        model_type: 'pytorch', 'sklearn', or 'mpc'
        episode_data: Dict with 'name', 'states', 'targets', 'actions' from dataset
        max_steps: Maximum steps per episode (None = use episode length)
        tolerance: Distance threshold to consider target reached
        render: Whether to render the episode
        viewer: MuJoCo viewer instance (if rendering)
    
    Returns:
        Dictionary containing episode metrics
    """
    n_sim_steps_per_mpc_step = int(0.05 / env.model.opt.timestep)
    
    # Get episode length from dataset
    episode_length = len(episode_data['states'])
    if max_steps is None:
        max_steps = episode_length
    
    # Reset to starting position (use first state from dataset)
    obs = env.reset()
    initial_state = episode_data['states'][0]
    env.data.qpos[:3] = initial_state[:3]
    env.data.qvel[:3] = initial_state[3:]
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
    
    # Get process for CPU tracking
    process = psutil.Process()
    
    for step in range(min(max_steps, episode_length)):
        current_state = obs[:6]  # [q, q_dot]
        current_ee_pos = env.get_ee_position()
        
        # Get target from dataset for this timestep
        target_pos = episode_data['targets'][step]
        # Use target joints from dataset (MPC's target)
        target_joints = episode_data['states'][step][:3] if step < episode_length else current_state[:3]
        
        # Update target visualization
        if render and viewer is not None:
            env.data.mocap_pos[0] = target_pos
        
        ee_error = np.linalg.norm(current_ee_pos - target_pos)
        joint_error = np.linalg.norm(current_state[:3] - target_joints)
        
        ee_errors.append(ee_error)
        joint_errors.append(joint_error)
        
        # Compute control action
        start_time = time.time()
        cpu_before = process.cpu_percent()
        
        if model_type == 'mpc':
            tau, solved = controller.solve(current_state, target_joints)
            if not solved:
                tau = np.zeros(3)
        else:
            tau = predict_action(model, current_state, target_joints, model_type)
        
        solve_time = time.time() - start_time
        cpu_after = process.cpu_percent()
        
        solve_times.append(solve_time)
        cpu_percentages.append((cpu_before + cpu_after) / 2)
        
        control_effort = np.linalg.norm(tau)
        control_efforts.append(control_effort)
        
        # Compare with MPC action from dataset
        if step < len(episode_data['actions']):
            mpc_action = episode_data['actions'][step]
            action_diff = np.linalg.norm(tau - mpc_action)
            action_differences.append(action_diff)
        
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
        'episode_name': episode_data['name'],
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


def run_evaluation(args):
    """
    Main evaluation loop: run multiple episodes and aggregate results.
    """
    print("=" * 80)
    print("Closed-Loop Evaluation (Dataset-based)")
    print("=" * 80)
    
    # Initialize environment
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    print(f"Environment initialized: {env.model.nq} joints")
    
    # Load episodes from dataset
    print(f"Loading episodes from: {args.dataset_path}")
    episodes = load_dataset_episodes(args.dataset_path, args.num_episodes, args.seed)
    print(f"Loaded {len(episodes)} episodes from dataset")
    
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
    print(f"\nRunning {len(episodes)} episodes with {model_type} controller...")
    print("-" * 80)
    
    all_metrics = []
    
    if args.render:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for ep_idx, episode_data in enumerate(episodes):
                print(f"\nEpisode {ep_idx + 1}/{len(episodes)} ({episode_data['name']})")
                
                # Run episode
                metrics = run_episode(
                    env, controller, model, model_type,
                    episode_data,
                    max_steps=args.max_steps,
                    tolerance=args.tolerance,
                    render=True,
                    viewer=viewer
                )
                
                metrics['episode'] = ep_idx
                all_metrics.append(metrics)
                
                # Print episode summary
                print(f"  Steps: {metrics['num_steps']}, "
                      f"Reached: {metrics['reached_target']} (final: {metrics['final_ee_error']:.4f}m, tol: {args.tolerance:.3f}m), "
                      f"Mean EE error: {metrics['mean_ee_error']:.4f}m, "
                      f"Action diff from MPC: {metrics['mean_action_diff_from_mpc']:.4f}")
    else:
        for ep_idx, episode_data in enumerate(episodes):
            print(f"\nEpisode {ep_idx + 1}/{len(episodes)} ({episode_data['name']})")
            
            # Run episode
            metrics = run_episode(
                env, controller, model, model_type,
                episode_data,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
                render=False
            )
            
            metrics['episode'] = ep_idx
            all_metrics.append(metrics)
            
            # Print episode summary
            print(f"  Steps: {metrics['num_steps']}, "
                  f"Reached: {metrics['reached_target']} (final: {metrics['final_ee_error']:.4f}m, tol: {args.tolerance:.3f}m), "
                  f"Mean EE error: {metrics['mean_ee_error']:.4f}m, "
                  f"Action diff from MPC: {metrics['mean_action_diff_from_mpc']:.4f}")
    
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
    
    print(f"Episodes Evaluated: {len(all_metrics)}")
    print(f"Success Rate: {success_rate * 100:.2f}% ({int(success_rate * len(all_metrics))}/{len(all_metrics)} reached target within {args.tolerance:.3f}m tolerance)")
    print(f"Mean Final EE Error: {mean_final_error:.4f}m")
    print(f"Mean Tracking Error (Average Position Error): {mean_tracking_error:.4f}m")
    print(f"Mean Control Effort: {mean_control_effort:.4f}")
    print(f"Mean Solve Time (Computational Efficiency): {mean_solve_time * 1000:.4f}ms")
    print(f"Mean Action Difference from MPC: {mean_action_diff:.4f}")
    print(f"Mean CPU Utilization (Computational Cost): {mean_cpu_percent:.2f}%")
    
    # Prepare summary
    summary = {
        'controller_type': model_type,
        'model_path': args.model_path if model_type != 'mpc' else None,
        'dataset_path': args.dataset_path,
        'num_episodes': len(all_metrics),
        'max_steps': args.max_steps,
        'tolerance': args.tolerance,
        'timestamp': datetime.now().isoformat(),
        'aggregate_metrics': {
            'success_rate': float(success_rate),
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
        help="Path to model file. If not specified and not using MPC, auto-selects from results/models/."
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='data/robot_mpc_dataset.h5',
        help="Path to HDF5 dataset (default: data/robot_mpc_dataset.h5)"
    )
    
    # Episode configuration
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=None,
        help="Number of episodes to load from dataset (default: None, loads all episodes)"
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
        default=0.2,
        help="Distance threshold for target reached in meters (default: 0.2m = 20cm, same as dataset generation)"
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
        help="List all available models in results/models/ and exit"
    )
    parser.add_argument(
        '--single-model',
        action='store_true',
        help="Evaluate only a single model (use with --model-path or --controller-type)"
    )
    
    args = parser.parse_args()
    
    # Interactive mode: prompt user if no arguments provided
    if len(sys.argv) == 1:  # Only script name, no arguments
        print("\n" + "="*80)
        print("CLOSED-LOOP EVALUATION - Interactive Mode")
        print("="*80)
        print("\nWhat would you like to evaluate?")
        print("  1) MPC Controller (from src/mpc_surrogate/mpc_controller.py)")
        print("  2) All learned models in results/models/")
        print("  3) Specific learned model (select from list)")
        print("  4) All sklearn models")
        print("  5) Exit")
        print("-"*80)
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            args.controller_type = 'mpc'
            args.single_model = True
            num_ep = input("Number of episodes (default=10, 'all' for all 1715): ").strip()
            args.num_episodes = None if num_ep.lower() == 'all' else (int(num_ep) if num_ep else 10)
            render = input("Enable rendering? (y/n, default=n): ").strip().lower()
            args.render = render == 'y'
            print(f"\n→ Running MPC evaluation with {args.num_episodes or 'all'} episodes, rendering={'ON' if args.render else 'OFF'}")
            
        elif choice == '2':
            args.single_model = False
            args.controller_type = None
            args.model_path = None
            num_ep = input("Number of episodes per model (default=all, or enter number): ").strip()
            args.num_episodes = None if not num_ep else int(num_ep)
            render = input("Enable rendering? (y/n, default=n): ").strip().lower()
            args.render = render == 'y'
            print(f"\n→ Evaluating all models with {args.num_episodes or 'all'} episodes each")
            
        elif choice == '3':
            available = list_available_models('results/models')
            if not available:
                print("\nNo models found in results/models/")
                return
            print("\nAvailable models:")
            for idx, (name, path, mtype) in enumerate(available, 1):
                print(f"  {idx}) [{mtype:8}] {name}")
            model_choice = int(input(f"\nSelect model (1-{len(available)}): ").strip())
            if 1 <= model_choice <= len(available):
                name, path, mtype = available[model_choice - 1]
                args.model_path = path
                args.controller_type = mtype
                args.single_model = True
                num_ep = input("Number of episodes (default=10, 'all' for all): ").strip()
                args.num_episodes = None if num_ep.lower() == 'all' else (int(num_ep) if num_ep else 10)
                render = input("Enable rendering? (y/n, default=n): ").strip().lower()
                args.render = render == 'y'
                print(f"\n→ Evaluating {name} with {args.num_episodes or 'all'} episodes")
            else:
                print("Invalid selection")
                return
                
        elif choice == '4':
            args.controller_type = 'sklearn'
            args.single_model = False
            args.model_path = None
            num_ep = input("Number of episodes per model (default=all, or enter number): ").strip()
            args.num_episodes = None if not num_ep else int(num_ep)
            render = input("Enable rendering? (y/n, default=n): ").strip().lower()
            args.render = render == 'y'    
            print(f"\n→ Evaluating all sklearn models with {args.num_episodes or 'all'} episodes each")
            
        elif choice == '5':
            print("Exiting...")
            return
        else:
            print("Invalid choice")
            return
    
    # Handle --list-models flag
    if args.list_models:
        available = list_available_models('results/models')
        if available:
            print("\nAvailable models (in results/models/):")
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
        available = list_available_models('results/models')
        if not available:
            print("No models found in results/models/")
            print("Expected file types: .pkl (scikit-learn), .pt/.pth (PyTorch)")
            return
        
        print(f"\n{'='*80}")
        print(f"Evaluating ALL {len(available)} models in results/models/")
        print(f"{'='*80}")
        print(f"Episodes per model: {args.num_episodes if args.num_episodes else 'ALL (1715)'}")
        print(f"Rendering: {'Enabled' if args.render else 'Disabled'}")
        print(f"{'='*80}\n")
        
        for idx, (name, path, mtype) in enumerate(available, 1):
            print(f"\n{'='*80}")
            print(f"[{idx}/{len(available)}] Evaluating: {name} ({mtype})")
            print(f"{'='*80}")
            args.model_path = path
            args.controller_type = mtype
            run_evaluation(args)
        
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
            # Look for learned models in results/models/
            available = list_available_models('results/models')
            matching = [m for m in available if m[2] == args.controller_type]
            if matching:
                if args.single_model:
                    args.model_path = matching[0][1]
                    print(f"Auto-selected model: {matching[0][0]}")
                else:
                    # Evaluate all models of this type
                    print(f"\nEvaluating all {len(matching)} {args.controller_type} models...")
                    for name, path, mtype in matching:
                        print(f"\n{'='*80}")
                        print(f"Evaluating: {name} ({mtype})")
                        print(f"{'='*80}")
                        args.model_path = path
                        run_evaluation(args)
                    return
            else:
                print(f"ERROR: No {args.controller_type} models found in results/models/.")
                print(f"Use --list-models to see available models.")
                return
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()
