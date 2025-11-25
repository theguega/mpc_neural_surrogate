"""
Closed-loop evaluation script for comparing MPC controller with learned surrogate models.

This script runs multiple episodes where the robot tracks randomly generated targets using either:
- The MPC controller (ground truth)
- A trained surrogate model (PyTorch .pt or scikit-learn .pkl)

Metrics are computed for each episode and aggregated at the end.
"""

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import mujoco
import mujoco.viewer
import numpy as np
from mpc_surrogate.mpc_controller import MPCController
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics


def load_model(model_path, model_type):
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the model file (.pt for PyTorch, .pkl for scikit-learn)
        model_type: 'pytorch' or 'sklearn'
    
    Returns:
        Loaded model object
    """
    if model_type == 'pytorch':
        import torch
        model = torch.load(model_path)
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


def generate_random_target(env, workspace_bounds=None):
    """
    Generate a random target position within the robot's workspace.
    
    Args:
        env: MuJoCoEnvironment instance
        workspace_bounds: Optional dict with 'x', 'y', 'z' keys containing (min, max) tuples
    
    Returns:
        target_pos: 3D position in workspace
        target_joints: Joint angles to reach that position (from IK)
    """
    if workspace_bounds is None:
        # Default workspace bounds based on 3-DOF robot arm
        workspace_bounds = {
            'x': (-0.6, -0.2),
            'y': (-0.4, 0.4),
            'z': (0.3, 0.6)
        }
    
    # Generate random position
    target_pos = np.array([
        np.random.uniform(*workspace_bounds['x']),
        np.random.uniform(*workspace_bounds['y']),
        np.random.uniform(*workspace_bounds['z'])
    ])
    
    # Solve IK to get joint target
    success, target_joints = solve_inverse_kinematics(env, target_pos)
    
    if not success:
        print(f"Warning: IK failed for target {target_pos}, using approximate solution")
    
    return target_pos, target_joints


def run_episode(env, controller, model, model_type, target_pos, target_joints, 
                max_steps=200, tolerance=0.02, render=False, viewer=None):
    """
    Run a single episode where the robot tracks a target.
    
    Args:
        env: MuJoCoEnvironment instance
        controller: MPCController instance (can be None if using model)
        model: Trained model (can be None if using MPC)
        model_type: 'pytorch', 'sklearn', or 'mpc'
        target_pos: Target position in workspace
        target_joints: Target joint angles
        max_steps: Maximum steps per episode
        tolerance: Distance threshold to consider target reached
        render: Whether to render the episode
        viewer: MuJoCo viewer instance (if rendering)
    
    Returns:
        Dictionary containing episode metrics
    """
    n_sim_steps_per_mpc_step = int(0.05 / env.model.opt.timestep)
    
    # Reset to starting position
    obs = env.reset()
    env.data.qpos[:3] = np.zeros(3)
    env.data.qvel[:3] = np.zeros(3)
    mujoco.mj_forward(env.model, env.data)
    
    # Update target visualization
    if render and viewer is not None:
        target_body_id = env.model.body("target").id
        env.model.body_mocapid[target_body_id] = 0
        env.data.mocap_pos[0] = target_pos
        mujoco.mj_forward(env.model, env.data)
    
    # Episode tracking
    ee_errors = []
    joint_errors = []
    control_efforts = []
    solve_times = []
    reached_target = False
    steps_to_target = max_steps
    
    for step in range(max_steps):
        current_state = obs[:6]  # [q, q_dot]
        current_ee_pos = env.get_ee_position()
        ee_error = np.linalg.norm(current_ee_pos - target_pos)
        joint_error = np.linalg.norm(current_state[:3] - target_joints)
        
        ee_errors.append(ee_error)
        joint_errors.append(joint_error)
        
        # Check if target reached
        if ee_error < tolerance and not reached_target:
            reached_target = True
            steps_to_target = step
        
        # Compute control action
        start_time = time.time()
        
        if model_type == 'mpc':
            tau, solved = controller.solve(current_state, target_joints)
            if not solved:
                tau = np.zeros(3)
        else:
            tau = predict_action(model, current_state, target_joints, model_type)
        
        solve_time = time.time() - start_time
        solve_times.append(solve_time)
        
        control_effort = np.linalg.norm(tau)
        control_efforts.append(control_effort)
        
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
    metrics = {
        'reached_target': reached_target,
        'steps_to_target': steps_to_target,
        'final_ee_error': ee_errors[-1],
        'final_joint_error': joint_errors[-1],
        'mean_ee_error': np.mean(ee_errors),
        'std_ee_error': np.std(ee_errors),
        'max_ee_error': np.max(ee_errors),
        'mean_joint_error': np.mean(joint_errors),
        'std_joint_error': np.std(joint_errors),
        'mean_control_effort': np.mean(control_efforts),
        'std_control_effort': np.std(control_efforts),
        'total_control_effort': np.sum(control_efforts),
        'mean_solve_time': np.mean(solve_times),
        'std_solve_time': np.std(solve_times),
        'total_time': np.sum(solve_times)
    }
    
    return metrics


def run_evaluation(args):
    """
    Main evaluation loop: run multiple episodes and aggregate results.
    """
    print("=" * 80)
    print("Closed-Loop Evaluation")
    print("=" * 80)
    
    # Initialize environment
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    print(f"Environment initialized: {env.model.nq} joints")
    
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
    print(f"\nRunning {args.num_episodes} episodes...")
    print("-" * 80)
    
    all_metrics = []
    
    if args.render:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for ep in range(args.num_episodes):
                print(f"\nEpisode {ep + 1}/{args.num_episodes}")
                
                # Generate random target
                target_pos, target_joints = generate_random_target(env)
                print(f"  Target position: {target_pos}")
                print(f"  Target joints: {target_joints}")
                
                # Run episode
                metrics = run_episode(
                    env, controller, model, model_type,
                    target_pos, target_joints,
                    max_steps=args.max_steps,
                    tolerance=args.tolerance,
                    render=True,
                    viewer=viewer
                )
                
                metrics['episode'] = ep
                metrics['target_pos'] = target_pos.tolist()
                metrics['target_joints'] = target_joints.tolist()
                all_metrics.append(metrics)
                
                # Print episode summary
                print(f"  Reached: {metrics['reached_target']}, "
                      f"Steps: {metrics['steps_to_target']}, "
                      f"Final error: {metrics['final_ee_error']:.4f}m")
    else:
        for ep in range(args.num_episodes):
            print(f"\nEpisode {ep + 1}/{args.num_episodes}")
            
            # Generate random target
            target_pos, target_joints = generate_random_target(env)
            print(f"  Target position: {target_pos}")
            print(f"  Target joints: {target_joints}")
            
            # Run episode
            metrics = run_episode(
                env, controller, model, model_type,
                target_pos, target_joints,
                max_steps=args.max_steps,
                tolerance=args.tolerance,
                render=False
            )
            
            metrics['episode'] = ep
            metrics['target_pos'] = target_pos.tolist()
            metrics['target_joints'] = target_joints.tolist()
            all_metrics.append(metrics)
            
            # Print episode summary
            print(f"  Reached: {metrics['reached_target']}, "
                  f"Steps: {metrics['steps_to_target']}, "
                  f"Final error: {metrics['final_ee_error']:.4f}m")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("Aggregated Results")
    print("=" * 80)
    
    success_rate = np.mean([m['reached_target'] for m in all_metrics])
    mean_steps = np.mean([m['steps_to_target'] for m in all_metrics if m['reached_target']])
    mean_final_error = np.mean([m['final_ee_error'] for m in all_metrics])
    mean_tracking_error = np.mean([m['mean_ee_error'] for m in all_metrics])
    mean_control_effort = np.mean([m['mean_control_effort'] for m in all_metrics])
    mean_solve_time = np.mean([m['mean_solve_time'] for m in all_metrics])
    
    print(f"Success Rate: {success_rate * 100:.2f}% ({int(success_rate * args.num_episodes)}/{args.num_episodes})")
    if success_rate > 0:
        print(f"Mean Steps to Target: {mean_steps:.2f}")
    print(f"Mean Final EE Error: {mean_final_error:.4f}m")
    print(f"Mean Tracking Error: {mean_tracking_error:.4f}m")
    print(f"Mean Control Effort: {mean_control_effort:.4f}")
    print(f"Mean Solve Time: {mean_solve_time * 1000:.4f}ms")
    
    # Prepare summary
    summary = {
        'controller_type': model_type,
        'model_path': args.model_path if model_type != 'mpc' else None,
        'num_episodes': args.num_episodes,
        'max_steps': args.max_steps,
        'tolerance': args.tolerance,
        'timestamp': datetime.now().isoformat(),
        'aggregate_metrics': {
            'success_rate': float(success_rate),
            'mean_steps_to_target': float(mean_steps) if success_rate > 0 else None,
            'mean_final_ee_error': float(mean_final_error),
            'std_final_ee_error': float(np.std([m['final_ee_error'] for m in all_metrics])),
            'mean_tracking_error': float(mean_tracking_error),
            'std_tracking_error': float(np.std([m['mean_ee_error'] for m in all_metrics])),
            'mean_control_effort': float(mean_control_effort),
            'std_control_effort': float(np.std([m['mean_control_effort'] for m in all_metrics])),
            'mean_solve_time': float(mean_solve_time),
            'std_solve_time': float(np.std([m['mean_solve_time'] for m in all_metrics]))
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
        description="Closed-loop evaluation of MPC controller vs surrogate models"
    )
    
    # Controller selection
    parser.add_argument(
        '--controller-type',
        type=str,
        required=True,
        choices=['mpc', 'pytorch', 'sklearn'],
        help="Controller type: 'mpc' for ground truth, 'pytorch' for .pt model, 'sklearn' for .pkl model"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help="Path to trained model file (.pt for PyTorch, .pkl for scikit-learn). Required if not using MPC."
    )
    
    # Episode configuration
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help="Number of episodes to run (default: 10)"
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)"
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.02,
        help="Distance threshold to consider target reached in meters (default: 0.02)"
    )
    
    # Visualization
    parser.add_argument(
        '--render',
        action='store_true',
        help="Render episodes in MuJoCo viewer"
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
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.controller_type != 'mpc' and args.model_path is None:
        parser.error("--model-path is required when using a learned controller")
    
    if args.controller_type == 'pytorch' and not args.model_path.endswith('.pt'):
        print("Warning: PyTorch model file should have .pt extension")
    
    if args.controller_type == 'sklearn' and not args.model_path.endswith('.pkl'):
        print("Warning: Scikit-learn model file should have .pkl extension")
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()
