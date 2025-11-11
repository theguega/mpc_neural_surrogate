
import h5py
import numpy as np
from tqdm import tqdm

from mpc_surrogate.mpc_controller import MPCController
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics

# episode length for use globally
episode_length = 50


def sample_reachable_target(env, link_lengths=(0.3, 0.3, 0.25), max_attempts=50):
    """
    Sample a reachable target position using rejection sampling.
    Validates reachability by attempting IK and checking the solution quality.
    
    Args:
        env: MuJoCo environment (needed for IK validation)
        link_lengths: Tuple of (link1, link2, link3) lengths
        max_attempts: Maximum number of sampling attempts before giving up
    
    Returns:
        np.array: A validated reachable target position [x, y, z]
    """
    LINK1, LINK2, LINK3 = link_lengths
    BASE_HEIGHT = 0.1  # Base platform height
    
    # Conservative workspace bounds based on robot geometry and joint limits
    # joint2 range: [-1.57, 1.57] (±90°), joint3 range: [-2.0, 2.0]
    MAX_REACH = LINK1 + LINK2 + LINK3 - 0.15  # Leave margin for joint limits
    MIN_REACH = 0.225  # Avoid singularities near base
    
    for attempt in range(max_attempts):
        # Sample in cylindrical coordinates for uniform workspace coverage around the robot
        # Allow full 360° around the base (theta in [0, 2pi))
        r_xy = np.random.uniform(MIN_REACH, MAX_REACH * 0.95)
        theta = np.random.uniform(0, 2 * np.pi)

        # Z should be reachable: from slightly above base to reasonable maximum height
        z_min = BASE_HEIGHT + 0.08  # Allow lower targets closer to base but avoid ground contact
        z_max = BASE_HEIGHT + LINK1 + LINK2 * 0.95  # Conservative vertical limit near max reach
        z = np.random.uniform(z_min, min(z_max, 0.7))
        
        target_xyz = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z])
        
        # Validate with IK - check if solution converges to actual target
        try:
            joint_solution = solve_inverse_kinematics(env, target_xyz, max_iters=200, tol=5e-3)
            
            # Verify the IK solution actually reaches the target
            tmp_data = env.data  # Use a temporary copy to test
            original_qpos = tmp_data.qpos.copy()
            
            tmp_data.qpos[:3] = joint_solution
            import mujoco
            mujoco.mj_forward(env.model, tmp_data)
            
            site_id = env.model.site("ee_site").id
            achieved_pos = tmp_data.site_xpos[site_id].copy()
            error = np.linalg.norm(achieved_pos - target_xyz)
            
            # Restore original state
            tmp_data.qpos[:] = original_qpos
            mujoco.mj_forward(env.model, tmp_data)
            
            # Accept if error is small (IK successfully reached the target)
            if error < 0.01:  # 1cm tolerance
                return target_xyz
                
        except Exception:
            continue  # Try again if IK fails
    
    # Fallback to a known safe position if all attempts fail
    print(f"Warning: Could not find reachable target after {max_attempts} attempts. Using safe default.")
    return np.array([-0.4, 0.1, 0.4])


def generate_data(num_episodes=100, episode_length=150, filename="data/robot_mpc_dataset.h5"):
    """
    Generate dataset for learning an MPC controller.
    Inputs  : [joint positions + joint velocities]
    Targets : end-effector (EE) target positions
    Actions : total torques (MPC + static bias)
    """

    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    controller = MPCController(dt=0.05, prediction_horizon=20)
    n_sim_steps_per_mpc_step = int(controller.dt / env.model.opt.timestep)

    all_states, all_targets, all_actions = [], [], []

    print(f"--- Generating {num_episodes} episodes ---")
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        obs = env.reset()
        target_joint_pos = obs[:3].copy()
        # generate the first target of the episode
        target_xyz = sample_reachable_target(env)
        target_joint_pos = solve_inverse_kinematics(env, target_xyz)

        for step in tqdm(range(episode_length), desc="Steps", leave=False):
            current_state = obs[:6]  # [q1, q2, q3, q̇1, q̇2, q̇3]

            tau_mpc, solved = controller.solve(current_state, target_joint_pos)

            if solved:
                tau_static = env.data.qfrc_bias.copy()
                total_tau = tau_mpc + tau_static

                all_states.append(current_state)
                all_targets.append(target_xyz)
                # we want learn the MPC policy, not the torque from MPC + the static torque given by MuJoCo
                all_actions.append(tau_mpc)

                for _ in range(n_sim_steps_per_mpc_step):
                    obs, _, _, _ = env.step(total_tau)
            else:
                # if MPC fails, we don't add it to the dataset
                pass

    # save dataset
    with h5py.File(filename, "w") as f:
        f.create_dataset("states", data=np.array(all_states), compression="gzip")
        f.create_dataset("targets", data=np.array(all_targets), compression="gzip")
        f.create_dataset("actions", data=np.array(all_actions), compression="gzip")

    print(f"--- Data Generation Complete ---\nSaved to: {filename}")
    print(f"States:  {np.array(all_states).shape}")
    print(f"Targets: {np.array(all_targets).shape}")
    print(f"Actions: {np.array(all_actions).shape}")


if __name__ == "__main__":
    generate_data(num_episodes=20, episode_length=episode_length)
