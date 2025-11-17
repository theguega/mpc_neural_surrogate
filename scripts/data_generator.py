import h5py
import numpy as np
from mpc_surrogate.mpc_controller import MPCController
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics
from tqdm import tqdm

TOL = 0.02  # 2cm tolerance


def sample_3d_cartesian_target(env):
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
    # the base platform is 0.05m high
    base = 0.05

    # first link is a 0.3m cylinder which can be rotated from -pi to pi around the z axis
    l1 = 0.3
    # angle1 = [-np.pi, np.pi]

    # second link is a 0.3 capsule which can be rotated from -pi/2 to pi/2 around the y axis
    l2 = 0.3
    # angle2 = [-np.pi / 2, np.pi / 2]

    # third link is a 0.25m cylinder which can be rotated from -2 to 2 around the z axis
    # l3 = 0.25
    # angle3 = [-2, 2]

    max_radius = l2 + l2 - 0.15
    min_radius = 0.25

    for _ in range(100):
        # sample in cylindrical coordinates for uniform workspace coverage around the robot
        r_xy = np.random.uniform(min_radius, max_radius)
        theta = np.random.uniform(0, 2 * np.pi)

        # z should be reachable: from slightly above base to reasonable maximum height
        z_min = base + 0.15
        z_max = base + l1 + l2 - 0.15
        z = np.random.uniform(z_min, min(z_max, 0.7))

        target_xyz = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z])

        # validate with IK - check if solution converges to actual target
        solved, _ = solve_inverse_kinematics(env, target_xyz, max_iters=200, tol=TOL)
        if solved:
            return target_xyz

    print("Warning: Could not find reachable target after 100 attempts. Using safe default.")
    return np.array([-0.4, 0.1, 0.4])


def generate_data(num_episodes=100, episode_max_length=150, filename="data/robot_mpc_dataset.h5"):
    """
    Generate dataset for learning an MPC controller.
    Inputs  : [joint positions + joint velocities]
    Targets : end-effector (EE) target positions
    Actions : torques from MPC only
    """

    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    controller = MPCController(dt=0.05, prediction_horizon=20)
    n_sim_steps_per_mpc_step = int(controller.dt / env.model.opt.timestep)

    all_states, all_targets, all_actions = [], [], []

    print(f"--- Generating {num_episodes} episodes ---")

    for episode in tqdm(range(num_episodes), desc="Episodes"):
        obs = env.reset()
        target_xyz = sample_3d_cartesian_target(env)
        _, target_joint_pos = solve_inverse_kinematics(env, target_xyz)

        # temporary storage for this episode
        ep_states = []
        ep_targets = []
        ep_actions = []

        converged = False

        for step in range(episode_max_length):
            current_state = obs[:6]
            tau_mpc, solved = controller.solve(current_state, target_joint_pos)

            if solved:
                tau_static = env.data.qfrc_bias.copy()
                total_tau = tau_mpc + tau_static

                ep_states.append(current_state)
                ep_targets.append(target_xyz)
                ep_actions.append(tau_mpc)

                for _ in range(n_sim_steps_per_mpc_step):
                    obs, _, _, _ = env.step(total_tau)

            # check if target reached
            ee_pos = env.get_ee_position()
            if np.linalg.norm(ee_pos - target_xyz) < TOL:
                print(f"Early stop in episode {episode} at step {step}")
                converged = True
                break

        # keep this episode only if converged
        if converged:
            all_states.extend(ep_states)
            all_targets.extend(ep_targets)
            all_actions.extend(ep_actions)
        else:
            print(f"Episode {episode} discarded (did not converge).")

    with h5py.File(filename, "w") as f:
        f.create_dataset("states", data=np.array(all_states), compression="gzip")
        f.create_dataset("targets", data=np.array(all_targets), compression="gzip")
        f.create_dataset("actions", data=np.array(all_actions), compression="gzip")

    print(f"--- Data Generation Complete ---\nSaved to: {filename}")
    print(f"States:  {np.array(all_states).shape}")
    print(f"Targets: {np.array(all_targets).shape}")
    print(f"Actions: {np.array(all_actions).shape}")


if __name__ == "__main__":
    generate_data(num_episodes=20, episode_max_length=1000)
