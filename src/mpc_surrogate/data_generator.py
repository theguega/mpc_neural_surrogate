import h5py
import numpy as np
from tqdm import tqdm

from mpc_surrogate.mpc_controller import MPCController
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics


def sample_reachable_target(link_lengths=(0.3, 0.3, 0.25)):
    LINK1, LINK2, LINK3 = link_lengths
    MAX_REACH = LINK1 + LINK2 + LINK3 - 0.05
    MIN_REACH = 0.15  # avoid near-base singularities

    r = np.random.uniform(MIN_REACH, MAX_REACH)
    theta = np.random.uniform(-np.pi / 2, np.pi / 2)
    z = np.random.uniform(0.05, 0.55)
    return np.array([r * np.cos(theta), r * np.sin(theta), z])


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

        for step in tqdm(range(episode_length), desc="Steps", leave=False):
            # Occasionally sample a new target
            if np.random.rand() < 0.05:
                target_xyz = sample_reachable_target()
                target_joint_pos = solve_inverse_kinematics(env, target_xyz)
            else:
                site_id = env.model.site("ee_site").id
                target_xyz = env.data.site_xpos[site_id].copy()

            current_state = obs[:6]  # [q1, q2, q3, q̇1, q̇2, q̇3]

            tau_mpc, solved = controller.solve(current_state, target_joint_pos)

            if solved:
                tau_static = env.data.qfrc_bias.copy()
                total_tau = tau_mpc + tau_static

                all_states.append(current_state)
                all_targets.append(target_xyz)
                all_actions.append(total_tau)

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
    generate_data(num_episodes=100, episode_length=150)
