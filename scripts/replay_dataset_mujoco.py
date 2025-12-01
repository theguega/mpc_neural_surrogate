import random
import time

import h5py
import mujoco
import mujoco.viewer
import numpy as np
from mpc_surrogate.mujoco_env import MuJoCoEnvironment

DEBUG = False


def replay_dataset(filename="data/robot_mpc_dataset.h5", slow_factor=1.0):
    """
    Replay the dataset stored in HDF5 to visualize robot motion and targets.
    Each frame applies the recorded action (tau_mpc) on the robot.

    Args:
        filename: path to the dataset (HDF5 file)
        slow_factor: >1 slows the playback (e.g., 2.0 = half speed)
    """

    with h5py.File(filename, "r") as f:
        episodes = f["episodes"]
        ep_keys = sorted(list(episodes.keys()))

        num_to_play = min(10, len(ep_keys))
        selected_eps = random.sample(ep_keys, num_to_play)

        env = MuJoCoEnvironment("models/3dof_robot_arm.xml")

        n_sim_steps_per_mpc_step = int(0.05 / env.model.opt.timestep)
        _ = env.reset()
        mujoco.mj_forward(env.model, env.data)

        # Mocap marker for EE target
        target_body_id = env.model.body("target").id
        env.model.body_mocapid[target_body_id] = 0  # enable mocap control

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            for episode_name in selected_eps:
                print(f"Episode {episode_name}")

                ep = episodes[episode_name]
                states = ep["states"][:]
                targets = ep["targets"][:]
                actions = ep["actions"][:]

                for i in range(len(states)):
                    qpos = states[i, :3]
                    qvel = states[i, 3:]
                    target_pos = targets[i]
                    total_tau = actions[i]

                    # Update robot state and target marker
                    env.data.qpos[:3] = qpos
                    env.data.qvel[:3] = qvel
                    env.data.mocap_pos[0] = target_pos
                    mujoco.mj_forward(env.model, env.data)

                    # Apply torque and step physics
                    # total_tau = tau_mpc + env.data.qfrc_bias
                    for _ in range(n_sim_steps_per_mpc_step):
                        obs, _, _, _ = env.step(total_tau)

                    # Render the frame
                    env.render(viewer)
                    time.sleep(env.model.opt.timestep * n_sim_steps_per_mpc_step * slow_factor)

                    if DEBUG:
                        ee_pos = env.get_ee_position()
                        dist = np.linalg.norm(ee_pos - target_pos)

                        print(f"Step {i}: EE pos = {ee_pos}, Target = {target_pos}, Error = {dist:.3f}")


if __name__ == "__main__":
    replay_dataset("data/robot_mpc_dataset.h5", slow_factor=1.0)
