import time

import h5py
import mujoco
import mujoco.viewer
import numpy as np
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.data_generator import episode_length


def replay_dataset(filename="data/robot_mpc_dataset.h5", slow_factor=1.0):
    """
    Replay the dataset stored in HDF5 to visualize robot motion and targets.
    Each frame applies the recorded action (tau_mpc) on the robot.

    Args:
        filename: path to the dataset (HDF5 file)
        slow_factor: >1 slows the playback (e.g., 2.0 = half speed)
    """

    # --- Load dataset ---
    with h5py.File(filename, "r") as f:
        states = np.array(f["states"])  # [N, 6]
        targets = np.array(f["targets"])  # [N, 3]
        actions = np.array(f["actions"])  # [N, 3]

    print(f"Loaded dataset: {len(states)} steps")
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")

    n_sim_steps_per_mpc_step = int(0.05 / env.model.opt.timestep)
    obs = env.reset()
    env.data.qpos[:3] = states[0, :3]
    env.data.qvel[:3] = states[0, 3:]
    mujoco.mj_forward(env.model, env.data)

    # Mocap marker for EE target
    target_body_id = env.model.body("target").id
    env.model.body_mocapid[target_body_id] = 0  # enable mocap control

    print("\n--- Starting replay ---")
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
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

            if i % (episode_length - 1) == 0:
                ee_pos = env.get_ee_position()
                dist = np.linalg.norm(ee_pos - target_pos)
                print(f"Step {i}: EE pos = {ee_pos}, Target = {target_pos}, Error = {dist:.3f}")

    print("--- Replay finished ---")


if __name__ == "__main__":
    replay_dataset("data/robot_mpc_dataset.h5", slow_factor=1.0)
