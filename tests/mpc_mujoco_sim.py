import time

import mujoco
import mujoco.viewer
import numpy as np
from mpc_surrogate.mpc_controller import MPCController
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics


def run_visual_test_three_targets():
    # --- Init environment and controller ---
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    controller = MPCController(dt=0.05, prediction_horizon=20)
    n_joints = 3
    n_sim_steps_per_mpc_step = int(controller.dt / env.model.opt.timestep)

    # --- Reset robot ---
    obs = env.reset()
    env.data.qpos[:3] = np.zeros(3)
    env.data.qvel[:3] = np.zeros(3)
    mujoco.mj_forward(env.model, env.data)

    # --- Hardcoded targets (in world coordinates) ---
    targets = [
        np.array([-0.5, 0.2, 0.5]),
        np.array([-0.4, 0.1, 0.40]),
        np.array([-0.2, -0.3, 0.35]),
        np.array([-0.5, 0.0, 0.55]),
    ]

    # --- Set up visible mocap target ---
    target_body_id = env.model.body("target").id
    env.model.body_mocapid[target_body_id] = 0  # enable mocap binding

    # --- Start simulation ---
    current_target_idx = 0
    target_pos = targets[current_target_idx]
    joint_target = solve_inverse_kinematics(env, target_pos)
    env.data.mocap_pos[0] = target_pos  # show first target
    mujoco.mj_forward(env.model, env.data)

    print(f"Starting simulation with {len(targets)} targets.")
    print(f"Target 1: {target_pos}")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(4000):
            current_ee_pos = env.get_ee_position()
            ee_error = np.linalg.norm(current_ee_pos - target_pos)

            # --- Switch to next target when close enough ---
            if ee_error < 0.03:
                current_target_idx += 1
                if current_target_idx >= len(targets):
                    print("✅ All targets reached. Ending simulation.")
                    break

                target_pos = targets[current_target_idx]
                joint_target = solve_inverse_kinematics(env, target_pos)
                env.data.mocap_pos[0] = target_pos
                mujoco.mj_forward(env.model, env.data)
                print(f"\n➡️  New target {current_target_idx + 1}: {target_pos}")

            # --- MPC Control ---
            current_state = obs[:6]
            tau_mpc, solved = controller.solve(current_state, joint_target)
            if not solved:
                tau_mpc = np.zeros(n_joints)

            total_tau = tau_mpc + env.data.qfrc_bias

            # Step simulation
            for _ in range(n_sim_steps_per_mpc_step):
                obs, _, _, _ = env.step(total_tau)

            env.render(viewer)
            time.sleep(0.001)

            if step % 100 == 0:
                print(f"Step {step}: EE error = {ee_error:.3f}")

    print("Simulation complete.")


if __name__ == "__main__":
    run_visual_test_three_targets()
