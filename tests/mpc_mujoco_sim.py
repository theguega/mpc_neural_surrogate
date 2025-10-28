import time

import mujoco
import mujoco.viewer
import numpy as np
from mpc_surrogate.mpc_controller import MPCController
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics


def run_visual_test_static_target():
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    controller = MPCController(dt=0.05, prediction_horizon=20)
    n_joints = 3
    n_sim_steps_per_mpc_step = int(controller.dt / env.model.opt.timestep)

    obs = env.reset()
    env.data.qpos[:3] = np.zeros(3)
    env.data.qvel[:3] = np.zeros(3)
    mujoco.mj_forward(env.model, env.data)

    # Static EE target
    target_pos = np.array([-0.5, 0.2, 0.5])
    print("EE Target position:", target_pos)

    # Compute IK target (joint angles)
    joint_target = solve_inverse_kinematics(env, target_pos)
    print("Computed joint target (IK):", joint_target)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(500):
            # True current EE position in simulation
            current_ee_pos = env.get_ee_position()
            ee_distance = np.linalg.norm(current_ee_pos - target_pos)

            if step % 50 == 0:
                print(f"Step {step}: EE pos = {current_ee_pos}, Target = {target_pos}, Error = {ee_distance:.3f}")

            # --- MPC control ---
            current_state = obs[:6]  # q, qdot
            tau_mpc, solved = controller.solve(current_state, joint_target)
            if not solved:
                tau_mpc = np.zeros(n_joints)

            tau_static = env.data.qfrc_bias
            total_tau = tau_mpc + tau_static

            # Apply torque for one MPC timestep
            for _ in range(n_sim_steps_per_mpc_step):
                obs, _, _, _ = env.step(total_tau)

            env.render(viewer)
            time.sleep(0.001)


if __name__ == "__main__":
    run_visual_test_static_target()
