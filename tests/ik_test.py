import mujoco
import numpy as np

from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics

TOL = 0.02  # 2cm tolerance


def test_inverse_kinematics():
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    target_xyz = np.array([-0.4, 0.1, 0.40])  # target to reach

    _, q_sol = solve_inverse_kinematics(env, target_xyz)
    print("IK solution joint angles:", q_sol)

    env.data.qpos[:] = q_sol
    mujoco.mj_forward(env.model, env.data)

    # reed EE position
    ee_pos = env.get_ee_position()
    error = np.linalg.norm(ee_pos - target_xyz)

    print("EE position from IK solution:", ee_pos)
    print("EE position error:", error)
    assert error < TOL, "IK solution is not accurate enough."


if __name__ == "__main__":
    test_inverse_kinematics()
