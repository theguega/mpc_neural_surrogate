import mujoco
import numpy as np
from mpc_surrogate.mujoco_env import MuJoCoEnvironment
from mpc_surrogate.utils import solve_inverse_kinematics


def test_inverse_kinematics():
    env = MuJoCoEnvironment("models/3dof_robot_arm.xml")
    target_xyz = np.array([-0.4, 0.1, 0.40])  # target to reach

    q_sol = solve_inverse_kinematics(env, target_xyz)
    print("IK solution joint angles:", q_sol)

    # Assign joint angles and run forward simulation
    env.data.qpos[:] = q_sol
    mujoco.mj_forward(env.model, env.data)

    # Read EE position
    ee_pos = env.get_ee_position()
    error = np.linalg.norm(ee_pos - target_xyz)

    print("EE position from IK solution:", ee_pos)
    print("EE position error:", error)
    assert error < 0.02, "IK solution is not accurate enough."


if __name__ == "__main__":
    test_inverse_kinematics()
