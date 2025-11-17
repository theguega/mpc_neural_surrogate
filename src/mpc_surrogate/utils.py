import mujoco
import numpy as np


def solve_inverse_kinematics(env, target_xyz, max_iters=100, tol=1e-4):
    """
    Pure IK solver using Jacobian pseudo-inverse with damped least squares
    Translates an end-effector position to joint angles
    """
    model, data = env.model, env.data
    n_joints = model.nq
    # work on a copy, start from current position
    q = data.qpos[:n_joints].copy()
    site_id = model.site("ee_site").id

    # allocate Jacobians
    jacp = np.zeros((3, n_joints))
    jacr = np.zeros((3, n_joints))

    # create a separate MjData object for computation to avoid modifying the original data
    tmp_data = mujoco.MjData(model)

    damping = 0.01
    step_size = 0.5
    error_norm = float("inf")

    for _ in range(max_iters):
        tmp_data.qpos[:n_joints] = q
        mujoco.mj_forward(model, tmp_data)
        ee_pos = tmp_data.site_xpos[site_id]
        error = target_xyz - ee_pos
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return True, q

        mujoco.mj_jacSite(model, tmp_data, jacp, jacr, site_id)

        # damped least squares (more stable than pure pseudo-inverse)
        J = jacp[:, :n_joints]
        dq = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3)) @ error

        # Limit step size to prevent wild jumps
        dq_norm = np.linalg.norm(dq)
        if dq_norm > step_size:
            dq = dq * (step_size / dq_norm)

        q += dq

        # Wrap angles to [-pi, pi] to avoid accumulation
        q = np.arctan2(np.sin(q), np.cos(q))

    print(f"Warning: IK did not converge for target {target_xyz}. Final error: {error_norm:.4f}")
    return False, q
