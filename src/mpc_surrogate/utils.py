import mujoco
import numpy as np


def solve_inverse_kinematics(env, target_xyz, max_iters=200, tol=1e-4):
    """
    Pure IK solver using Jacobian pseudo-inverse
    Translates an end-effector position to joint angles
    """
    model, data = env.model, env.data
    n_joints = model.nq
    q = data.qpos.copy()  # work on a copy
    site_id = model.site("ee_site").id

    # allocate Jacobians
    jacp = np.zeros((3, n_joints))
    jacr = np.zeros((3, n_joints))

    # create a separate MjData object for computation to avoid modifying the original data
    tmp_data = mujoco.MjData(model)

    for _ in range(max_iters):
        tmp_data.qpos[:] = q
        mujoco.mj_forward(model, tmp_data)
        ee_pos = tmp_data.site_xpos[site_id]
        error = target_xyz - ee_pos
        if np.linalg.norm(error) < tol:
            return q

        mujoco.mj_jacSite(model, tmp_data, jacp, jacr, site_id)
        dq = np.linalg.pinv(jacp) @ error
        q += dq

    return q
