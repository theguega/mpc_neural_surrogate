import os

import mujoco
import numpy as np
from mpc_surrogate.generate_data import get_ee_position, mpc_control

XML_PATH = "models/3dof_arm.xml"


def test_mpc_reaches_target():
    """Check that MPC drives the arm's end-effector close to the target and log trajectory."""
    assert os.path.exists(XML_PATH), f"Model XML not found at {XML_PATH}"
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    target_pos = np.array([0.4, 0.0, 0.3])
    mujoco.mj_resetData(model, data)
    initial_pos = get_ee_position(model, data)

    print(f"\nInitial EE pos: {initial_pos}")
    print(f"Target pos:     {target_pos}\n")

    positions = [initial_pos.copy()]

    # Run MPC control loop
    for i in range(50):
        torque = mpc_control(model, data, target_pos, horizon=10, dt=dt)
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)

        ee_pos = get_ee_position(model, data)
        positions.append(ee_pos.copy())

        if i % 5 == 0:
            dist = np.linalg.norm(ee_pos - target_pos)
            print(f"[iter {i:02d}] EE pos: {ee_pos}, dist={dist:.3f}")

    final_pos = positions[-1]
    final_dist = np.linalg.norm(final_pos - target_pos)

    print(f"\nInitial distance: {np.linalg.norm(initial_pos - target_pos):.3f}")
    print(f"Final distance:   {final_dist:.3f}")

    assert final_dist < 0.05, f"MPC failed to reach target (final dist={final_dist:.3f})"
