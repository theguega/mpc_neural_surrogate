import time

import h5py
import mujoco
import numpy as np
from scipy.optimize import minimize


def get_ee_position(model, data):
    """Helper to get the end-effector position."""
    site_id = model.site("ee_site").id
    return data.site_xpos[site_id].copy()


def mpc_control(model, data, target_pos, horizon=30):
    """
    MPC controller using MuJoCo's mj_step for accurate dynamics prediction.
    Optimizes a sequence of torques over a short horizon.
    """
    n_joints = 3

    # store the initial state to reset to it for each optimization iteration
    q0 = data.qpos[:n_joints].copy()
    dq0 = data.qvel[:n_joints].copy()

    # Compute gravity compensation torques for initial guess
    temp_data = mujoco.MjData(model)
    temp_data.qpos[:n_joints] = q0
    temp_data.qvel[:n_joints] = 0  # zero velocity for gravity compensation
    mujoco.mj_forward(model, temp_data)
    gravity_comp = -temp_data.qfrc_bias[:n_joints].copy()

    def cost_function(u_seq_flat):
        temp_data = mujoco.MjData(model)
        temp_data.qpos[:n_joints] = q0
        temp_data.qvel[:n_joints] = dq0
        mujoco.mj_forward(model, temp_data)

        total_cost = 0.0
        u_seq = u_seq_flat.reshape((horizon, n_joints))

        for t in range(horizon):
            # apply control and step the simulation
            temp_data.ctrl[:n_joints] = u_seq[t]
            mujoco.mj_step(model, temp_data)

            # Get end-effector position
            ee_pos = get_ee_position(model, temp_data)

            # calculate cost with exponentially increasing weight on tracking
            # Weight increases toward the end of the horizon
            tracking_weight = 50.0 * (1 + t / horizon)
            tracking_error = tracking_weight * np.sum((ee_pos - target_pos) ** 2)
            control_effort = 0.01 * np.sum(u_seq[t] ** 2)
            total_cost += tracking_error + control_effort

        return total_cost

    # Initialize with gravity compensation
    u_init = np.tile(gravity_comp, horizon)

    # Increased bounds for faster motion
    bounds = [(-10.0, 10.0)] * (horizon * n_joints)

    # solve the optimization problem
    result = minimize(
        cost_function, u_init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 50}
    )

    if result.success:
        u_seq_opt = result.x.reshape((horizon, n_joints))
        # return the first torque command (receding horizon principle)
        return u_seq_opt[0]
    else:
        # fallback to gravity compensation if optimization fails
        print("MPC optimization failed, using gravity compensation.")
        return gravity_comp


def generate_dataset(
    xml_path,
    num_samples=2000,
    save_path="mpc_3dof_dataset.h5",
):
    """
    Generates a dataset of (state, target) -> torque tuples for the 3-DoF arm.
    The data is saved in an HDF5 file.
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    n_joints = 3

    # states: (joint_pos, joint_vel)
    inputs = np.zeros((num_samples, 2 * n_joints + 3))  # (q, dq, target_pos)
    outputs = np.zeros((num_samples, n_joints))  # (torques)

    print(f"Generating {num_samples} samples...")
    start_time = time.time()

    joint_ranges = np.array([[-3.14, 3.14], [-1.57, 1.57], [-2.0, 2.0]])

    for i in range(num_samples):
        # random target in the reachable space
        # arm total length ~ 0.85m, positioned at height 0.05m
        target_pos = np.random.uniform([0.15, -0.5, 0.2], [0.6, 0.5, 0.8])

        # random initial state
        for j in range(n_joints):
            data.qpos[j] = np.random.uniform(*joint_ranges[j])
        data.qvel[:n_joints] = np.random.uniform(-0.5, 0.5, n_joints)
        mujoco.mj_forward(model, data)

        # MPC control - uses default horizon of 30
        torque = mpc_control(model, data, target_pos)

        # input: [joint_pos, joint_vel, target_pos]
        inputs[i, :n_joints] = data.qpos[:n_joints]
        inputs[i, n_joints : 2 * n_joints] = data.qvel[:n_joints]
        inputs[i, 2 * n_joints :] = target_pos

        # output: [torques]
        outputs[i] = torque

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  Generated {i + 1}/{num_samples} samples. Time elapsed: {elapsed:.2f}s")

    with h5py.File(save_path, "w") as hf:
        input_group = hf.create_group("inputs")
        output_group = hf.create_group("outputs")

        input_group.create_dataset("states_and_targets", data=inputs)
        output_group.create_dataset("torques", data=outputs)

        hf.attrs["num_samples"] = num_samples
        hf.attrs[
            "description"
        ] = "Dataset for 3-DoF arm MPC imitation. Inputs: [q, dq, target], Outputs: [torques]"

    print(f"Dataset successfully saved to {save_path}")
    return save_path


if __name__ == "__main__":
    XML_FILE = "models/3dof_arm.xml"
    OUTPUT_FILE = "data/mpc_3dof_dataset.h5"
    NUM_SAMPLES = 500

    generate_dataset(
        xml_path=XML_FILE,
        num_samples=NUM_SAMPLES,
        save_path=OUTPUT_FILE,
    )