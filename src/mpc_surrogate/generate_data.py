import time

import h5py
import mujoco
import numpy as np
from scipy.optimize import minimize


def get_ee_position(model, data):
    """Helper to get the end-effector position."""
    site_id = model.site("ee_site").id
    return data.site_xpos[site_id].copy()


def mpc_control(model, data, target_pos, horizon=10, dt=0.01, n_joints=3):
    """
    Simple MPC: optimize torque sequence over horizon to minimize
    end-effector tracking error and control effort.
    """
    import mujoco
    import numpy as np

    # Get current state
    q0 = data.qpos[:n_joints].copy()
    qd0 = data.qvel[:n_joints].copy()

    # Create a separate data instance for rollouts
    data_sim = mujoco.MjData(model)

    # Torque limits
    torque_limit = 5.0

    # Weight matrices
    Q_pos = 10000.0  # End-effector position tracking weight
    Q_final = 50000.0  # Extra weight on final position
    R = 0.01  # Very small control effort penalty

    def rollout_dynamics(torques, q_init, qd_init):
        """
        Simulate forward dynamics over horizon using mj_step.
        """
        torques = torques.reshape(horizon, n_joints)

        # Initialize simulation state
        data_sim.qpos[:n_joints] = q_init
        data_sim.qvel[:n_joints] = qd_init

        # Zero out other DOFs if any
        if len(data_sim.qpos) > n_joints:
            data_sim.qpos[n_joints:] = 0
            data_sim.qvel[n_joints:] = 0

        ee_positions = []

        for t in range(horizon):
            # Apply control
            data_sim.ctrl[:n_joints] = torques[t]

            # Step the simulation
            mujoco.mj_step(model, data_sim)

            # Get end-effector position
            ee_pos = get_ee_position(model, data_sim)
            ee_positions.append(ee_pos)

        return np.array(ee_positions)

    def cost_function(torques):
        """
        Compute total cost: tracking error + control effort
        """
        try:
            ee_trajectory = rollout_dynamics(torques, q0, qd0)

            # Tracking cost: all timesteps
            tracking_cost = 0.0
            for ee_pos in ee_trajectory[:-1]:
                error = ee_pos - target_pos
                tracking_cost += Q_pos * np.sum(error**2)

            # Final position cost (heavily weighted)
            final_error = ee_trajectory[-1] - target_pos
            final_cost = Q_final * np.sum(final_error**2)

            # Control effort cost
            torques_reshaped = torques.reshape(horizon, n_joints)
            control_cost = R * np.sum(torques_reshaped**2)

            return tracking_cost + final_cost + control_cost
        except Exception as e:
            print(f"Simulation failed with error: {e}")
            return 1e10  # Return large cost if simulation fails

    # Compute initial guess using inverse kinematics heuristic
    current_ee = get_ee_position(model, data)
    error = target_pos - current_ee
    error_norm = np.linalg.norm(error)

    # Multiple random restarts to escape local minima
    best_result = None
    best_cost = np.inf

    n_restarts = 3
    for restart in range(n_restarts):
        if restart == 0:
            # First try: aggressive movement toward target
            u0 = np.zeros(horizon * n_joints)
            if error_norm > 0.01:
                for t in range(horizon):
                    # Joint 1: rotate toward target x-y position
                    if abs(error[0]) > 0.01 or abs(error[1]) > 0.01:
                        u0[t * n_joints] = np.sign(error[0]) * 3.0

                    # Joints 2 & 3: adjust height
                    if abs(error[2]) > 0.01:
                        u0[t * n_joints + 1] = -np.sign(error[2]) * 3.0
                        u0[t * n_joints + 2] = -np.sign(error[2]) * 3.0
        else:
            # Random restarts with decreasing magnitude
            u0 = np.random.randn(horizon * n_joints) * (3.0 / restart)

        # Clip to bounds
        u0 = np.clip(u0, -torque_limit, torque_limit)

        # Optimize
        result = minimize(
            cost_function,
            u0,
            method="SLSQP",
            bounds=[(-torque_limit, torque_limit)] * (horizon * n_joints),
            options={"maxiter": 150, "ftol": 1e-5},
        )

        if result.fun < best_cost:
            best_cost = result.fun
            best_result = result

    # Return the first control action from best result
    optimal_torques = best_result.x.reshape(horizon, n_joints)

    # Debug: print first torque to verify it's non-zero
    # print(f"First torque: {optimal_torques[0]}, cost: {best_cost:.2f}")

    return optimal_torques[0]


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
    dt = model.opt.timestep

    n_joints = 3

    # states: (joint_pos, joint_vel)
    inputs = np.zeros((num_samples, 2 * n_joints + 3))  # (q, dq, target_pos)
    outputs = np.zeros((num_samples, n_joints))  # (torques)

    print(f"Generating {num_samples} samples...")
    start_time = time.time()

    joint_ranges = np.array([[-3.14, 3.14], [-1.57, 1.57], [-2.0, 2.0]])

    for i in range(num_samples):
        # random target in the reachable space
        target_pos = np.random.uniform([0.2, -0.4, 0.1], [0.7, 0.4, 0.6])

        # random initial state
        for j in range(n_joints):
            data.qpos[j] = np.random.uniform(*joint_ranges[j])
        data.qvel[:n_joints] = np.random.uniform(-1, 1, n_joints)
        mujoco.mj_forward(model, data)

        # MPC control
        torque = mpc_control(model, data, target_pos, horizon=10, dt=dt)

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
