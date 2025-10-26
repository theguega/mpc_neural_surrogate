import h5py
import mujoco
import mujoco.viewer
import numpy as np


def load_h5_dataset(filepath):
    with h5py.File(filepath, "r") as hf:
        X = hf["inputs/states_and_targets"][:]
        num_samples = hf.attrs["num_samples"]
    print(f"Loaded dataset with {X.shape[0]} samples.")
    return X, num_samples


def get_ee_position(model, data):
    site_id = model.site("ee_site").id
    return data.site_xpos[site_id].copy()


def replay_random_sample(xml_path, h5_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    n_joints = 3

    X, num_samples = load_h5_dataset(h5_path)

    # pick random dataset sample
    idx = np.random.randint(0, num_samples)
    sample = X[idx]
    target_pos = sample[2 * n_joints :]

    # initialize sim state
    data.qpos[:n_joints] = sample[:n_joints]
    data.qvel[:n_joints] = sample[n_joints : 2 * n_joints]
    mujoco.mj_forward(model, data)

    # Position the target marker
    target_body_id = model.body("target").id
    model.body_pos[target_body_id] = target_pos

    # Compute current EE position
    ee_pos = get_ee_position(model, data)

    print(f"\n--- Sample #{idx} ---")
    print(f"Initial joint positions: {sample[:n_joints]}")
    print(f"Initial joint velocities: {sample[n_joints:2*n_joints]}")
    print(f"EE start position : {ee_pos}")
    print(f"Target position   : {target_pos}\n")

    # Import MPC controller
    from mpc_surrogate.generate_data import mpc_control

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer ready. Running MPC in closed-loop...\n")

        # cam setup
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        # Run MPC in closed-loop
        for step in range(200):
            # Compute MPC torque based on current state
            # Use longer horizon for better visualization performance
            torques = mpc_control(model, data, target_pos, horizon=50)
            
            # Apply torques and step simulation
            data.ctrl[:n_joints] = torques
            mujoco.mj_step(model, data)
            viewer.sync()

            # Print progress every 50 steps
            if step % 50 == 0:
                ee_pos = get_ee_position(model, data)
                distance = np.linalg.norm(ee_pos - target_pos)
                print(f"Step {step}: EE at {ee_pos}, distance to target: {distance:.4f}m")

    print("\nReplay finished.")


if __name__ == "__main__":
    XML_FILE = "models/3dof_arm.xml"
    H5_FILE = "data/mpc_3dof_dataset.h5"
    replay_random_sample(XML_FILE, H5_FILE)
