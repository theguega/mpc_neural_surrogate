import time

import h5py
import mujoco
import mujoco.viewer
import numpy as np


def load_h5_dataset(filepath):
    with h5py.File(filepath, "r") as hf:
        X = hf["inputs/states_and_targets"][:]
        y = hf["outputs/torques"][:]
    print(f"Loaded dataset with {X.shape[0]} samples.")
    return X, y


def get_ee_position(model, data):
    site_id = model.site("ee_site").id
    return data.site_xpos[site_id].copy()


def replay_random_sample(xml_path, h5_path):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    n_joints = 3

    X, y = load_h5_dataset(h5_path)

    # pick random dataset sample
    idx = np.random.randint(0, X.shape[0])
    sample = X[idx]
    torque = y[idx]
    target_pos = sample[2 * n_joints :]

    # initialize sim state
    data.qpos[:n_joints] = sample[:n_joints]
    data.qvel[:n_joints] = sample[n_joints : 2 * n_joints]
    mujoco.mj_forward(model, data)

    # Compute current EE position
    ee_pos = get_ee_position(model, data)

    print(f"\n--- Sample #{idx} ---")
    print(f"EE start position : {ee_pos}")
    print(f"Target position   : {target_pos}\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer ready. Waiting 2s before playback...")

        # cam setup
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90

        # wait before motion
        start = time.time()
        while viewer.is_running() and time.time() - start < 2.0:
            viewer.sync()
            time.sleep(0.01)

        # replay motion
        print("Playing...")
        start = time.time()
        while viewer.is_running() and time.time() - start < 10:
            print(f"EE position : {get_ee_position(model, data)}")
            step_start = time.time()
            data.ctrl[:n_joints] = torque
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))

    print("Replay finished.")


if __name__ == "__main__":
    XML_FILE = "models/3dof_arm.xml"
    H5_FILE = "data/mpc_3dof_dataset.h5"
    replay_random_sample(XML_FILE, H5_FILE)
