import h5py
import mujoco
import mujoco.viewer
import numpy as np
import time


def load_h5_dataset(filepath):
    with h5py.File(filepath, "r") as hf:
        X = hf["inputs/states_and_targets"][:]
        Y = hf["outputs/torques"][:]
        num_samples = hf.attrs["num_samples"]
    print(f"Loaded dataset with {X.shape[0]} samples.")
    return X, Y, num_samples


def get_ee_position(model, data):
    site_id = model.site("ee_site").id
    return data.site_xpos[site_id].copy()


def replay_random_sample(xml_path, h5_path, mode="precomputed"):
    """
    Replay a random sample from the dataset.
    
    Args:
        xml_path: Path to MuJoCo XML model
        h5_path: Path to HDF5 dataset
        mode: "precomputed" (use stored torques) or "mpc" (recompute MPC in closed-loop)
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    n_joints = 3
    
    X, Y, num_samples = load_h5_dataset(h5_path)
    
    # Pick random dataset sample
    idx = np.random.randint(0, num_samples)
    sample = X[idx]
    precomputed_torques = Y[idx]
    target_pos = sample[2 * n_joints :]
    
    # Initialize sim state
    data.qpos[:n_joints] = sample[:n_joints]
    data.qvel[:n_joints] = sample[n_joints : 2 * n_joints]
    mujoco.mj_forward(model, data)
    
    # Position the target marker
    target_body_id = model.body("target").id
    model.body_pos[target_body_id] = target_pos
    
    # Compute current EE position
    ee_pos = get_ee_position(model, data)
    
    print(f"\n--- Sample #{idx} ---")
    print(f"Mode: {mode}")
    print(f"Initial joint positions: {sample[:n_joints]}")
    print(f"Initial joint velocities: {sample[n_joints:2*n_joints]}")
    print(f"EE start position : {ee_pos}")
    print(f"Target position   : {target_pos}")
    if mode == "precomputed":
        print(f"Pre-computed torques: {precomputed_torques}\n")
    
    if mode == "mpc":
        # Import Jacobian controller only if needed
        from mpc_surrogate.generate_data_jacobian import operational_space_control
        mpc_update_interval = 1  # Recompute control every step for smooth motion
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer ready. Press Ctrl+C to exit.\n")
        
        # Cam setup
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 90
        
        # Control variables
        torques = precomputed_torques if mode == "precomputed" else np.zeros(n_joints)
        
        # Run simulation (increase steps for longer simulation time)
        num_steps = 5000  # 5 seconds at 0.001s timestep
        start_time = time.time()
        
        # Always use real-time playback so we can see the motion
        use_realtime = True
        
        for step in range(num_steps):
            step_start = time.time()
            
            # Recompute control periodically if in MPC mode
            if mode == "mpc" and step % mpc_update_interval == 0:
                torques = operational_space_control(model, data, target_pos, kp=200.0, kd=40.0)
            
            # Apply torques and step simulation
            data.ctrl[:n_joints] = torques
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Sleep to maintain real-time playback
            if use_realtime:
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
            
            # Print progress every 500 steps (every 0.5 seconds)
            if step % 500 == 0:
                ee_pos = get_ee_position(model, data)
                distance = np.linalg.norm(ee_pos - target_pos)
                print(f"Step {step}: EE at {ee_pos}, distance to target: {distance:.4f}m")
        
        elapsed = time.time() - start_time
        print(f"\nReplay finished in {elapsed:.2f}s.")


if __name__ == "__main__":
    XML_FILE = "models/3dof_arm.xml"
    H5_FILE = "data/mpc_3dof_dataset_jacobian.h5"  # Use new Jacobian dataset
    
    # Choose mode:
    # "precomputed" - Fast, uses stored torques from dataset (won't reach target, just shows initial action)
    # "mpc" - Recomputes control in closed-loop (proper tracking behavior)
    
    MODE = "mpc"  # Use MPC mode for proper target reaching behavior
    
    replay_random_sample(XML_FILE, H5_FILE, mode=MODE)