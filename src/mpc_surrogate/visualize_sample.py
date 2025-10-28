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
        # Choose controller based on dataset type
        import os
        jacobian_dataset = os.path.basename(h5_path).startswith("mpc_3dof_dataset_jacobian")
        if jacobian_dataset:
            from mpc_surrogate.generate_data_jacobian import operational_space_control
            controller = lambda model, data, target: operational_space_control(model, data, target, kp=200.0, kd=40.0)
            print("Closed-loop: Jacobian controller.")
        else:
            from mpc_surrogate.generate_data import setup_do_mpc_controller
            print("Closed-loop: do-mpc controller.")
            # Increase horizon and cost weights for faster, more aggressive movement
            mpc = setup_do_mpc_controller(target_pos, horizon=60)
            # Set initial state ONCE
            x0 = np.concatenate([data.qpos[:n_joints], data.qvel[:n_joints]])
            mpc.x0 = x0
            mpc.set_initial_guess()
            def controller(model, data, target):
                x = np.concatenate([data.qpos[:n_joints], data.qvel[:n_joints]])
                try:
                    u_opt = mpc.make_step(x)
                    return u_opt.flatten()
                except Exception as e:
                    print(f"MPC step failed: {e}")
                    return np.zeros(n_joints)
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
        num_steps = 100000  # 5 seconds at 0.001s timestep
        start_time = time.time()

        # Use real-time playback only for Jacobian controller
        use_realtime = False
        if mode == "mpc":
            import os
            jacobian_dataset = os.path.basename(h5_path).startswith("mpc_3dof_dataset_jacobian")
            if jacobian_dataset:
                use_realtime = True
                num_steps = 3000  # 3 seconds for Jacobian dataset
        
        stop_threshold = 0.01  # 1cm
        # For do-mpc, visualize only every Nth step for speed
        mpc_skip = 20 if (mode == "mpc" and not jacobian_dataset) else 1
        step = 0
        while step < num_steps:
            step_start = time.time()

            ee_pos = get_ee_position(model, data)
            distance = np.linalg.norm(ee_pos - target_pos)

            # Exit if target is reached
            if distance < stop_threshold:
                print(f"Target reached at step {step}, distance: {distance:.4f}m. Exiting.")
                break

            # Recompute control periodically if in MPC mode
            if mode == "mpc" and step % mpc_update_interval == 0:
                torques = controller(model, data, target_pos)

            # For do-mpc, step and sync viewer mpc_skip times per loop
            for _ in range(mpc_skip):
                if step >= num_steps:
                    break
                data.ctrl[:n_joints] = torques
                mujoco.mj_step(model, data)
                step += 1

            # Sync viewer
            viewer.sync()

            # Sleep to maintain real-time playback
            if use_realtime:
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

            # Print progress every 500 steps (every 0.5 seconds)
            if step % 500 < mpc_skip:
                print(f"Step {step}: EE at {ee_pos}, distance to target: {distance:.4f}m")
        
        elapsed = time.time() - start_time
        print(f"\nReplay finished in {elapsed:.2f}s.")


if __name__ == "__main__":
    import sys
    XML_FILE = "models/3dof_arm.xml"
    # Option: choose dataset type
    # Set to "jacobian" or "mpc" (do-mpc)
    DATASET_TYPE = "mpc"  # Change to "mpc" for do-mpc dataset

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ("jacobian", "mpc"):
            DATASET_TYPE = arg

    if DATASET_TYPE == "jacobian":
        H5_FILE = "data/mpc_3dof_dataset_jacobian.h5"
        print("Using Jacobian dataset.")
    else:
        H5_FILE = "data/mpc_3dof_dataset.h5"
        print("Using do-mpc dataset.")

    # Choose mode:
    # "precomputed" - Fast, uses stored torques from dataset (won't reach target, just shows initial action)
    # "mpc" - Recomputes control in closed-loop (proper tracking behavior)

    MODE = "mpc"  # Use MPC mode for proper target reaching behavior

    replay_random_sample(XML_FILE, H5_FILE, mode=MODE)