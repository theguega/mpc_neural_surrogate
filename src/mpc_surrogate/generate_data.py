import time

import casadi as ca
import do_mpc
import h5py
import mujoco
import numpy as np


def get_ee_position(model, data):
    """Helper to get the end-effector position."""
    site_id = model.site("ee_site").id
    return data.site_xpos[site_id].copy()


def forward_kinematics(q):
    """
    Compute end-effector position from joint angles using forward kinematics.
    This is a simplified analytical FK for the 3DOF arm.
    
    Args:
        q: Joint angles [q1, q2, q3] (CasADi symbolic or numpy array)
    
    Returns:
        End-effector position [x, y, z]
    """
    # Link lengths and base height (from MuJoCo XML)
    base_height = 0.05  # base_platform pos z
    L1 = 0.3           # link1 height (link1 body pos z)
    L2 = 0.3           # link2 length (link2 body pos z)
    L3 = 0.25          # link3 length (link3 body pos z)

    if isinstance(q, np.ndarray):
        cos = np.cos
        sin = np.sin
    else:
        cos = ca.cos
        sin = ca.sin

    # The base of link1 is at (0, 0, base_height)
    # The top of link1 is at (0, 0, base_height + L1)
    # Joint 1 (q[0]) rotates around Z at base
    # Joint 2 (q[1]) rotates around Y at shoulder (top of link1)
    # Joint 3 (q[2]) rotates around Y at elbow (end of link2)

    # Position after link1 (shoulder position)
    shoulder_z = base_height + L1

    # Compute position of end effector relative to shoulder
    # In the shoulder frame:
    #   - X' = L2 * cos(q2) + L3 * cos(q2 + q3)
    #   - Z' = L2 * sin(q2) + L3 * sin(q2 + q3)
    x_sh = L2 * cos(q[1]) + L3 * cos(q[1] + q[2])
    z_sh = L2 * sin(q[1]) + L3 * sin(q[1] + q[2])

    # Rotate by base (q[0]) around Z
    x = x_sh * cos(q[0])
    y = x_sh * sin(q[0])
    z = shoulder_z + z_sh

    if isinstance(q, np.ndarray):
        return np.array([x, y, z])
    else:
        return ca.vertcat(x, y, z)


def setup_do_mpc_controller(target_pos, horizon=30):
    """
    Set up a do-mpc controller for the 3DOF arm.
    
    Args:
        target_pos: Target end-effector position [x, y, z]
        horizon: MPC prediction horizon
    
    Returns:
        Configured do-mpc MPC controller
    """
    # Model setup
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type)
    
    # States: joint positions and velocities
    q = model.set_variable('_x', 'q', shape=(3, 1))
    dq = model.set_variable('_x', 'dq', shape=(3, 1))
    
    # Control inputs: joint torques
    u = model.set_variable('_u', 'tau', shape=(3, 1))
    
    # Improved dynamics model to better match MuJoCo
    # Mass matrix (better estimates based on MuJoCo model)
    # These account for link masses and their distribution
    m1, m2, m3 = 1.0, 0.8, 0.5  # Link masses from XML
    L1, L2, L3 = 0.3, 0.3, 0.25  # Link lengths
    
    # Simplified mass matrix (diagonal for computational efficiency)
    # These values need to account for rotational inertia
    M11 = 0.01 + m2 * (L2/2)**2 + m3 * (L2**2 + (L3/2)**2)  # Base rotation
    M22 = 0.008 + m2 * (L2/2)**2 + m3 * (L2**2 + (L3/2)**2)  # Shoulder
    M33 = 0.005 + m3 * (L3/2)**2  # Elbow
    
    M = ca.diag(ca.vertcat(M11, M22, M33))
    
    # Damping (from joint damping in XML)
    damping = ca.diag(ca.vertcat(5.0, 5.0, 5.0))  # Match XML damping values
    
    # Improved gravity compensation
    g_accel = 9.81
    # Joint 1 (base rotation) has no gravity torque
    # Joint 2 and 3 need to account for link weights
    g2 = -g_accel * (m2 * (L2/2) * ca.cos(q[1]) + m3 * (L2 * ca.cos(q[1]) + (L3/2) * ca.cos(q[1] + q[2])))
    g3 = -g_accel * m3 * (L3/2) * ca.cos(q[1] + q[2])
    
    g = ca.vertcat(0, g2, g3)
    
    # Acceleration: q̈ = M^-1 * (τ - D*q̇ - g)
    ddq = ca.solve(M, u - damping @ dq - g)
    
    # Set right-hand side of ODE
    model.set_rhs('q', dq)
    model.set_rhs('dq', ddq)
    
    # Setup model
    model.setup()
    
    # MPC setup
    mpc = do_mpc.controller.MPC(model)
    
    setup_mpc = {
        'n_horizon': horizon,
        't_step': 0.01,  # 10ms timestep
        'n_robust': 0,
        'store_full_solution': False,
    }
    mpc.set_param(**setup_mpc)
    
    # Cost function
    ee_pos = forward_kinematics(q)
    target = ca.DM(target_pos)
    
    # Terminal cost: heavily penalize tracking error at end of horizon
    mterm = 500.0 * ca.sumsqr(ee_pos - target)
    
    # Stage cost: tracking error + control effort + velocity regularization
    lterm = 200.0 * ca.sumsqr(ee_pos - target) + 0.001 * ca.sumsqr(u) + 0.1 * ca.sumsqr(dq)
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    
    # Control bounds (match XML actuator limits)
    mpc.bounds['lower', '_u', 'tau'] = -5.0
    mpc.bounds['upper', '_u', 'tau'] = 5.0
    
    # State bounds (joint limits)
    mpc.bounds['lower', '_x', 'q'] = [-3.14, -1.57, -2.0]
    mpc.bounds['upper', '_x', 'q'] = [3.14, 1.57, 2.0]
    
    # Velocity limits
    mpc.bounds['lower', '_x', 'dq'] = -5.0
    mpc.bounds['upper', '_x', 'dq'] = 5.0
    
    # Setup MPC
    mpc.setup()
    
    return mpc


def mpc_control(model, data, target_pos, horizon=30):
    """
    MPC controller using do-mpc library.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        target_pos: Target end-effector position [x, y, z]
        horizon: MPC prediction horizon
    
    Returns:
        Optimal torque command for current timestep
    """
    n_joints = 3
    
    # Get current state
    q0 = data.qpos[:n_joints].copy()
    dq0 = data.qvel[:n_joints].copy()
    
    # Setup do-mpc controller
    mpc = setup_do_mpc_controller(target_pos, horizon)
    
    # Set initial state
    x0 = np.concatenate([q0, dq0])
    mpc.x0 = x0
    mpc.set_initial_guess()
    
    # Solve MPC problem
    try:
        u_opt = mpc.make_step(x0)
        return u_opt.flatten()
    except Exception as e:
        print(f"MPC optimization failed: {e}. Using gravity compensation.")
        # Fallback to gravity compensation
        temp_data = mujoco.MjData(model)
        temp_data.qpos[:n_joints] = q0
        temp_data.qvel[:n_joints] = 0
        mujoco.mj_forward(model, temp_data)
        gravity_comp = -temp_data.qfrc_bias[:n_joints].copy()
        return gravity_comp


def generate_dataset(
    xml_path,
    num_samples=500,
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
    
    print(f"Generating {num_samples} samples using do-mpc...")
    start_time = time.time()
    
    joint_ranges = np.array([[-3.14, 3.14], [-1.57, 1.57], [-2.0, 2.0]])
    
    # Arm parameters for reachability check
    # Total arm reach: L2 + L3 = 0.3 + 0.25 = 0.55m max horizontal
    # Base height + L1 = 0.1 + 0.3 = 0.4m base
    max_reach = 0.55
    min_reach = 0.1  # Minimum distance from base
    
    i = 0
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loop
    
    while i < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Generate random target in spherical coordinates for better coverage
        radius = np.random.uniform(min_reach, max_reach)
        theta = np.random.uniform(-np.pi, np.pi)  # Azimuth
        phi = np.random.uniform(-0.5, 1.0)  # Elevation angle (more upward bias)
        
        # Convert to Cartesian (relative to base at height 0.4m)
        base_height = 0.4
        target_x = radius * np.cos(theta) * np.cos(phi)
        target_y = radius * np.sin(theta) * np.cos(phi)
        target_z = base_height + radius * np.sin(phi)
        
        target_pos = np.array([target_x, target_y, target_z])
        
        # Reachability check: distance from base should be within arm's workspace
        horizontal_dist = np.sqrt(target_x**2 + target_y**2)
        vertical_dist = abs(target_z - base_height)
        total_dist = np.sqrt(horizontal_dist**2 + vertical_dist**2)
        
        # Skip if target is unreachable or too close
        if total_dist > max_reach or total_dist < min_reach or target_z < 0.15:
            continue
        
        # Random initial state
        for j in range(n_joints):
            data.qpos[j] = np.random.uniform(*joint_ranges[j])
        data.qvel[:n_joints] = np.random.uniform(-0.5, 0.5, n_joints)
        mujoco.mj_forward(model, data)
        
        # MPC control using do-mpc
        torque = mpc_control(model, data, target_pos, horizon=30)
        
        # Input: [joint_pos, joint_vel, target_pos]
        inputs[i, :n_joints] = data.qpos[:n_joints]
        inputs[i, n_joints : 2 * n_joints] = data.qvel[:n_joints]
        inputs[i, 2 * n_joints :] = target_pos
        
        # Output: [torques]
        outputs[i] = torque
        
        i += 1  # Only increment on successful sample
        
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (num_samples - i)
            print(
                f"  Generated {i}/{num_samples} samples. "
                f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s"
            )
    
    if attempts >= max_attempts:
        print(f"Warning: Reached max attempts. Generated {i}/{num_samples} samples.")
        # Trim arrays to actual number of samples
        inputs = inputs[:i]
        outputs = outputs[:i]
        num_samples = i
    
    with h5py.File(save_path, "w") as hf:
        input_group = hf.create_group("inputs")
        output_group = hf.create_group("outputs")
        
        input_group.create_dataset("states_and_targets", data=inputs)
        output_group.create_dataset("torques", data=outputs)
        
        hf.attrs["num_samples"] = num_samples
        hf.attrs[
            "description"
        ] = "Dataset for 3-DoF arm MPC imitation using do-mpc. Inputs: [q, dq, target], Outputs: [torques]"
    
    print(f"Dataset successfully saved to {save_path}")
    return save_path


if __name__ == "__main__":
    XML_FILE = "models/3dof_arm.xml"
    OUTPUT_FILE = "data/mpc_3dof_dataset.h5"
    NUM_SAMPLES = 100  # Reduced for testing since do-mpc is slower
    
    generate_dataset(
        xml_path=XML_FILE,
        num_samples=NUM_SAMPLES,
        save_path=OUTPUT_FILE,
    )