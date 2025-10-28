"""
Generate dataset using Jacobian-based operational space control.
This is a proven controller that works reliably with robot arms.
"""
import time
import h5py
import mujoco
import numpy as np


def get_ee_position(model, data):
    """Helper to get the end-effector position."""
    site_id = model.site("ee_site").id
    return data.site_xpos[site_id].copy()


def compute_jacobian(model, data, site_name="ee_site"):
    """
    Compute the Jacobian matrix for the end-effector.
    
    Returns:
        J: 3x3 positional Jacobian matrix
    """
    site_id = model.site(site_name).id
    n_joints = 3
    
    # Allocate Jacobian matrices
    jacp = np.zeros((3, model.nv))  # Position Jacobian
    jacr = np.zeros((3, model.nv))  # Rotation Jacobian
    
    # Compute Jacobian
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    
    # Return only the position Jacobian for the first 3 joints
    return jacp[:, :n_joints]


def operational_space_control(model, data, target_pos, kp=200.0, kd=40.0):
    """
    Operational space controller using Jacobian transpose.
    This is a well-established controller for robot manipulators.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data  
        target_pos: Target end-effector position [x, y, z]
        kp: Position gain (increased for stronger response)
        kd: Damping gain (increased for stability)
    
    Returns:
        Torque commands for the joints
    """
    n_joints = 3
    
    # Get current end-effector position
    ee_pos = get_ee_position(model, data)
    
    # Compute position error
    pos_error = target_pos - ee_pos
    
    # Compute Jacobian
    J = compute_jacobian(model, data)
    
    # Check for singularity (determinant close to zero)
    # If near singular, use damped least squares instead of transpose
    damping_factor = 0.01
    JtJ = J.T @ J + damping_factor * np.eye(n_joints)
    
    # Compute end-effector velocity (J * q_dot)
    ee_vel = J @ data.qvel[:n_joints]
    
    # Desired force in task space (PD control)
    f_task = kp * pos_error - kd * ee_vel
    
    # Map to joint torques using damped pseudo-inverse
    # τ = (J^T J + λI)^{-1} J^T f
    # This is more robust near singularities than pure transpose
    torque = np.linalg.solve(JtJ, J.T @ f_task)
    
    # Add gravity compensation
    mujoco.mj_forward(model, data)
    gravity_comp = -data.qfrc_bias[:n_joints]
    
    # Total torque
    total_torque = torque + gravity_comp
    
    # Clip to actuator limits
    total_torque = np.clip(total_torque, -5.0, 5.0)
    
    return total_torque


def generate_dataset(
    xml_path,
    num_samples=500,
    save_path="mpc_3dof_dataset.h5",
):
    """
    Generates a dataset of (state, target) -> torque tuples for the 3-DoF arm
    using operational space control (Jacobian-based).
    """
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    n_joints = 3
    
    # states: (joint_pos, joint_vel)
    inputs = np.zeros((num_samples, 2 * n_joints + 3))  # (q, dq, target_pos)
    outputs = np.zeros((num_samples, n_joints))  # (torques)
    
    print(f"Generating {num_samples} samples using Jacobian-based control...")
    start_time = time.time()
    
    joint_ranges = np.array([[-3.14, 3.14], [-1.57, 1.57], [-2.0, 2.0]])
    
    # Arm parameters for reachability check
    max_reach = 0.55
    min_reach = 0.25  # Minimum distance from base (avoid targets too close to arm)
    
    i = 0
    attempts = 0
    max_attempts = num_samples * 10
    
    while i < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Generate random target in spherical coordinates
        radius = np.random.uniform(min_reach, max_reach)
        theta = np.random.uniform(-np.pi, np.pi)
        phi = np.random.uniform(-0.5, 1.0)
        
        base_height = 0.4
        target_x = radius * np.cos(theta) * np.cos(phi)
        target_y = radius * np.sin(theta) * np.cos(phi)
        target_z = base_height + radius * np.sin(phi)
        
        target_pos = np.array([target_x, target_y, target_z])
        
        # Reachability check
        horizontal_dist = np.sqrt(target_x**2 + target_y**2)
        vertical_dist = abs(target_z - base_height)
        total_dist = np.sqrt(horizontal_dist**2 + vertical_dist**2)
        
        if total_dist > max_reach or total_dist < min_reach or target_z < 0.15:
            continue
        
        # Random initial state
        for j in range(n_joints):
            data.qpos[j] = np.random.uniform(*joint_ranges[j])
        data.qvel[:n_joints] = np.random.uniform(-0.5, 0.5, n_joints)
        mujoco.mj_forward(model, data)
        
        # Compute control using operational space control
        torque = operational_space_control(model, data, target_pos, kp=200.0, kd=40.0)
        
        # Input: [joint_pos, joint_vel, target_pos]
        inputs[i, :n_joints] = data.qpos[:n_joints]
        inputs[i, n_joints : 2 * n_joints] = data.qvel[:n_joints]
        inputs[i, 2 * n_joints :] = target_pos
        
        # Output: [torques]
        outputs[i] = torque
        
        i += 1
        
        if i % 50 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (num_samples - i)
            print(
                f"  Generated {i}/{num_samples} samples. "
                f"Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s"
            )
    
    if attempts >= max_attempts:
        print(f"Warning: Reached max attempts. Generated {i}/{num_samples} samples.")
        inputs = inputs[:i]
        outputs = outputs[:i]
        num_samples = i
    
    with h5py.File(save_path, "w") as hf:
        input_group = hf.create_group("inputs")
        output_group = hf.create_group("outputs")
        
        input_group.create_dataset("states_and_targets", data=inputs)
        output_group.create_dataset("torques", data=outputs)
        
        hf.attrs["num_samples"] = num_samples
        hf.attrs["description"] = (
            "Dataset for 3-DoF arm using Jacobian-based operational space control. "
            "Inputs: [q, dq, target], Outputs: [torques]"
        )
    
    print(f"\nDataset successfully saved to {save_path}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    return save_path


if __name__ == "__main__":
    XML_FILE = "models/3dof_arm.xml"
    OUTPUT_FILE = "data/mpc_3dof_dataset_jacobian.h5"
    NUM_SAMPLES = 1000
    
    generate_dataset(
        xml_path=XML_FILE,
        num_samples=NUM_SAMPLES,
        save_path=OUTPUT_FILE,
    )
