### Project Background and Problem Formulation

#### 1. System Description

We consider a 3-degree-of-freedom (3-DOF) robotic manipulator operating in a MuJoCo simulation environment, as defined by the provided XML model. The system's configuration is defined by the generalized coordinates $\mathbf{q} = [q_1, q_2, q_3]^\top$, representing the joint angles, and their time derivatives $\dot{\mathbf{q}}$. The full, observable state vector at discrete time step $k$ is $\mathbf{x}_k = [\mathbf{q}_k^\top, \dot{\mathbf{q}}_k^\top]^\top \in \mathbb{R}^{6}$. The control objective is to drive the end-effector (EE) to track randomly sampled, reachable 3D Cartesian targets $\mathbf{p}_{\text{des}} \in \mathbb{R}^3$ within the robot's workspace.

---

#### 2. Existing Controller: MPC with Inverse Kinematics

Our baseline controller is a hierarchical structure combining an **Inverse Kinematics (IK)** module and a **Model Predictive Control (MPC)** module.

**a) Inverse Kinematics Formulation:**

The IK module is responsible for translating a desired end-effector position $\mathbf{p}_{\text{des}}$ into a feasible joint-space configuration $\mathbf{q}_{\text{ref}}$. This is formulated as an optimization problem minimizing the Cartesian error, solved iteratively using the Jacobian transpose method with Damped Least Squares (DLS) for numerical stability.

Let $\mathbf{p}(\mathbf{q})$ be the forward kinematics function mapping joint angles to the end-effector position. The error is defined as:
$$
\mathbf{e} = \mathbf{p}_{\text{des}} - \mathbf{p}(\mathbf{q})
$$

The update rule for the joint angles at iteration $i$ is derived from the Jacobian $\mathbf{J}(\mathbf{q}) = \frac{\partial \mathbf{p}}{\partial \mathbf{q}}$ and is given by:
$$
\Delta\mathbf{q} = \mathbf{J}^\top (\mathbf{J} \mathbf{J}^\top + \lambda \mathbf{I})^{-1} \mathbf{e}
$$
$$
\mathbf{q}_{i+1} = \mathbf{q}_i + \alpha \cdot \Delta\mathbf{q}
$$
where $\lambda > 0$ is a damping factor to ensure invertibility near singularities, and $\alpha$ is a step size limiting the update to prevent divergence. The process iterates until $\|\mathbf{e}\| < \epsilon_{\text{tol}}$.

**b) Model Predictive Control Formulation:**

The MPC module receives the joint-space reference $\mathbf{q}_{\text{ref}}$ from the IK solver and computes optimal control torques. We define a discrete-time dynamics model for prediction. The state vector is $\mathbf{x} = [\mathbf{q}^\top, \dot{\mathbf{q}}^\top]^\top$, and the control input is $\boldsymbol{\tau} \in \mathbb{R}^3$.

A simplified double-integrator model is used for prediction:
$$
\dot{\mathbf{x}} = \begin{bmatrix} \dot{\mathbf{q}} \\ \boldsymbol{\tau} \end{bmatrix}
$$
The corresponding discrete-time dynamics function $f(\mathbf{x}_k, \boldsymbol{\tau}_k)$ is:
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \Delta t \cdot \begin{bmatrix} \dot{\mathbf{q}}_k \\ \boldsymbol{\tau}_k \end{bmatrix}
$$

The MPC solves a finite-horizon optimal control problem over a prediction horizon $N$:
$$
\min_{\boldsymbol{\tau}_{0:N-1}} \quad \sum_{k=0}^{N-1} \left( \| \mathbf{x}_k - \mathbf{x}_{\text{ref}} \|^2_{\mathbf{Q}} + \| \boldsymbol{\tau}_k \|^2_{\mathbf{R}} \right) + \| \mathbf{x}_N - \mathbf{x}_{\text{ref}} \|^2_{\mathbf{Q}_N}
$$
$$
\text{subject to:} \quad \mathbf{x}_{k+1} = f(\mathbf{x}_k, \boldsymbol{\tau}_k), \quad \mathbf{x}_0 = \mathbf{x}(t), \quad \boldsymbol{\tau}_{\text{min}} \leq \boldsymbol{\tau}_k \leq \boldsymbol{\tau}_{\text{max}}
$$
where $\mathbf{x}_{\text{ref}} = [\mathbf{q}_{\text{ref}}^\top, \mathbf{0}^\top]^\top$ is the target state (desired position with zero velocity), and $\mathbf{Q}$, $\mathbf{R}$, $\mathbf{Q}_N$ are positive definite weight matrices. The first optimal control input $\boldsymbol{\tau}^{*}_0$ is applied to the system.

**Limitation:** While this controller is effective, the computational burden of solving the IK problem iteratively and the MPC optimization problem in real-time (using CasADi and IPOPT) is significant, limiting its use in applications requiring high control frequencies or on computationally constrained platforms.

---

#### 3. Dataset Generation Pipeline

To enable imitation learning, we generate a dataset $\mathcal{D} = \{ (\mathbf{X}_i, \boldsymbol{\tau}_i^{\text{MPC}}) \}_{i=1}^N$ by recording the closed-loop behavior of the expert MPC controller, where $\mathbf{X}_i = [\mathbf{x}_{i-H+1}, \ldots, \mathbf{x}_{i-1}, \mathbf{x}_i]$ represents a sequence of $H$ consecutive states leading up to time $i$. The process for each episode is:
1.  **Target Sampling:** A reachable end-effector target $\mathbf{p}_{\text{des}}$ is sampled within the workspace, validated using the IK solver.
2.  **Reference Calculation:** The IK solver computes the corresponding joint-space reference $\mathbf{q}_{\text{ref}}$.
3.  **Expert Demonstration:** The MPC controller generates torque commands $\boldsymbol{\tau}^{\text{MPC}}_k$ to track $\mathbf{q}_{\text{ref}}$, given the current state $\mathbf{x}_k$.
4.  **Data Collection:** After accumulating a history of $H$ states, successful state sequence-action pairs $(\mathbf{X}_k, \boldsymbol{\tau}_k^{\text{MPC}})$ are stored, explicitly excluding the gravity compensation term to learn only the MPC's corrective actions.

---

#### 4. Objective: Neural Network Imitation of MPC Policy

We aim to train a parameterized policy $\pi_\theta$ that approximates the expert MPC controller, mapping sequences of historical states to future control actions. The learning objective is to minimize the mean-squared error between the neural network's predictions and the expert's actions over the collected dataset:

$$
\min_{\theta} \quad \frac{1}{|\mathcal{D}|} \sum_{(\mathbf{X}_i, \boldsymbol{\tau}_i) \in \mathcal{D}} \| \pi_\theta(\mathbf{X}_i) - \boldsymbol{\tau}_i^{\text{MPC}} \|^2_2
$$

We will investigate several neural network architectures for $\pi_\theta$, each designed to handle the temporal nature of the input state sequences:
-   **Feedforward Network (FFN):** Flattens the state sequence $\mathbf{X}_i \in \mathbb{R}^{H \times 6}$ into a single vector and processes it through dense layers. This serves as a simple baseline.
-   **Recurrent Neural Network (RNN):** Utilizes LSTM or GRU layers to process the state sequence $\mathbf{X}_i$, capturing temporal dependencies through hidden states.
-   **Transformer:** Employs self-attention mechanisms to model complex relationships between states across the entire history window, potentially capturing longer-term dependencies than RNNs.

---

#### 5. Key Challenges

-   **Generalization:** The policy must perform robustly on state distributions not encountered in the training data, especially near the workspace boundaries and singularities.
-   **Stability and Safety:** Unlike the constrained MPC, the neural network policy lacks inherent stability guarantees. Ensuring safe operation is a critical challenge.
-   **Data Efficiency:** The data collection process is computationally expensive. The learning algorithm must be sample-efficient.
-   **Performance Parity:** The surrogate policy must match the MPC's performance in terms of tracking error and settling time to be a viable replacement.

---

#### 6. Expected Outcomes

The successful outcome of this project will be a neural network policy $\pi_\theta$ that:
-   **Achieves real-time performance:** Executes orders of magnitude faster than the online MPC optimization.
-   **Maintains control performance:** Exhibits tracking error and control effort comparable to the original MPC controller across the workspace.
-   **Leverages temporal information:** Effectively utilizes past state information to improve control decisions.
-   **Generalizes effectively:** Performs robustly when presented with new, reachable target points and state trajectories.
