#import "@preview/charged-ieee:0.1.4": ieee

#set page(numbering: "1")
#show cite: set text(blue)

#show: ieee.with(
  title: [Behavior Cloning of MPC for 3-DOF Robotic Manipulators],
  abstract: [
    While Model Predictive Control (MPC) provides strong stability and robustness, it imposes a significant computational burden on real-time systems and resource-constrained devices. This paper investigates the application of Behavior Cloning to approximate MPC policies for the real-time control of a 3-degree-of-freedom (3-DOF) robotic manipulator. We present a baseline controller combining Inverse Kinematics (IK) with MPC and evaluate a spectrum of neural network architectures, ranging from classical regression algorithms to complex deep learning models including Deep MLPs, RNNs, and Transformers to derive computationally efficient surrogate policies. We analyze generalization capabilities, stability considerations, and the trade-offs inherent in different architectural choices. Our empirical study employs both online and offline evaluations to assess performance regarding accuracy, computational efficiency, and fidelity to the original MPC policy.
  ],
  authors: (
    (
      name: "Theo Guegan",
      department: [21229606],
      organization: [University of Waterloo],
      email: "tguegan@uwaterloo.ca"
    ),
    (
      name: "Wen Jie Dexter Teo",
      department: [21230211],
      organization: [University of Waterloo],
      email: "d2teo@uwaterloo.ca"
    ),
  ),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Figure],
)

= Introduction

Model Predictive Control (MPC) has been widely used for robotic manipulation @zhou2022modelpredictivecontroldesign, offering an optimal control strategy with strong stability and robustness. However, the computational cost of MPC for solving the optimization problems limits its applicability for both real-time systems and resource-constrained devices. Neural networks, with their diverse architectures, offer a promising and computationally efficient alternative for approximating MPC policies @gonzalez2024neuralnetworksfastoptimisation. We consider a 3-degree-of-freedom (3-DOF) robotic manipulator operating in a MuJoCo simulation environment. The simulation environment provides a realistic and controllable environment for testing and evaluating the proposed methodology. MuJoCo also handles gravity compensation and joint friction, allowing us to simplify the control problem and focus on the learning aspect. The control objective centers on driving the end-effector (EE) to reach a 3D cartesian target position within the robot's reachable workspace. Inspired by the recent usage of imitation learning for complex controls @deAPorto2025, we present a complete data generation pipeline for collecting high-quality demonstrations of the desired behavior and an empirical evaluation of both feedforward and recurrent neural networks for policy learning. Our experiment focuses on minimizing the control error and testing the ability of the learned policy to generalize in the simulation environment.

= Problem Formulation

== System Description

We consider a 3-degree-of-freedom (3-DOF) robotic manipulator defined by generalized coordinates $q = [q_1, q_2, q_3]^T in RR^3$, representing joint angles, and their time derivatives $dot(q) in RR^3$. The full observable state at discrete time step $k$ is

$
  x_k = [q_k^T, dot(q)_k^T]^T in RR^6
$

The manipulator operates in a MuJoCo simulation environment (@fig:3dof-arm-mujoco) governed by rigid-body dynamics with gravity compensation. The control objective is to drive the end-effector (EE) to track randomly sampled, reachable 3D Cartesian target positions $p_"des" in RR^3$ within the robot's workspace $cal(W) subset RR^3$.

#grid(
  columns: 2,
  grid.cell([
    #figure(
    image("figures/3dof-arm-mujoco.png", width: 70%),
    caption: [3-DOF Arm in MuJoCo],
  )<fig:3dof-arm-mujoco>
  ]),
  grid.cell([
    #figure(
    image("figures/3dof-arm-schema.png", width: 75%),
    caption: [3-DOF Arm Schema @NgocSon2016],
  )<fig:3dof-arm-schema>
  ])
)


= Baseline Controller: MPC with Inverse Kinematics

Our baseline controller uses a hierarchical architecture combining an Inverse Kinematics (IK) module and a Model Predictive Control (MPC) module. The IK module computes the joint angles required to achieve the desired end-effector position, while the MPC module optimizes the joint velocities to minimize the control error.

== Inverse Kinematics Formulation

The IK module translates desired end-effector positions into feasible joint-space configurations. Let $p(q): RR^3 -> RR^3$ denote the forward kinematics mapping. The Cartesian error is defined as:

$
 e = p_"des" - p(q)
$

We solve the IK problem using the Jacobian transpose method with Damped Least Squares (DLS) for numerical stability near singularities. The iterative update rule is

$
  Delta q = J^T (J J^T + lambda^2 I)^(-1) e
$

With $J(q) = dif(partial p, partial q) in RR^(3 times 3)$ is geometric Jacobian and $lambda$ is the damping factor.

To prevent overshooting or divergence, the joint update is clamped to a maximum norm relative to the step size $alpha in [0,1]$ :
$
  Delta q = cases(
    Delta q "if" norm(Delta q) <= alpha,
    Delta q * alpha / norm(Delta q) "otherwise",
  )
$

Finally, joint angles are wrapped to avoid numerical drift:

$
  q_i <- "atan2"(sin(q_i), cos(q_i))
$

== Model Predictive Control Formulation

The MPC module is given the desired joint angles $q_"des" in RR^3$ from the IK module and computes optimal control torques $tau_"MPC"$ with a specified prediction horizon. We simplify the system dynamics for the MPC formulation by assuming unit inertia and neglecting Coriolis effects, justified by MuJoCo's compensation of complex dynamics including gravity and friction. This yields a double-integrator model where the control input $tau_"MPC"$ effectively commands joint acceleration:

$
  dot.double(q) = tau_"MPC"
$

With $x = [q, dot(q)]^T in RR^6$ and discrete-time dynamics:

$
  x_(k+1) = x_k + Delta t * vec(dot(q)_k, tau_"MPC,k", delim: "[") = f(x_k,tau_k)
$

The MPC solves a finite-horizon optimal control problem with quadratic cost function:

$
  min_(tau_(0:N-1)) quad sum_(k=0)^(N-1) (norm(x_k - x_"ref")_"Q"^2 + norm(tau_k)_"R"^2) + norm(x_N - x_"ref")_"Q"_"N"^2
$<cost-function>

subject to:

$
  x_"k+1" = f(x_k,tau_k), quad x_0 = x(t), quad tau_min <= tau_k <= tau_max
$

Where $x_"ref" = [q_"des", 0^T]^T$ is the target state, and $bold(Q), bold(R), bold(Q_N)$ are positive definite matrices. The optimization problem is solved using CasADI @Andersson2018 with OSQP optimization solver.

= Data Generation Pipeline

To enable behavior cloning from the expert IK–MPC controller, we generate a dataset of joint angles, joint velocities, target positions, and predicted torques from the closed-loop MuJoCo simulation. The process for each episode is as follows:

== Collection process

1. Target sampling: A reachable end-effector target $p_"des" in RR^3$ is sampled within the workspace $cal(W)$ using cylindrical coordinates to sample a radius $r$ and height $z$ uniformly within the workspace bounds.
2. The IK solver computes the corresponding joint-space reference $q_"des"$.
3. The MPC controller generates torque commands $tau_"MPC"$ to achieve the desired joint angles and velocities, given the current state $x_k$ and a specified prediction horizon $N$.
4. For each time step $k$, we record the current state $[q_1, q_2, q_3, dot(q)_1, dot(q)_2, dot(q)_3] in RR^6$, the target $p_"des" in RR^3$, and the predicted torque $tau_"MPC" in RR^3$.
5. We step the simulation until the end of the episode or until the target is reached using $tau = tau_"MPC" + tau_"env"$ (where $tau_"env"$ represents the forces computed by MuJoCo to compensate for Coriolis, centrifugal, and gravitational effects, ensuring the simplified dynamics model in the MPC controller remains valid; MuJoCo bias force: `mjData.qfrc_bias`).

During this process, if either the MPC controller or the IK solver fails to converge, we discard the data for that time step to retain only high-quality samples.

== Dataset Structure

After generation, the dataset is stored in an episode-based format within a HDF5 file.

#figure(
  align(left)[
    - episodes
      - ep_0000
        - `states`: $(T_0 times 6)$
        - `targets`: $(T_0 times 3)$
        - `actions`: $(T_0 times 3)$
      - ep_0001
        - `states`: $(T_1 times 6)$
        - `targets`: $(T_1 times 3)$
        - `actions`: $(T_1 times 3)$
      - ...
  ],
  caption: [Hierarchical HDF5 dataset structure],
)<fig:dataset-structure>

This hierarchical format preserves the temporal integrity of each trial, allowing us to process the data differently depending on the model architecture. The raw data is loaded via a custom MPCDataset class, which constructs the input feature vector $x$ by concatenating the state ($RR^6$) and the target ($RR^3$), resulting in a 9-dimensional input vector.

#figure(
  image("figures/random_episode.png", width: 80%),
  caption: [Visualization of a random episode],
)<fig:random_episode>

Depending on the learning algorithm, the data is processed in two different ways according to the model architecture.

=== Flat Formatting

For non-sequential algorithms (e.g., MLPs, Random Forests), temporal dependencies are discarded to maximize sample efficiency. We treat every timestep t from every episode as an independent sample (i.i.d).

$
  X_"flat" in RR^(N times 9), quad Y_"flat" in RR^(N times 3)
$

Where $N = sum_(i=0)^(E)T_i$ is the total number of timesteps across all episodes.

=== Sequential Formatting

For time-series algorithms (e.g., LSTM, GRU, Transformer), preserving the temporal dependencies is crucial. We treat every episode as a sequence of timesteps, where each timestep is a sample.

$
  X_"seq" in RR^(E times T times 9), quad Y_"seq" in RR^(E times T times 3)
$

Where $E$ is the number of episodes and $T$ is the number of timesteps per episode.

== Data Preprocessing

Our goal is to develop a robust and reliable controller that can handle uncertainties and disturbances in the system. To this end, we introduce small Gaussian noise to both the input state $[q_1, q_2, q_3, dot(q)_1, dot(q)_2, dot(q)_3]$ and the output action $tau_"MPC"$. This noise simulates real-world conditions such as sensor noise, actuator noise, and environmental disturbances. Because the data generation pipeline can produce an arbitrary number of samples, we collect a large dataset for training and increase the number of samples until the validation loss plateaus or computational limits are reached. We use an 80/20 split, where 80% of the data is used for training and 20% for validation.

= Neural Network Architecture

We first formulated the learning problem as a regression task, where the goal was to predict the torque $tau_"MPC"$ given the current state $x_k$ and the target $p_"des"$. We aimed to minimize the error between the neural network policy $pi_theta$ and the expert MPC actions:

$
  min_theta quad L(pi_theta(X), tau_"MPC")
$

where $L$ is a loss function that measures the difference between the predicted torque and the expert MPC torque. We investigated a range of models, from traditional machine learning to deep learning architecture, to understand their effectiveness in approximating the MPC policy. For this task, we compared the performance with 2 different loss functions:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

== Regression Baselines

As baselines, we evaluated several models from the scikit-learn library to establish performance benchmarks. These models operate on the flat dataset, treating each timestep as an independent sample. We included tree-based regressors (Random Forest and Gradient Boosting) and a shallow MLP regressor as a lightweight baseline to compare against our deeper custom architectures.

== Custom Multi-Layer Perceptron (MLP)

We implemented a custom feedforward network to explore the impact of model capacity on cloning accuracy. This model processes the flat input vector through a series of fully connected linear layers with ReLU activations. We conducted an architectural search by varying:
  - Depth: Number of hidden layers.
  - Width: Number of neurons per layer
This memory-less architecture captures and learns directly the mapping from the current state and target to the required control action.

== Time Series Models

To leverage the temporal structure of our system, we also employed sequential architectures. Unlike the flat models, these architectures maintain a history of past inputs and outputs to predict the current torque @PonKumar2018.

=== Recurrent Neural Networks (RNNs)

We evaluated both Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) networks. These models maintain a hidden state $h_t$​ that summarizes the history of the episode up to time $t-1$. For all the recurrent models, the final hidden state was passed through a linear output layer to produce the predicted torque $pi_theta(X)$. Here again, we varied the number of hidden units per layer and the number of layers.

$
  h_t = "RNN"(x_t, h_(t-1)), quad pi_theta(X) = "Linear"(h_t)
$

== Transformer Architecture

If time permits, we will investigate a Transformer model. Unlike RNNs, the multi-head self-attention mechanisms used to weigh the importance of different past timesteps may result in better performance due to their ability to capture long-range dependencies.

= Evaluation Methodology

We evaluated the learned policies using a combination of offline and online metrics:

== Offline Metrics
We used Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to measure the average deviation of the predicted torques from the expert torques.

$
  "MAE"_j = 1/N sum_(i=0)^n abs(tau_"MPC,i,j" - pi_theta (X_i)_j)
$
$
  "RMSE"_j = 1/N sum_(i=0)^n norm(tau_"MPC,i,j" - pi_theta (X_i)_j)^2_2
$
 We also evaluated the percentage of predictions where the sign of each torque component matches the expert's using Direction Accuracy (DA). This assesses whether the model correctly identifies the direction of joint acceleration.
$
  "DA" = 1/(3N) sum_(i=0)^N sum_(j=1)^3 II ("sign"(tau_"MPC,i,j") = "sign"(pi_theta (X_i)_j))
$

Finally, we measured the proportion of variance in the expert's action that is explained by our model, using explained variance. It is a normalized, scale-invariant metric for comparing performance defined as follows:
$
  "Explained Variance" = 1 - "Var"(tau_"MPC" - pi_theta(X))/"Var"(tau_"MPC")
$

== Online Metrics

To assess the closed-loop performance, we deployed the trained policies in our simulation environment and evaluated them on the following criteria:

- Success Rate: The percentage of episodes where the end-effector's final position converges within a specified tolerance $epsilon$ of the target
$
  norm(p_"final" - p_"des")_2 < epsilon
$
- Average Position Error: The mean Euclidean distance between the end-effector and the target across the entire trajectory. This verifies that the model actively minimizes error rather than just drifting near the goal.
- Computational Time Efficiency: The average inference latency per control step. We compare this against the baseline MPC solution time to confirm that the neural networks achieve higher control frequencies.
- Computational Cost: Average CPU utilization during operation, ensuring the surrogate model is sufficiently lightweight for potential deployment on embedded systems.

= Results

== Regression Baseline

For our regression baseline with scikit-learn, we evaluated several standard regression algorithms using the collected offline dataset. The dataset consists of 1715 episodes, which were flattened to remove temporal dependencies, resulting in a total of $N = 85,789$ samples. The input feature space $X in RR^9$ consists of the robot's current state and target coordinates, while the output target $Y in RR^3$ corresponds to the applied joint actions (torques). The data was partitioned into a training set ($80%$) and a test set ($20%$) via random shuffling. To ensure the statistical significance of the reported metrics, each model was trained and evaluated over 5 independent runs. @table:regression_baseline summarizes the performance across Mean Squared Error (MSE), Mean Absolute Error (MAE), Explained Variance, and Directional Accuracy. We also tried scaling our features using Scikit-Learn's StandardScaler, which did not significantly improve performance.

#let model_col(name) = strong(name)
#let vector_val(v) = text(size: 0.7em, $mono([#v])$)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 5pt,
    align: (col, row) => (if col == 0 { left } else { center + horizon }),
    stroke: (x, y) => (
      top: if y == 0 { 1pt } else if y == 1 { 0.5pt } else { 0pt },
      bottom: 1pt,
    ),

    table.header(
      [*Model*],
      [*MSE* \ (Mean $plus.minus$ CI)],
      [*MAE* \ (Mean $plus.minus$ CI)],
      [*Expl. Var*],
      [*Dir. Acc.*],
      [*MSE/Torque*]
    ),

    model_col("Ridge"), $8.729 plus.minus 0.098$, $1.418 plus.minus 0.009$, [0.221], [0.681], vector_val("4.24, 5.54, 16.41"),
    model_col("Random Forest"), $0.097 plus.minus 0.003$, $0.111 plus.minus 0.002$, [0.991], [0.999], vector_val("0.05, 0.08, 0.16"),
    model_col("MLP Regressor"), $0.053 plus.minus 0.013$, $0.094 plus.minus 0.009$, [0.994], [0.938], vector_val("0.05, 0.03, 0.07"),
    model_col("Gradient Boosting"), $1.017 plus.minus 0.059$, $0.243 plus.minus 0.005$, [0.843], [0.996], vector_val("2.30, 0.10, 0.65"),
    model_col("KNN Regressor"), $0.237 plus.minus 0.038$, $0.091 plus.minus 0.002$, [0.976], [0.999], vector_val("0.20, 0.11, 0.40"),
  ),
  caption: [Comparison of regression algorithms on the validation set (averaged over 5 runs).]
)<table:regression_baseline>

The results highlight the inherent non-linearity of the inverse dynamics mapping. Linear Regression failed to capture the underlying relationship, exhibiting high variance across all torque dimensions. In contrast, non-linear methods performed significantly better. The MLP Regressor achieved the lowest overall Mean Squared Error ($0.053$), indicating its superior capability in minimizing large control deviations, which is critical for preventing hardware damage. While KNN Regressor achieved the highest Directional Accuracy ($99.87%$) and lowest MAE, its higher MSE suggests it suffers from occasional large prediction errors (outliers). Finally, different SVM models with linear and radial basis function (RBF) kernels were evaluated. However, as mentioned in the documentation, RBF kernels cannot scale with that many samples, whereas linear kernels yielded poor performance.


#figure(
  image("figures/mse_per_torque.png", width: 90%),
  caption: [Mean Squared Error per Torque],
)<fig:mse_per_torque>

== Custom MLP
== Time Series Models
== Online evaluation (MuJoCo)

#pagebreak()

= Future Work

This work establishes a foundation for behavior cloning of MPC on 3-DOF manipulators, which can be extended in several directions. Firstly, the scalability of the approach should be evaluated on robotic manipulators with higher degrees of freedom (e.g., 6-DOF). This is to assess how the method handles increased state and action space dimensionality. Second, to advance towards real-world deployment, the methodology should be extended to handle more complex control scenarios. It could be interesting to investigate the cloning of a non-linear MPC which is capable of handling more complex dynamics.

From a methodological perspective, exploring advanced neural network architectures represents a promising direction. Transformer models, with their self-attention mechanisms, could be investigated for their ability to capture complex, long-range dependencies. Furthermore, the Legendre Memory Unit (LMU) @NEURIPS2019_952285b9, developed at the University of Waterloo, offers a complementary, principled approach to continuous time memory, which may prove to be well-suited for the robotic system's underlying dynamics. Inverse reinforcement learning @deAPorto2025 may prove to be an efficient alternative to learn the underlying MPC cost function.

= Acknowledgments

This project is a final project for the course "Foundations of Artificial Intelligence" - SYDE522. We would like to thank our instructor Terry Stewart for his guidance and support.
