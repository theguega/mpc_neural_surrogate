#import "@preview/bloated-neurips:0.7.0": botrule, midrule, neurips2025, paragraph, toprule, url

#let affls = (
  waterloo-theo: (
    institution: "University of Waterloo",
    department: "21229606",
    country: "Canada"),
  waterloo-dexter: (
    institution: "University of Waterloo",
    department: "",
    country: "Canada"),
)

#let authors = (
  (name: "Theo Guegan",
   affl: "waterloo-theo",
   email: "tguegan@uwaterloo.ca",
   equal: true),
  (name: "Dexter Teo",
   affl: "waterloo-dexter",
   email: "@uwaterloo.ca",
   equal: false),
)

#show: neurips2025.with(
  title: [Behavior Cloning of MPC for 3-DOF Robotic Manipulators],
  authors: (authors, affls),
  keywords: ("Behavior Cloning", "MPC", "Robotics", "Deep Learning"),
  abstract: [
    This paper investigates the application of behavior cloning to approximate Model Predictive Control (MPC) policies for real-time control of a 3-degree-of-freedom (3-DOF) robotic manipulator. We present a baseline controller combining inverse kinematics (IK) with MPC, and evaluate multiple neural network architectures—including feedforward networks, and recurrent neural networks (RNNs) to learn a surrogate policy. We analyze generalization capabilities, stability considerations, and trade-offs between different architectural choices. The proposed methodology provides a path toward deploying complex optimal control strategies on computationally constrained platforms.
  ],
  bibliography: bibliography("main.bib"),
  accepted: true,
)

= Introduction

Model Predictive Control (MPC) has been widely used for robotic manipulation @zhou2022modelpredictivecontroldesign, offering an optimal control strategy with strong stability and robustness. However, the computational cost of MPC for solving the optimization problems limits its applicability for both real-time systems and resource-constrained devices. Neural networks on the other hand can offer a computationally efficient alternative for approximating MPC policies with different architectures @gonzalez2024neuralnetworksfastoptimisation.

We consider a 3-degree-of-freedom (3-DOF) robotic manipulator operating in a MuJoCo simulation environment. The simulation environment provides a realistic and controllable environment for testing and evaluating the proposed methodology. MuJoCo also handles gravity compensation and joint friction, allowing us to simplify the control problem and focus on the learning aspect. The control objective centers on driving the end-effector (EE) to reach a 3D cartesian target position within the robot's reachable workspace.

Inspired by the recent usage of imitation learning for complex controls @deAPorto2025, we present a complete data generation pipeline for collecting high-quality demonstrations of the desired behavior and an empirical evaluation of both feedforward and recurrent neural networks for policy learning. Our experiment focuses on minimizing the control error and testing the ability of the learned policy to generalize in the simulation environment.

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


= Baseline Controller : MPC with Inverse Kinematics

Our baseline controller uses a hierarchical architecture combining an Inverse Kinematics (IK) module and a Model Predictive Control (MPC) module. The IK module computes the joint angles required to achieve the desired end-effector position, while the MPC module optimizes the joint velocities to minimize the control error.

== Inverse Kinematics Formulation

The IK module translates desired end-effector positions into feasible joint-space configurations. Let $p(q): RR^3 -> RR^3$ denote the forward kinematics mapping. The Cartesian error is defined as :

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

The MPC modules is given the desired joint angles $q_"des" in RR^3$ from the IK module and computes optimal control torques $tau_"MPC"$ with a specified prediction horizon. We can simplify our system dynamics and represent it as a simplified double-integrator model as MuJoCo is used to compensate dynamics including gravity or joint friction.

#pagebreak()

Our simplified dynamic system can be defined as :

$
  dot.double(q) = tau_"MPC"
$

With $x = [q, dot(q)]^T in RR^6$ and discrete-time dynamics :

$
  x_(k+1) = x_k + Delta t * vec(dot(q)_k, tau_"MPC,k", delim: "[") = f(x_k,tau_k)
$

where $x = [q, dot(q)]^T in RR^6$

The MPC solves a finite-horizon optimal control problem with quadratic cost function :

$
  min_(tau_(0:N-1)) quad sum_(k=0)^(N-1) (norm(x_k - x_"ref")_"Q"^2 + norm(tau_k)_"R"^2) + norm(x_N - x_"ref")_"Q"_"N"^2
$<cost-function>

subject to :

$
  x_"k+1" = f(x_k,tau_k), quad x_0 = x(t), quad tau_min <= tau_k <= tau_max
$

Where $x_"ref" = [q_"des", 0^T]^T$ is the target state, and $bold(Q), bold(R), bold(Q_N)$ are positive definite matrices. The optimization problem is solved using CasADI @Andersson2018 with IPOPT @Wchter2005 optimization solver.

= Data Generation Pipeline

To enable behavior-cloning from the expert IK-MPC controller, we generate a dataset of joint angles, joint velocities, target, and predicted torques from the closed-loop MuJoCo simulation. The process for each episode is as follows:

1. Target sampling: A reachable end-effector target $p_"des" in RR^3$ is sampled within the workspace $cal(W)$.
2. The IK solver computes the corresponding joint-space reference $q_"des"$.
3. The MPC controller generates torque commands $tau_"MPC"$ to achieve the desired joint angles and velocities, given the current state $x_k$ and a specified prediction horizon $N$.
4. For each time step $k$, we record the current state $[q_1, q_2, q_3, dot(q)_1, dot(q)_2, dot(q)_3]$, the target $p_"des"$, and the predicted torque $tau_"MPC"$.
5. We step the simulation until the end of the episode or until the target is reached using $tau = tau_"MPC" + tau_"env"$ (with $tau_"env"$ from MuJoCo bias force : `mjData.qfrc_bias`).

= Neural Network Architecture

We formulate the learning problem as a regression task, where the goal is to predict the torque $tau_"MPC"$ given the current state $x_k$ and the target $p_"des"$. We want to minimize the error between the neural network policy $pi_theta$ and the expert MPC actions :

$
  min_theta quad L(pi_theta(X), tau_"MPC")
$

where $L$ is a loss function that measures the difference between the predicted torque and the expert MPC torque.

We investigate two different loss functions:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

#set quote(block: true)
#quote(attribution: [@deAPorto2025])[
  From this paper : For the loss metrics, we employed Mean Absolute Error (MAE) for both the regression and classification outputs. Hyperparameter tuning revealed that MAE outperformed other regression loss functions such as mean squared error.
]

3 main architectures to try, from this paper @PonKumar2018 :

- Feed forward network with 1 input state and 1 output state
- RNN with multiple input states and 1 output state (try GRU and LSTM) + FFN support at the end
- Feed forward network with multiple input states and 1 output state (RNN like)

include graph here


= Evaluation Methodology

== Data augmentation

For each batch, randomly perturb the states and/or actions with Gaussian noise.

Add noise to the input state [q1, q2, q3, q̇1, q̇2, q̇3] before feeding it to your neural network.

1. Simulates sensor noise or slight inaccuracies in state observation.
2. Helps the network generalize to real-world scenarios where states are never perfectly measured.

Add Gaussian noise (e.g., N(0, 0.1)) to the output action (torque) τ_mpc before using it as the target for your network.

1. Simulates actuator noise or imperfections in the control signal.
2. Encourages the network to learn a smoother, more robust policy.


== Metrics

- Computational Efficiency
- Control performance (RMSE, MAE)
- Accuracy in simulation (number of successful simulations)
- Direction accuracy with sign of direction of the torque
- explained variance ? (proportion of variance in expert action explained by the model)

$
  "Explained Vairance" = 1 - "Var"(tau_"MPC" - pi_theta(X))/"Var"(tau_"MPC")
$


= Results

== Offline evaluation

== Online evaluation (MuJoCo)

= Future Work

- improving
- extandable to more degree of freedom ? 6-DOF ?
- more complex controller (Non linear MPC for real-life scenarios)
- other methodology :
  - Transfomer or Legendre Memory Unit (LMU) @NEURIPS2019_952285b9
  - Inverse reinforcement learning (IRL) @deAPorto2025

= Acknowledgments

This project is a final project for the course "Foundations of Artificial Intelligence" - SYDE522. We would like to thank our instructor Terry Stewart for his guidance and support.
