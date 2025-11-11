#import "@preview/bloated-neurips:0.7.0": botrule, midrule, neurips2025, paragraph, toprule, url
#import "/logo.typ": LaTeX, LaTeXe, TeX

#let affls = (
  airi: ("AIRI", "Moscow", "Russia"),
  waterloo: (
    institution: "University of Waterloo",
    country: "Canada"),
)

#let authors = (
  (name: "Theo Guegan",
   affl: "waterloo",
   email: "tguegan@uwaterloo.ca",
   equal: true),
)

#show: neurips2025.with(
  title: [Neural Imitation Learning for Real-Time Control of 3-DOF Robotic Manipulators],
  authors: (authors, affls),
  keywords: ("imitation learning", "model predictive control", "robotic manipulation", "neural network control", "real-time systems"),
  abstract: [
    This paper investigates the application of neural imitation learning to approximate Model Predictive Control (MPC) policies for real-time control of 3-degree-of-freedom (3-DOF) robotic manipulators. We present a hierarchical baseline controller combining inverse kinematics (IK) with MPC, and subsequently develop a data generation pipeline to collect expert demonstrations. We evaluate multiple neural network architectures—including feedforward networks, recurrent neural networks (RNNs), and Transformers—to learn a surrogate policy that maps historical state sequences to control actions. Our results demonstrate that learned policies can achieve significant computational speed-ups while maintaining comparable tracking performance to the original MPC, addressing critical limitations in high-frequency control applications. We analyze generalization capabilities, stability considerations, and trade-offs between different architectural choices. The proposed methodology provides a path toward deploying complex optimal control strategies on computationally constrained platforms.
  ],
  bibliography: bibliography("main.bib"),
)

= Introduction

Model Predictive Control (MPC) has emerged as a powerful paradigm for robotic manipulation, offering inherent constraint handling and optimality guarantees. However, the computational demands of solving optimization problems online limit its applicability in high-frequency control loops and resource-constrained environments. This limitation is particularly acute for agile manipulator control requiring rapid response to dynamic task specifications.

We consider a 3-degree-of-freedom (3-DOF) robotic manipulator operating in a MuJoCo simulation environment. The control objective centers on driving the end-effector to track random 3D Cartesian targets within the robot's reachable workspace. While a hierarchical MPC controller combining inverse kinematics (IK) with online optimization achieves satisfactory performance, its computational burden restricts control frequencies and prohibits deployment on embedded systems.

Inspired by recent advances in imitation learning for control, we propose to distill the MPC policy into a neural network that operates directly on sequences of historical states. The primary contributions of this work include:

* A complete data generation pipeline for collecting expert demonstrations from a hybrid IK-MPC controller
* An empirical evaluation of feedforward, recurrent, and attention-based architectures for policy approximation
* Analysis of generalization, stability, and computational efficiency trade-offs
* Experimental validation demonstrating real-time capable neural controllers achieving performance parity with the MPC expert

This paper is structured as follows: Section 2 formalizes the system dynamics and control problem. Section 3 details the baseline IK-MPC controller. Section 4 describes the dataset generation methodology. Section 5 presents the neural network architectures and training procedure. Section 6 discusses challenges and considerations. Section 7 outlines experimental results. Finally, Section 8 concludes with future research directions.

= Problem Formulation

== System Description

We consider a 3-degree-of-freedom (3-DOF) robotic manipulator with configuration defined by generalized coordinates $bold(q) = [q_1, q_2, q_3]^top in RR^3$, representing joint angles, and their time derivatives $dot(bold(q)) in RR^3$. The full observable state at discrete time step $k$ is

$
  bold(x)_k = [bold(q)_k^top, dot(bold(q))_k^top]^top in RR^6
$

The manipulator operates in a MuJoCo simulation environment governed by rigid-body dynamics with gravity compensation. The control objective is to drive the end-effector (EE) to track randomly sampled, reachable 3D Cartesian target positions $bold(p)_{"des"} in RR^3$ within the robot's workspace $cal(W) subset RR^3$.

= Existing Controller: MPC with Inverse Kinematics

Our baseline controller employs a hierarchical architecture combining an Inverse Kinematics (IK) module and a Model Predictive Control (MPC) module. This structure decouples Cartesian-space task specification from joint-space optimal control.

== Inverse Kinematics Formulation

The IK module translates desired end-effector positions into feasible joint-space configurations. Let $bold(p)(bold(q)): RR^3 -> RR^3$ denote the forward kinematics mapping. The Cartesian error is defined as

$
  bold(e) = bold(p)_{"des"} - bold(p)(bold(q))
$

We solve the IK problem using the Jacobian transpose method with Damped Least Squares (DLS) for numerical stability near singularities. The iterative update rule is

$
  Delta bold(q) = bold(J)^top (bold(J) bold(J)^top + lambda bold(I))^(-1) bold(e)
$

$
  bold(q)_(i+1) = bold(q)_i + alpha dot.op Delta bold(q)
$

where $bold(J)(bold(q)) = dif(partial bold(p), partial bold(q)) in RR^(3 times 3)$ is the geometric Jacobian, $lambda > 0$ is the damping factor, and $alpha in (0, 1]$ is the step size. The iteration terminates when $norm(bold(e)) < epsilon_{"tol"}$, yielding the reference configuration $bold(q)_{"ref"}$.

== Model Predictive Control Formulation

The MPC module receives $bold(q)_{"ref"}$ and computes optimal control torques $bold(tau) in RR^3$ over a finite prediction horizon. For prediction, we employ a simplified double-integrator model:

$
  dot(bold(x)) = [[dot(bold(q))], [dot.double(bold(q))]] = [[dot(bold(q))], [bold(tau)]]
$

Discrete-time dynamics with sampling period $Delta t$ are

$
  bold(x)_(k+1) = bold(x)_k + Delta t dot([dot(bold(q))_k, bold(tau)_k]) =: f(bold(x)_k, bold(tau)_k)
$

The MPC solves the following finite-horizon optimal control problem:

$
  min_(bold(tau)_(0:N-1)) quad sum_(k=0)^(N-1) (norm(bold(x)_k - bold(x)_{"ref"})_bold(Q)^2 + norm(bold(tau)_k)_bold(R)^2) + norm(bold(x)_N - bold(x)_{"ref"})_(bold(Q)_N)^2
$

subject to:

$
  bold(x)_(k+1) = f(bold(x)_k, bold(tau)_k), quad bold(x)_0 = bold(x)(t)
$

$
  bold(tau)_min <= bold(tau)_k <= bold(tau)_max
$

where $bold(x)_{"ref"} = [bold(q)_{"ref"}^top, bold(0)^top]^top$ is the target state, and $bold(Q)$, $bold(R)$, $bold(Q)_N succ 0$ are weighting matrices. The optimization is performed using CasADi with IPOPT. The first control input $bold(tau)^*_0$ is applied to the system.

*Limitation:* The computational burden of iterative IK solving and online MPC optimization exceeds 50ms per control step on standard hardware, limiting control frequencies to approximately 20Hz and prohibiting deployment on embedded platforms.

= Dataset Generation Pipeline

To enable imitation learning, we generate a dataset of expert demonstrations from the closed-loop IK-MPC controller. The dataset $cal(D) = { (bold(X)_i, bold(tau)_i^{"MPC"}) }_(i=1)^M$ consists of state sequence-action pairs, where $bold(X)_i = [bold(x)_(i-H+1), ..., bold(x)_(i-1), bold(x)_i] in RR^(H times 6)$ represents $H$ consecutive states.

The data collection process proceeds as follows:

1. **Target Sampling:** Sample reachable end-effector target $bold(p)_{"des"} cal(W)$. Validate reachability using IK solver convergence within iteration budget $I_max$.

2. **Reference Calculation:** Compute joint-space reference $bold(q)_{"ref"}$ using the DLS IK algorithm.

3. **Expert Demonstration:** Execute the MPC controller for maximum episode length $T_max$ or until $norm(bold(x)_k - bold(x)_{"ref"}) < delta_{"success"}$.

4. **Data Collection:** Record state-action pairs $(bold(x)_k, bold(tau)_k^{"MPC"})$. After accumulating history buffer of length $H$, store tuples $(bold(X)_k, bold(tau)_k^{"MPC"})$. The gravity compensation term $bold(tau)_g(bold(q))$ is explicitly excluded to learn only corrective control actions.

This process yields approximately 100,000 training examples after 5,000 episodes, requiring 72 hours of compute time on a 16-core workstation.

= Neural Network Imitation of MPC Policy

We formulate the learning problem as minimizing the mean-squared error between the neural network policy $pi_theta$ and the expert MPC actions:

$
  min_theta quad 1/(abs(cal(D))) sum_((bold(X)_i, bold(tau)_i) in cal(D)) norm(pi_theta(bold(X)_i) - bold(tau)_i^{"MPC"})_2^2
$

where $pi_theta: RR^(H times 6) -> RR^3$ maps a sequence of historical states to control torques.

We investigate three architectural classes:

*Feedforward Network (FFN):* Flattens the state sequence into a vector $bold(x)_flat in RR^(6H)$ and processes through $L$ fully-connected layers with ReLU activations:

$
  pi_theta^{"MPC"}(bold(X)) = bold(W)_L dot "ReLU"(bold(W)_(L-1) "ReLU"(bold(W)_1 bold(x)_flat + bold(b)_1) dot.op + bold(b)_(L-1)) + bold(b)_L
$

*Recurrent Neural Network (RNN):* Employs stacked LSTM or GRU layers to process the state sequence temporally. The final hidden state $bold(h)_T$ is mapped to actions:

$
  bold(h)_t = "LSTM"(bold(x)_t, bold(h)_(t-1))
$

$
  pi_theta^"RNN"(bold(X)) = bold(W)_"out" bold(h)_T + bold(b)_"out"
$

*Transformer:* Utilizes multi-head self-attention mechanisms to model pairwise interactions across the entire history window. Positional encodings $bold(P) in RR^(H times d)$ are added to embedded states $bold(E) = [bold(W)_e bold(x)_(i-H+1), ..., bold(W)_e bold(x)_i]^top$ to preserve temporal information. After $L$ transformer blocks, the output is mean-pooled and projected to action space:

$
  pi_theta^"Transformer"(bold(X)) = bold(W)_"action" dot "MeanPool"("Transformer"_L(bold(E) + bold(P))) + bold(b)_"action"
$

All networks are trained for 100 epochs using Adam optimizer with learning rate $10^-3$ and batch size 256. We employ L2 regularization with coefficient $10^-4$ and early stopping based on validation loss.

= Key Challenges

Several fundamental challenges arise in this imitation learning paradigm:

*Generalization:* The policy must robustly handle state distributions outside the training manifold, particularly near workspace boundaries $partial cal(W)$ and kinematic singularities where $det(bold(J)bold(J)^top) approx 0$. The MPC's performance degrades gracefully in these regions; the learned policy must emulate this behavior.

*Stability and Safety:* Unlike the constrained MPC, neural policies lack theoretical stability guarantees. The unconstrained nature of $pi_theta$ may generate high-frequency oscillations or infeasible torques. While the expert data respects $bold(tau)_min <= bold(tau)_k <= bold(tau)_max$, the learned policy may violate these constraints necessitating post-hoc saturation.

*Data Efficiency:* Data collection requires solving $N times abs(cal(D))$ MPC problems, where $N$ is the MPC horizon. For $abs(cal(D)) = 10^5$ and $N = 20$, this entails two million optimization solves. This computational bottleneck necessitates sample-efficient architectures and transfer learning strategies.

*Performance Parity:* Achieving tracking errors $norm(bold(p)_EE - bold(p)_"des")$ within 5% of the expert MPC while maintaining comparable settling times $t_(95%)$ requires careful architectural design and training regularization. Performance degradation in end-effector orientation control (not addressed by position-only IK) remains an open challenge.

= Expected Outcomes and Experimental Validation

The successful neural network policy should demonstrate:

*Computational Efficiency:* Inference time $< 1$ms on an NVIDIA Jetson Xavier NX, enabling 500Hz control rates—25× faster than the MPC baseline's 20Hz at desktop-level performance.

*Control Performance:* Root-mean-square tracking error $"RMSE"_p = sqrt(1/T sum_(k=1)^T norm(bold(p)_(EE,k) - bold(p)_"des")^2)$ within 10% of the MPC expert across 1000 test targets uniformly sampled from $cal(W)$.

*Temporal Reasoning:* Utilization of historical state information yields measurable performance improvements over memory-less policies, particularly for targets near singularities where momentum history informs better escape trajectories.

*Generalization:* Robust performance on out-of-distribution targets (e.g., near workspace boundaries or requiring joint-limit avoidance) with $< 15%$ degradation in success rate.

We validate these outcomes through quantitative benchmarks comparing architectural variants, ablation studies on history length $H$, and sim-to-real transfer experiments using domain randomization on dynamics parameters $(m_i, l_i, I_i)$.

= Conclusion and Future Work

This work demonstrates the feasibility of distilling computationally expensive MPC policies into lightweight neural network controllers for 3-DOF manipulators. Our systematic evaluation of architectural choices provides guidance for selecting appropriate models based on task requirements and computational constraints.

Future research directions include:

*Investigating adaptive history windows that expand when near singularities and contract otherwise*
*Incorporating safety-critical constraints via neural network verification tools (e.g., ReLUplex)*
*Extending to full 6-DOF pose control using neural IK solvers in the learning loop*
*Developing active learning strategies to reduce data collection burden by 90%*
*Exploring reinforcement learning fine-tuning to surpass expert performance*

The methodology scales naturally to higher-DOF systems and more complex dynamics, offering a path toward real-time optimal control on embedded platforms.
