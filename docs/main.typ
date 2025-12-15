#import "@preview/charged-ieee:0.1.4": ieee

#set page(numbering: "1")
#show cite: set text(blue)

#show: ieee.with(
  title: [Behavior Cloning of MPC for 3-DOF Robotic Manipulators],
  abstract: [
    While Model Predictive Control (MPC) provides strong stability and robustness, it imposes a significant computational burden on real-time systems and resource-constrained devices. This paper investigates the application of Behavior Cloning to approximate MPC policies for the real-time control of a 3-degree-of-freedom (3-DOF) robotic manipulator. We present a baseline controller combining Inverse Kinematics with MPC and evaluate a spectrum of neural network architectures, ranging from classical regression algorithms to complex deep learning models including Deep MLPs and RNNs, to derive computationally efficient surrogate policies. We analyze generalization capabilities, stability considerations, and the trade-offs inherent in different architectural choices. Our empirical study employs both online and offline evaluations to assess performance regarding accuracy, computational efficiency, and fidelity to the original MPC policy. Our results demonstrate that Behavior Cloning can effectively reduce the computational burden of MPC policies for 3-DOF robotic manipulators, achieving a 3x reduction in inference latency with a 84.98% success rate under relaxed tolerances. Notably, we find that static architectures outperform temporal variants, confirming the sufficiency of instantaneous state observations for this task. However, we observe a precision gap under strict tolerances, which suggest that while Behavior Cloning captures the global optimal trajectory, further research is needed to minimize terminal steady-state error.
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

Model Predictive Control (MPC) has been widely used for robotic manipulation @zhou2022modelpredictivecontroldesign, offering an optimal control strategy with strong stability and robustness. However, the computational cost of MPC for solving the optimization problems limits its applicability for both real-time systems and resource-constrained devices. Neural networks, with their diverse architectures, offer a promising and computationally efficient alternative for approximating MPC policies @gonzalez2024neuralnetworksfastoptimisation. We consider a 3-degree-of-freedom (3-DOF) robotic manipulator operating in a MuJoCo simulation environment. The simulation environment provides a realistic and controllable environment for testing and evaluating the proposed methodology. MuJoCo also handles gravity compensation and joint friction, allowing us to simplify the control problem and focus on the learning aspect. The control objective centers on driving the end-effector to reach a 3D cartesian target position within the robot's reachable workspace. Inspired by the recent usage of imitation learning for complex controls @deAPorto2025, we present a complete data generation pipeline for collecting high-quality demonstrations of the desired behavior and an empirical evaluation of both feedforward and recurrent neural networks for policy learning. Our experiment focuses on minimizing the control error and testing the ability of the learned policy to generalize in the simulation environment.

= Problem Formulation

== System Description

We consider a 3-DOF robotic manipulator defined by generalized coordinates $q = [q_1, q_2, q_3]^T in RR^3$, representing joint angles, and their time derivatives $dot(q) in RR^3$. The full observable state at discrete time step $k$ is

$
  x_k = [q_k^T, dot(q)_k^T]^T in RR^6
$

The manipulator operates in a MuJoCo simulation environment (@fig:3dof-arm-mujoco) governed by rigid-body dynamics with gravity compensation. The control objective is to drive the end-effector to track randomly sampled, reachable 3D Cartesian target positions $p_"des" in RR^3$ within the robot's workspace $cal(W) subset RR^3$.

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

Our baseline controller uses a hierarchical architecture combining an Inverse Kinematics (IK) module and a MPC module. The IK module computes the joint angles required to achieve the desired end-effector position, while the MPC module optimizes the joint velocities to minimize the control error.

== Inverse Kinematics Formulation

The IK module translates desired end-effector positions into feasible joint-space configurations. Let $p(q): RR^3 -> RR^3$ denote the forward kinematics mapping. The Cartesian error is defined as:

$
 e = p_"des" - p(q)
$

We solve the IK problem using the Jacobian transpose method with Damped Least Squares (DLS) for numerical stability near singularities. The iterative update rule is:

$
  Delta q = J^T (J J^T + lambda^2 I)^(-1) e
$

where $J(q) = dif(partial p, partial q) in RR^(3 times 3)$ denotes the geometric Jacobian and $lambda$ represents the damping factor.

To prevent overshooting or divergence, the joint update is clamped to a maximum norm relative to the step size $alpha in [0,1]$:
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

The MPC module is given the desired joint angles $q_"des" in RR^3$ from the IK module and computes optimal control torques $tau_"MPC"$ with a specified prediction horizon. We simplify the system dynamics for the MPC formulation by assuming unit inertia and neglecting Coriolis effects, justified by MuJoCo's compensation of complex dynamics. This yields a double-integrator model where the control input $tau_"MPC"$ effectively commands joint acceleration:

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

To enable behavior cloning from the expert IK–MPC controller, we generate a dataset of joint angles, joint velocities, target positions, and predicted torques from a closed-loop MuJoCo simulation. The process for each episode is:

== Collection process

1. Target sampling: A reachable end-effector target $p_"des" in RR^3$ is sampled within the workspace $cal(W)$ using cylindrical coordinates to sample a radius $r$ and height $z$ uniformly within the workspace bounds.
2. The IK solver computes the corresponding joint-space reference $q_"des"$.
3. Given the current state $x_k$ and a specified prediction horizon $N$, the MPC controller generates torque commands $tau_"MPC"$ to achieve the desired joint angles and velocities.
4. For each time step $k$, we record the current state $[q_1, q_2, q_3, dot(q)_1, dot(q)_2, dot(q)_3] in RR^6$, the target $p_"des" in RR^3$, and the predicted torque $tau_"MPC" in RR^3$.
5. We step the simulation until the end of the episode or until the target is reached using $tau = tau_"MPC" + tau_"env"$ (where $tau_"env"$ represents the forces computed by MuJoCo to compensate for Coriolis, centrifugal, and gravitational effects, ensuring the simplified dynamics model in the MPC controller remains valid; MuJoCo bias force: `mjData.qfrc_bias`).

During this process, if either the MPC controller or the IK solver fails to converge, we discard the data for that time step to retain only high-quality samples.

== Dataset Structure

After generation, the data is stored in HDF5 format, grouped by episodes containing synchronized sequences of states ($6D$), targets ($3D$), and actions ($3D$).

This hierarchical format preserves the temporal integrity of each trial, allowing us to process the data differently depending on the model architecture. The raw data is loaded via a custom MPCDataset class, which constructs the input feature vector $x$ by concatenating the state ($RR^6$) and the target ($RR^3$), resulting in a 9-dimensional input vector.

#figure(
  image("figures/random_episode.png", width: 70%),
  caption: [Visualization of a random episode],
)<fig:random_episode>

Depending on the learning algorithm, the data is processed differently according to the model architecture.

=== Flat Formatting

For non-sequential algorithms (e.g., MLPs, Random Forests), temporal dependencies are discarded to maximize sample efficiency. We treat every timestep $t$ from every episode as an independent sample (i.i.d).

$
  X_"flat" in RR^(N times 9), quad Y_"flat" in RR^(N times 3)
$

Where $N = sum_(i=0)^(E)T_i$ is the total number of timesteps across all episodes.

#pagebreak()

=== Sequential Formatting

For time-series algorithms such as GRU, preserving the temporal dependencies is crucial. We treat every episode as a sequence of timesteps, where each timestep is a sample.

$
  X_"seq" in RR^(E times T times 9), quad Y_"seq" in RR^(E times T times 3)
$

Where $E$ is the number of episodes and $T$ is the number of timesteps per episode.

=== Sliding Window Formatting

To approximate a recurrent structure with our MLP architecture, we augment each time step with its previous $W$ timesteps. This injects short-term memory into an otherwise i.i.d. formulation.

For each timestep $t$, we create a new input vector $X_"seq"_{t}$ by concatenating the current state $X_{t}$ with the previous $W$ states $X_{t-1}, X_{t-2}, ..., X_{t-W}$, and the target $in R^3$ only once to avoid redundancy.

$
  X_"seq" in RR^(N times (W dot 6 + 3)), quad Y_"seq" in RR^(N)
$

== Data Preprocessing

Our goal is to develop a robust and reliable controller that can handle uncertainties and disturbances in the system. To this end, we introduce small Gaussian noise to both the input state $[q_1, q_2, q_3, dot(q)_1, dot(q)_2, dot(q)_3]$ and the output action $tau_"MPC"$. This noise simulates real-world conditions such as sensor noise, actuator noise, and environmental disturbances. Because the data generation pipeline can produce an arbitrary number of samples, we collect a large dataset for training and increase the number of samples until the validation loss plateaus or computational limits are reached.

We ran a simple experiment with both SVR and MLP models from `Scikit-learn` and compared their performance while increasing the number of samples progressively.

#figure(
  image("figures/dataset_scale_plot.png", width: 80%),
  caption: [MSE vs number of samples (5 trials per model)],
)<fig:dataset_scale_plot>

Our experiment showed a clear plateau after 30000 samples @fig:dataset_scale_plot, therefore we decided to use only 35000 samples for the training of our regression algorithms. However, we also observed that RNN models such as GRU were taking full benefits of the full dataset by improving significantly compared to MLP models even after 30000 samples.

Nevertheless, since our focus is on finding a suitable architecture for a neural surrogate, we continue to utilize the full dataset for training all of our models for a fair comparison.

= Neural Network Architecture

We formulated the learning problem as a regression task, where the goal is to predict the torque $tau_"MPC"$ given the current state $x_k$ and the target $p_"des"$. We aimed to minimize the error between the neural network policy $pi_theta$ and the expert MPC actions:

$
  min_theta quad L(pi_theta(X), tau_"MPC")
$

where $L$ is a loss function that measures the difference between the predicted torque and the expert MPC torque. We investigated a range of models, from traditional machine learning to deep learning architecture, to understand their effectiveness in approximating the MPC policy. For this task, we compared the performance with 2 different loss functions:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

== Regression Baselines

To establish performance benchmarks, we evaluated several standard regression algorithms from the Scikit-learn library. These models operate on the "flat" dataset format, treating each timestep as an independent sample. We included tree-based regressors (Random Forest and Gradient Boosting) and a shallow MLP regressor as a lightweight baseline to compare against our deeper custom architectures.

== Custom Multi-Layer Perceptron (MLP)

We implemented a custom feedforward network to explore the impact of model capacity on cloning accuracy. This model processes the flat input vector through a series of fully connected linear layers with ReLU activations. We conducted an architectural search by varying:
  - Depth: Number of hidden layers
  - Width: Number of neurons per layer
This memory-less architecture captures and learns directly the mapping from the current state and target to the required control action.

== Time Series Models

To investigate whether historical context improves control fidelity, we evaluated architectures designed to capture temporal dependencies.

=== Sliding Window MLP

We implemented Sliding Window variants with the same depth and width parameters as our custom MLP. These variants receive a concatenated history of the past W state observations as input, where we set $W=5$. This explicitly injects short-term memory into the network to test if providing temporal history can improve predictions.

=== Recurrent Neural Networks (RNNs)

We evaluated Gated Recurrent Units (GRU) networks. These models maintain a hidden state $h_t$​ that summarizes the history of the episode up to time $t-1$. The final hidden state is then passed through a linear output layer to produce the predicted torque $pi_theta(X)$. Here again, we varied the number of hidden units per layer and the number of layers.

$
  h_t = "RNN"(x_t, h_(t-1)), quad pi_theta(X) = "Linear"(h_t)
$

= Evaluation Methodology

We evaluated the learned policies using a combination of offline and online metrics:

== Offline Metrics
We used MAE and Root Mean Squared Error (RMSE) to measure the average deviation of the predicted torques from the expert torques. Additionally, we defined Direction Accuracy (DA) to evaluate whether the model correctly identifies the sign (direction) of joint acceleration.

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

For our regression baseline with scikit-learn, we evaluated several standard regression algorithms using the collected offline dataset. As discussed before, we used only a subset of the whole dataset for a total of $35000$ samples, split into $80%$ for training and $20%$ for testing. The input feature space $X in RR^9$ consists of the robot's current state and target coordinates, while the output target $Y in RR^3$ corresponds to the applied joint actions (torques). The data was partitioned into a training set ($80%$) and a test set ($20%$) via random shuffling. To ensure the statistical significance of the reported metrics, each model was trained and evaluated over 5 independent runs. @table:regression_baseline summarizes the performance across MSE, MAE, Explained Variance, and Directional Accuracy. We also tried scaling our features using Scikit-Learn's StandardScaler, which did not significantly improve performance.

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

The results highlight the inherent non-linearity of the inverse dynamics mapping. Linear Regression failed to capture the underlying relationship, exhibiting high variance across all torque dimensions. In contrast, non-linear methods performed significantly better. The MLP Regressor achieved the lowest overall MSE ($0.053$), indicating its superior capability in minimizing large control deviations, which is critical for preventing hardware damage. While KNN Regressor achieved the highest Directional Accuracy ($99.87%$) and lowest MAE, its higher MSE suggests it suffers from occasional large prediction errors (outliers). Finally, different SVM models with linear and radial basis function (RBF) kernels were evaluated. However, as mentioned in the documentation, RBF kernels cannot scale with that many samples, whereas linear kernels yielded poor performance.

#figure(
  image("figures/mse_per_torque_without_linear.png", width: 100%),
  caption: [Mean Squared Error per Torque],
)<fig:mse_per_torque>

== Loss comparison

In this section, we compare performance of training models with two different loss functions. For this part we used a PyTorch implementation of the MLP regressor and of a GRU RNNs. @table:loss_comparison summarizes the results of our experiments where the name of the model configuration is the name of the model and the loss function used for training.

#let model_col(name) = strong(name)
#figure(
  table(
    columns: (auto, auto, auto),
    inset: 5pt,
    align: (col, row) => (if col == 0 { left } else { center + horizon }),
    stroke: (x, y) => (
      top: if y == 0 { 1pt } else if y == 1 { 0.5pt } else { 0pt },
      bottom: 1pt,
    ),
    table.header(
      [*Model Config*],
      [*MSE (Mean $plus.minus$ Std)*],
      [*MAE (Mean $plus.minus$ Std)*]
    ),

    model_col("MLP_mse"), $0.1460 plus.minus 0.0522$, $0.1474 plus.minus 0.0126$,
    model_col("MLP_mae"), $0.2017 plus.minus 0.0455$, $0.0892 plus.minus 0.0078$,
    model_col("GRU_mse"), $1.1442 plus.minus 0.1441$, $0.2581 plus.minus 0.0138$,
    model_col("GRU_mae"), $2.1821 plus.minus 0.1001$, $0.3570 plus.minus 0.0101$,
  ),
  caption :[Loss Comparison]
)<table:loss_comparison>

Following this experiment, we adopted the MSE loss function for our experiments because models trained with it consistently achieved lower test error and lower variance across runs, as shown in @table:loss_comparison. In particular, the MLP_mse and GRU_mse configurations outperformed their MAE-trained counterparts in terms of MSE, which was our primary performance metric for policy imitation.

== Hyperparameters Tuning of MLP and GRU

In this section, we present the offline evaluation metrics collected during the training of MLP and GRU models to determine the optimal architecture for our task.

=== Experimental Setup
To ensure the reliability of our results, each model configuration was trained and evaluated 5 times. The dataset was split into training (80%), validation (10%), and testing (10%) sets. The results presented in @fig:architecture_comparison reflect the MSE on the test set.

We also monitored validation loss to detect potential overfitting. No significant overfitting was observed in any of the experiments across the tested architectures and data augmentation strategies. As established in previous sections, the MLP models were trained on a subset of 35,000 timesteps, which was deemed sufficient for convergence, while the GRU models utilized the full dataset to capture temporal dependencies effectively.

=== Tested Architectures
We evaluated four variations of the Multi-Layer Perceptron (MLP) and four variations of the Gated Recurrent Unit (GRU) network. The specific hyperparameters for each configuration are detailed in @tab:model_configs.

#figure(
  table(
    columns: (auto, auto, auto),
      inset: 5pt,
      align: (col, row) => (if col == 0 { left } else { center + horizon }),
      stroke: (x, y) => (
        top: if y == 0 { 1pt } else if y == 1 { 0.5pt } else { 0pt },
        bottom: 1pt,
      ),
    table.header(
      [*Model Name*], [*Type*], [*Parameters*]
    ),
    [MLP_Small],        [MLP], [Hidden Layers: [64, 32]],
    [MLP_Medium],       [MLP], [Hidden Layers: [128, 64]],
    [MLP_Deep],         [MLP], [Hidden Layers: [256, 128, 64, 32]],
    [MLP_Deep_Scaled],  [MLP], [Hidden Layers: [512, 256, 128, 64]],
    [GRU_Shallow],      [GRU], [Hidden Dim: 64, Layers: 1],
    [GRU_Medium],       [GRU], [Hidden Dim: 128, Layers: 2],
    [GRU_Deep],         [GRU], [Hidden Dim: 128, Layers: 4],
    [GRU_Wide],         [GRU], [Hidden Dim: 256, Layers: 2],
  ),
  caption: [Summary of model architectures and hyperparameters used during tuning.]
) <tab:model_configs>

To approximate temporal awareness, we also implemented Sliding Window variants (W=5) across the four MLP architectures, using the same hidden layer parameters.

=== Results Analysis
#figure(
  image("figures/grouped_complexity_comparison.png", width: 100%),
  caption :[Architecture Comparison (Metric: MSE). Error bars represent the standard deviation across 5 runs. Note: GRU_Wide is plot under Deep+.]
) <fig:architecture_comparison>

The performance comparison in @fig:architecture_comparison highlights distinct trends between the MLP and GRU architectures:

+ *MLP Superiority:* In our experimental setting, the MLP architectures consistently outperformed the GRU variants. The MLP_Deep configuration achieved the lowest Mean MSE overall (approximately 0.05).
+ *Depth vs. Width:* For the MLP, increasing network depth provided significant performance gains, with MLP_Deep notably outperforming MLP_Medium and MLP_Small. Increasing width did not improve performance as MLP_Deep_Scaled was unable to achieve a significant reduction in MSE over MLP_Deep. This suggests that for this task, increasing the depth of the network is more beneficial for performance.
+ *Impact of Sliding Window:* Contrary to the expectation that history would aid prediction, the inclusion of a sliding window did not result in a reduction in MSE. In fact, the sliding window resulted in a slight increase in MSE for medium and deep models. This counter-intuitive result suggests that for this specific task, the current state contains sufficient information to determine the optimal control action (Markov property). The sliding window likely complicated the optimization landscape, introducing redundant noise to the model.
+ *GRU Performance:* The GRU models struggled to match the precision of the MLPs. GRU_Shallow performed the worst among all tested configurations (MSE $approx$ 2.7). Increasing complexity helped; GRU_Deep and GRU_Wide achieved significant improvements upon the shallower variants, though still lagging behind the MLP models.

Based on these results, MLP_Deep demonstrates the strongest predictive capability and stability on the test set.

== Online evaluation (MuJoCo)

To validate these offline findings, we conducted a comprehensive online evaluation. We deployed every trained controller into the closed-loop MuJoCo simulation. These were benchmarked against the original MPC policy (Expert). We executed 1000 random test episodes, with a limit of 150 steps per episode, for each model. Performance was measured using *Success Rate*, defined as the percentage of episodes where the end-effector converges within a Euclidean distance $epsilon$ of the target. To analyze precision, three tiers were established: Strict ($epsilon = 0.02m$), Moderate ($epsilon = 0.03m$), and Relaxed ($epsilon = 0.05m$). We also tracked *Inference Latency* to quantify the speedup relative to the MPC solver.

=== Success Rate Results
While Scikit-learn baselines failed to control the robot ($<35%$ success rate), the PyTorch-based architectures produced results comparable to the MPC.

#figure(
  image("figures/closed_loop_success_thresholds.png", width: 100%),
  caption :[Comparison of closed-loop success rates across different error tolerances ($epsilon$). The bar graphs represent the mean values across 5 runs.]
) <fig:success_thresholds>

@fig:success_thresholds illustrates the performance of the top-performing models. The results highlight a precision gap in behavior cloning. MLP_Deep was able to achieve 84.98% success rate under relaxed tolerances, but only 64.16% under strict tolerance. However, the mean final tracking error was only 2.9 cm, confirming that the majority of failures were near-misses. This evaluation also confirms our results from earlier, whereby static MLP outperformed temporal architectures, validating that temporal history is unnecessary for this task.

=== Computational Efficiency
We compared the distribution of solve times between the MPC and MLP_Deep.

#figure(
  image("figures/mpc_mlp_solvetime_boxplot.png", width: 65%),
  caption :[Distribution of Inference Latency per Control Step.]
) <fig:solvetime_boxplot>

The MLP_Deep policy achieved a mean solve time of 1.102$plus.minus$0.614ms. This is nearly one-third of the mean solve time of the MPC Expert. Notably, the neural network offers a much more deterministic inference time as seen in @fig:solvetime_boxplot, making it more suitable for hard real-time constraints than the MPC, whose solve time fluctuates significantly based on the optimization landscape. Additionally, MLP_Deep utilized significantly fewer system resources, with a 25% decrease in CPU utilization as compared to the MPC.

== Discussion and Conclusion

Our results show a strong alignment between offline regression metrics and closed-loop control performance. The low offline MSE $approx$ 0.5 successfully translated to a robust online policy, achieving a 84.98% success rate under relaxed tolerances. The mean final tracking error of 2.9cm is not a failure of generalization, but rather a direct reflection of the resolution limit inherent in the offline data. The neural network successfully learned the expert's global trajectory, but lacks the gradient-based feedback required to eliminate the final centimeter of error.

Crucially, the failure of Sliding Window and GRU models to outperform static MLPs confirms that the dynamics for our specific setup are strictly Markovian and are fully captured by the current state ($q, dot(q)$). Regarding model capacity, while network depth proved critical for capturing the non-linear inverse dynamics, increasing width did not achieve significant performance gains. This suggests that for this specific task, adding more layers is far more effective than adding more neurons.

In conclusion, a standard Feedforward Neural Network is able to approximate an MPC policy with approximately 1ms inference latency and 2.9cm average error. MLP_Deep emerged as the optimal architecture, achieving significant improvements in speed and resource utilization, confirming that the learned policy is sufficiently lightweight for deployment on resource-constrained embedded systems.

= Future Work

This work establishes a foundation for behavior cloning of MPC on 3-DOF manipulators, which can be extended in several directions. Firstly, to bridge the precision gap under strict tolerances, future work should incorporate Data Aggregation (DAgger), allowing the learner to query the expert and correct the terminal steady-state error.

Secondly, future work should investigate this approach on robotic manipulators with higher degrees of freedom (e.g., 6-DOF). We hypothesize that increasing network depth and dataset size will become critical when state and action space dimensionality increases significantly.

Additionally, to advance towards real-world deployment, the methodology should be extended to handle more complex control scenarios. It could be interesting to investigate the cloning of a non-linear MPC which is capable of handling more complex dynamics.

From a methodological perspective, exploring advanced neural network architectures represents a promising direction. Transformer models, with their self-attention mechanisms, could be investigated for their ability to capture complex, long-range dependencies. Furthermore, the Legendre Memory Unit (LMU) @NEURIPS2019_952285b9, developed at the University of Waterloo, offers a complementary, principled approach to continuous time memory, which may prove to be well-suited for the robotic system's underlying dynamics. Inverse reinforcement learning @deAPorto2025 may prove to be an efficient alternative to learn the underlying MPC cost function.

= Acknowledgments

This project is a final project for the course "Foundations of Artificial Intelligence" - SYDE522. We would like to thank our instructor Terry Stewart for his guidance and support.
