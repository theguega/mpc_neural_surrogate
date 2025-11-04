import casadi as ca
import numpy as np


class MPCController:
    def __init__(self, dt=0.05, prediction_horizon=20):
        self.dt = dt
        self.N = prediction_horizon

        # define symbolic variables
        self.q = ca.SX.sym("q", 3)  # joint positions
        self.q_dot = ca.SX.sym("q_dot", 3)  # joint velocities
        self.tau = ca.SX.sym("tau", 3)  # control inputs (torques)

        # Simplified dynamics model: assumes unit mass and no gravity
        # Gravity compensation should be added externally when applying torques
        q_ddot = self.tau

        # state and input vectors
        x = ca.vertcat(self.q, self.q_dot)
        u = self.tau

        # discrete dynamics: x_next = x + dt * f(x,u)
        x_next = x + self.dt * ca.vertcat(self.q_dot, q_ddot)
        self.f = ca.Function("f", [x, u], [x_next])  # discrete dynamics function

        # optimization problem
        self.opti = ca.Opti()
        self.x = self.opti.variable(6, self.N + 1)  # states over horizon
        self.u = self.opti.variable(3, self.N)  # inputs over horizon
        self.p = self.opti.parameter(6)  # initial state
        self.x_ref = self.opti.parameter(3)  # target position

        cost = 0

        # cost matrices - increased position weights to prioritize reaching target
        Q = np.diag([200, 200, 200, 5, 5, 5])  # state error weight (position >> velocity)
        R = np.diag([0.1, 0.1, 0.1])  # input weight (lower to allow larger torques)
        Q_N = np.diag([1000, 1000, 1000, 50, 50, 50])  # terminal cost (high position penalty)

        # stage cost
        for k in range(self.N):
            target_state_k = ca.vertcat(self.x_ref, ca.DM.zeros(3, 1))
            cost += ca.mtimes([(self.x[:, k] - target_state_k).T, Q, (self.x[:, k] - target_state_k)])
            cost += ca.mtimes([self.u[:, k].T, R, self.u[:, k]])

        # terminal cost
        target_state_N = ca.vertcat(self.x_ref, ca.DM.zeros(3, 1))
        cost += ca.mtimes(
            [
                (self.x[:, self.N] - target_state_N).T,
                Q_N,
                (self.x[:, self.N] - target_state_N),
            ]
        )

        self.opti.minimize(cost)  # set cost function

        # dynamics constraints
        for k in range(self.N):
            self.opti.subject_to(self.x[:, k + 1] == self.f(self.x[:, k], self.u[:, k]))
        self.opti.subject_to(self.x[:, 0] == self.p)  # initial state constraint

        # Add torque limits to prevent unrealistic commands
        tau_max = 50.0  # Maximum torque per joint
        for k in range(self.N):
            self.opti.subject_to(self.u[:, k] <= tau_max)
            self.opti.subject_to(self.u[:, k] >= -tau_max)

        # solver options
        solver_opts = {
            "ipopt.print_level": 0,  # suppress ipopt output
            "print_time": 0,
            "ipopt.max_iter": 100,
        }
        self.opti.solver("ipopt", solver_opts)  # choose solver

    def solve(self, x0, x_ref_val):
        self.opti.set_value(self.p, x0)  # set current state
        self.opti.set_value(self.x_ref, x_ref_val)  # set target

        try:
            sol = self.opti.solve()  # solve MPC
            return sol.value(self.u[:, 0]), True  # return first control
        except RuntimeError as e:
            print(f"MPC solver failed: {e}")
            return np.zeros(3), False  # return zero if fails
