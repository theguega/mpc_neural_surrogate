import casadi as ca
import numpy as np


class MPCController:
    def __init__(self, dt=0.05, prediction_horizon=20):
        self.dt = dt
        self.N = prediction_horizon

        self.q = ca.SX.sym("q", 3)
        self.q_dot = ca.SX.sym("q_dot", 3)
        self.tau = ca.SX.sym("tau", 3)

        # linear dynamics: x_next = A*x + B*u
        # x = [q, q_dot], u = tau
        x = ca.vertcat(self.q, self.q_dot)
        u = self.tau

        # explicit euler integration
        x_next = x + self.dt * ca.vertcat(self.q_dot, u)
        self.f = ca.Function("f", [x, u], [x_next])

        self.opti = ca.Opti("conic")
        self.x = self.opti.variable(6, self.N + 1)
        self.u = self.opti.variable(3, self.N)
        self.p = self.opti.parameter(6)
        self.x_ref = self.opti.parameter(3)

        cost = 0
        Q = np.diag([200, 200, 200, 20, 20, 20])
        R = np.diag([1.0, 1.0, 1.0])
        Q_N = np.diag([1000, 1000, 1000, 100, 100, 100])

        # QP matrices
        for k in range(self.N):
            target_state_k = ca.vertcat(self.x_ref, ca.DM.zeros(3, 1))
            err = self.x[:, k] - target_state_k
            cost += ca.mtimes([err.T, Q, err])
            cost += ca.mtimes([self.u[:, k].T, R, self.u[:, k]])

        target_state_N = ca.vertcat(self.x_ref, ca.DM.zeros(3, 1))
        err_N = self.x[:, self.N] - target_state_N
        cost += ca.mtimes([err_N.T, Q_N, err_N])

        self.opti.minimize(cost)

        for k in range(self.N):
            self.opti.subject_to(self.x[:, k + 1] == self.f(self.x[:, k], self.u[:, k]))
        self.opti.subject_to(self.x[:, 0] == self.p)

        tau_max = 50.0
        self.opti.subject_to(self.opti.bounded(-tau_max, self.u, tau_max))

        # OSQP solver
        opts = {"print_time": False, "osqp": {"verbose": False}}
        self.opti.solver("osqp", opts)

    def solve(self, x0, x_ref_val):
        self.opti.set_value(self.p, x0)
        self.opti.set_value(self.x_ref, x_ref_val)

        try:
            sol = self.opti.solve()
            return sol.value(self.u[:, 0]), True
        except Exception as e:
            print(f"MPC solver failed: {e}")
            # Try to get suboptimal solution if available
            try:
                return self.opti.debug.value(self.u[:, 0]), False
            except Exception as e:
                return np.zeros(3), False
