import numpy as np
from control import matlab as mat
from scipy import integrate
from scipy import linalg

class MPC:
    x = 0
    y = 1
    theta = 2
    dT = .1
    v_max = 0.6
    v_min = -v_max
    omega_max = np.pi/4
    omega_min = -omega_max
    wheel_base = .7
    t0, tf = 0, 20
    time = np.linspace(t0, tf, int(1/dT))

    def __init__(self, start_state, goal_state, start_control, N, Q, R):

        self.horizon = N
        self.num_states = start_state.size
        self.control_input = start_control
        self.num_controls = self.control_input.size
        self.controls = np.zeros((self.num_controls, self.horizon))
        self.controls[:, [0]] = start_control
        self.prediction_state = np.concatenate([start_state, goal_state])
        self.state = np.zeros((self.num_states, self.horizon+1))
        self.goal = goal
        self.A = np.eye(self.num_states)
        self.B = np.array([[np.cos(start_state[-1]), 0],
                     [np.sin(start_state[-1]), 0],
                     [0, 1]])
        self.Q = Q
        self.R = R

    def compute_state(self):

        temp = self.prediction_state[:self.num_states]
        self.state[:, [0]] = temp

        for idx in range(1, self.horizon):
            print(temp, idx)

            self.state[:, [idx]] = self.propagate(temp, idx)
            temp = self.state[:, [idx]]

    def propagate(self, vector, idx):

        theta = vector[-1]
        b = np.array([[np.cos(theta), 0],
                     [np.sin(theta), 0],
                     [0, 1]])
        P = linalg.solve_continuous_are(self.A, b, self.Q, self.R)
        K = np.linalg.inv(self.R)*b.T*P
        self.controls[:, [idx]] = integrate.solve_ivp(self.sys_func, self.time, vector, args=(self.A, b, K))
        print(self.controls[:, [idx]])
        return self.A.dot(vector) + self.dT*b.dot(self.controls[:, [idx]])

    def compute_objective_func(self):
        obj = 0
        for idx in range(self.horizon):
            state_error = self.state[:, [idx]] - self.prediction_state[:, self.num_states:]
            control = self.controls[:, idx]
            obj += state_error.T*self.Q*state_error + control*self.R*control

    def compute_constraints(self):
        pass

    @staticmethod
    def sys_func(t, x, A, b, K): return (A - b*K)*x


if __name__ == '__main__':

    # Define gains
    kx = 1
    ky = 5
    ktheta = .2
    kv = .5
    komega = .05

    # Choose horizon
    horizon = 5
    start = np.array([[0], [0], [0]])
    goal = np.array([[2], [2], [0]])
    start_control = np.zeros((2, 1))

    Q = np.array([[kx, 0, 0],
                  [0, ky, 0],
                  [0, 0, ktheta]])
    R = np.array([[kv, 0], [0, komega]])

    counter = 0
    solution = []

    # Start MPC

    dummy = MPC(start, goal, start_control, horizon, Q, R)
    dummy.compute_state()

