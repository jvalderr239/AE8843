import numpy as np
from scipy import integrate
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt
from matplotlib import animation


class MPC:
    # state indices
    xidx, yidx, thetaidx = 0, 1, 2
    # control max vals
    v_max = 0.6
    v_min = -v_max
    omega_max = np.pi/4
    omega_min = -omega_max
    wheel_base = .7
    t0, dT, tf = 0, .5,  5
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
                     [0, 1]], dtype=np.float)
        self.Q = Q
        self.R = R

    def compute_state(self):
        """
        This method propagates the state of the system by the horizon
        :return: Null
        """
        temp = self.prediction_state[:self.num_states]
        self.state[:, [0]] = temp

        for idx in range(1, self.horizon):

            #print("----------------------------------")
            #print("Current initial state: \n")
            #print(temp, idx)
            #print("----------------------------------")
            self.state[:, [idx]] = self.propagate(temp, idx)
            temp = self.state[:, [idx]]

        "Plot troubleshoot"
        color = 'red'
        plt.plot(self.state[self.xidx,:], self.state[self.yidx, :],
                 color=color, label='Vehicle')
        plt.xlim([-5, 5])
        plt.ylim([-10, 10])
        plt.grid(axis='both', color='0.95')
        plt.show()

    def propagate(self, vector, idx):
        """

        :param vector: the current state of the system
        :param idx: the current horizon at which the dynamics are estimated
        :return: the next state of the system and an update to control matrix
        """
        theta = vector[-1]

        b = np.array([[np.cos(theta), 0],
                     [np.sin(theta), 0],
                     [0, 1]], dtype=np.float)
        P = solve_continuous_are(self.A, b, self.Q, self.R)
        K = np.linalg.inv(self.R).dot(np.dot(b.T, P))

        sol = integrate.solve_ivp(
            self.sys_func, self.time, vector.T[0], args=(self.A, b, K), method='RK45', t_eval=self.time)

        # Optimal Trajectory and Control
        optimal_trajectory = sol.y
        print(optimal_trajectory)
        optimal_control = np.matmul(-K, optimal_trajectory)
        # Only take the first action
        self.controls[:, [idx]] = optimal_control[:, [1]]
        # Apply control to current state

        return self.A.dot(vector) + self.dT*b.dot(self.controls[:, [idx]])

    def compute_objective_func(self):
        obj = 0
        for idx in range(self.horizon):
            state_error = self.state[:, [idx]] - self.prediction_state.T[:, self.num_states:]
            control = self.controls[:, idx]
            print("----STATE COST----")
            xQx = self.quadratic_cost(state_error, self.Q)
            print(state_error.T*self.Q*state_error)
            print("----CONTROL COST----")
            uRu = self.quadratic_cost(control, self.R)
            obj += xQx + uRu
        return obj

    @staticmethod
    def quadratic_cost(vector, cost): return np.matmul(np.matmul(vector.T, cost), vector)

    def compute_constraints(self):
        pass

    def sys_func(self, t, x, a, b, k):
        """

        :param t: Current time at which the dynamics are evaluated
        :param x: Current state of the system
        :param a: the system matrix
        :param b: the control matrix
        :param k: the gain matrix
        :return: state space evaluation
        """
        #print("------current solution--------")
        #print("X: ", "\t", str(x))
        #print("A", "\t", a, a.shape, "\t", t)
        #print("B", "\t", b, b.shape, "\t", t)
        #print("K", "\t", k, k.shape, "\t", t)
        #print("----------------------------------")
        return np.matmul(a - b.dot(k), x.reshape(self.num_states, 1)).T

    def plot_state_and_estimate(self):
        pass


class Plotter:
    # Plot attributes
    xlim, ylim = 5, 10
    fig = plt.figure()
    ax = plt.axes(xlim=(-xlim, xlim), ylim=(-ylim, ylim))
    line = ax.plot([], [], lw=2)

    # animation props
    frames = 100

    def __init__(self, dT):
        self.dT = dT

    def init(self):
        self.line.set_data([], [])
        return self.line

    def animate(self, i):

        self.line.set_data()
        return self.line

    def run(self):
        anim = animation.FuncAnimation(fig=self.fig, func=self.animate, init_func= self.init,
                                       frames= self.frames, interval=self.dT, blit=True)
        anim.save('basic.mp4', fps=30)
        plt.show()


if __name__ == '__main__':

    # Define gains
    kx = 15
    ky = 10
    ktheta = 1
    kv = 100
    komega = 200

    # Choose horizon
    horizon = 10
    start = np.array([[0], [0], [np.pi/10]], dtype=np.float)
    goal = np.array([[2], [2], [0]], dtype=np.float)
    start_control = np.full((2, 1), 0, dtype=np.float)

    Q = np.array([[kx, 0, 0],
                  [0, ky, 0],
                  [0, 0, ktheta]], dtype=np.float)
    R = np.array([[kv, 0], [0, komega]], dtype=np.float)

    counter = 0
    solution = []

    # Start MPC

    dummy = MPC(start, goal, start_control, horizon, Q, R)
    while dummy.compute_objective_func() > .5:

        dummy.compute_state()

