import numpy as np
from scipy import integrate
from scipy import linalg as LA
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
    time_elapsed = 0

    def __init__(self, initial_state, goal_state, initial_control, N, Q, R):

        self.horizon = N
        self.num_states = initial_state.size
        self.control_input = initial_control
        self.num_controls = self.control_input.size
        self.state = np.zeros((self.num_states, self.horizon+1))
        self.goal = goal_state
        self.controls = np.zeros((self.num_controls, self.horizon))
        self.controls[:, [0]] = initial_control
        self.prediction_state = self.update_initial(initial_state)

        self.A = np.array([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]], dtype=np.float)
        self.B = np.array([[np.cos(initial_state[self.thetaidx]), 0.1],
                     [np.sin(initial_state[self.thetaidx]), 0],
                     [0, .1]], dtype=np.float)
        self.Q = Q
        self.R = R
        self.time_elapsed = 0
        # Initialize animation
        self.plot = Plotter(self.prediction_state[self.xidx],
                            self.prediction_state[self.yidx],
                            self.dT)

    def update_initial(self, new_initial):
        return np.concatenate([new_initial, self.goal])

    def update_time(self):
        self.time_elapsed += self.dT

    def get_time_elapsed(self):
        return self.time_elapsed

    def compute_state(self):
        """
        This method propagates the state of the system by the horizon
        :return: Null
        """
        temp = self.prediction_state[:self.num_states]
        self.state[:, [0]] = temp

        for idx in range(1, self.horizon):

            print("----------------------------------")
            print("Current initial state and horizon: \n")
            print(temp, "\t", idx)
            print("----------------------------------")
            self.state[:, [idx]] = self.propagate(temp, idx)
            temp = self.state[:, [idx]]
        self.update_time()

        self.plot_state_and_estimate(self.state, self.get_time_elapsed())
        """
        "Plot troubleshoot"
        color = 'red'
        plt.plot(self.state[self.xidx,:], self.state[self.yidx, :],
                 color=color, label='Vehicle')
        plt.xlim([-5, 5])
        plt.ylim([-10, 10])
        plt.grid(axis='both', color='0.95')
        plt.show()
        """

        return self.state

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
        P = LA.solve_continuous_are(self.A, b, self.Q, self.R)
        K = np.linalg.inv(self.R).dot(np.dot(b.T, P))
        eigVals, eigVecs = LA.eig(self.A-np.matmul(b, K))

        # Project the solver
        span = np.linspace(self.t0 + self.get_time_elapsed(), self.tf-self.get_time_elapsed(), int(1 / self.dT))
        sol = integrate.solve_ivp(
            fun=self.sys_func, t_span=span,
            y0=vector.T[0], args=(self.A, b, K), method='RK45', t_eval=span
            )

        # Optimal Trajectory and Control
        print('----OPT TRAJ---')
        print(sol.y)
        optimal_trajectory = sol.y
        #print(optimal_trajectory)
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
            xQx = self.quadratic_cost(state_error, self.Q)
            uRu = self.quadratic_cost(control, self.R)
            obj += xQx + uRu
        print("---OBJECTIVE COST---")
        print(LA.norm(obj))
        print("----------")
        return LA.norm(obj)

    @staticmethod
    def quadratic_cost(vector, cost): return np.matmul(np.matmul(vector.T, cost), vector)

    @staticmethod
    def dlqr(A, B, Q, R):
        """Solve the discrete time lqr controller.

        x[k+1] = A x[k] + B u[k]

        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """

        # ref Bertsekas, p.151

        # first, try to solve the ricatti equation
        P = LA.solve_discrete_are(A, B, Q, R)

        # compute the LQR gain
        K = LA.inv(B.T * P * B + R) * (B.T * P * A)

        eigVals, eigVecs = LA.eig(A - B * K)

        return K, P, eigVals

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

    def plot_state_and_estimate(self, state, idx):

        self.plot.set_state(state[self.xidx], state[self.yidx])
        self.plot.animate(idx)


class Plotter:
    # Plot attributes
    xlim, ylim = 10, 10
    fig = plt.figure()
    ax = plt.axes(xlim=(-xlim, xlim), ylim=(-ylim, ylim))
    line, = ax.plot([], [], lw=2, color='red')

    # animation props
    frames = 100

    def __init__(self, x, y, dT):
        self.dT = dT
        self.x = x
        self.y = y

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def set_state(self, x, y):
        self.x, self.y = x, y

    def get_state(self): return self.x, self.y

    def animate(self, i):
        x, y = self.get_state()
        self.line.set_data(x, y)
        return self.line,

    def run(self):
        anim = animation.FuncAnimation(fig=self.fig, func=self.animate, init_func= self.init,
                                       frames=self.frames, interval=self.dT, blit=True)
        anim.save('basic.mp4', fps=30)
        plt.show()


if __name__ == '__main__':


    # Define gains
    kx = 100
    ky = 100
    ktheta = 100
    kv = 10
    komega = 2

    # Choose horizon
    horizon = 5
    start = np.array([[2], [2], [np.pi/3]], dtype=np.float)
    goal = np.array([[8], [2], [0]], dtype=np.float)
    start_control = np.full((2, 1), .2, dtype=np.float)

    Q = np.array([[kx, 0.1, 0.1],
                  [0.1, ky, 0.1],
                  [0.1, 0.1, ktheta]], dtype=np.float)

    R = np.array([[kv, 0], [0, komega]], dtype=np.float)

    counter = 0
    solution = []

    # Start MPC

    dummy = MPC(start, goal, start_control, horizon, Q, R)
    while dummy.compute_objective_func() > .5 or dummy.get_time_elapsed() < dummy.tf:

        xo = dummy.compute_state()
        print("----UPDATING INITIAL STATE----")
        print(xo[:, [1]])
        dummy.update_initial(xo[:, [1]])


