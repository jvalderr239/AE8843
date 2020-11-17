import numpy as np
from scipy import integrate
from scipy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib import animation


class MPC:
    # state indices
    xidx, yidx, thetaidx = 0, 1, 2
    # control max vals
    v_max = 3
    v_min = -v_max
    omega_max = 5
    omega_min = -omega_max
    wheel_base = .7
    t0, dT, tf = 0, .5,  25
    num_pts =int(1/dT)
    time = np.linspace(t0, tf, num_pts)
    time_elapsed = 0
    near_zero = .001

    def __init__(self, initial_state, goal_state, initial_control, N, Q, R):

        self.horizon = N
        self.num_states = initial_state.size
        self.control_input = initial_control
        self.num_controls = self.control_input.size
        self.state = np.zeros((self.num_states, self.horizon+1))
        self.start = initial_state
        self.goal = goal_state
        self.controls = np.zeros((self.num_controls, self.horizon))
        self.controls[:, [0]] = initial_control
        self.prediction_state = []
        self.update_initial(initial_state)
        self.A = np.array([[1, self.near_zero, self.near_zero],
                           [self.near_zero, 1, self.near_zero],
                           [1*np.exp(-5), 1*np.exp(-5), 1]], dtype=np.float)
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

        self.previous_loc = self.start[:self.num_states-1]

    def update_initial(self, new_initial):
        self.prediction_state = np.concatenate([new_initial, self.goal])

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

        for idx in range(1, self.horizon+1):
            dstate = self.propagate(temp, idx-1)
            print(dstate)
            self.state[:, [idx]] = self.state[:, [idx-1]] + dstate
            temp = self.state[:, [idx]]
        self.update_time()
        print("----------------------------------")
        print("Current initial state and horizon: \n")
        print(self.state[self.xidx, :], "\t", self.state[self.yidx, :])
        print("----------------------------------")
        current_initial = self.state[:self.num_states - 1, [1]]
        self.previous_loc = np.concatenate((self.previous_loc, current_initial), axis=1)
        return self.state

    def troubleshoot(self):

        "Plot troubleshoot"
        color = 'red'
        plt.plot(self.previous_loc[self.xidx],self.previous_loc[self.yidx], 'bx', label='Path Traveled')
        plt.plot(self.goal[self.xidx], self.goal[self.yidx], 'gx', label='Goal')
        plt.plot(self.state[self.xidx, :], self.state[self.yidx, :], 'r--',
                 color=color, label='Predicted Trajectory')
        plt.legend()
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.grid(axis='both', color='0.95')
        plt.show()

    def propagate(self, vector, idx):
        """

        :param vector: the current deviation of the goal state
        :param idx: the current horizon at which the dynamics are estimated
        :return: the next state of the system and an update to control matrix
        """
        deviation_from_goal = vector.T[0]-self.goal.T[0]
        print("GOAL")
        print(deviation_from_goal[-1])

        theta = deviation_from_goal[-1]

        b = np.array([[np.cos(theta), 0],
                     [np.sin(theta), 0],
                     [0, 1]], dtype=np.float)
        P = LA.solve_discrete_are(self.A, b, self.Q, self.R)
        K = np.linalg.inv(self.R).dot(np.dot(b.T, P))
        eigVals, eigVecs = LA.eig(self.A-np.matmul(b, K))

        # Project the solver
        span = np.linspace(
            self.t0,
            self.tf ,
            self.num_pts
        )
        # Drive deviation to zero by solving xdot = (A-BK)x
        sol = integrate.solve_ivp(
            fun=self.sys_func,
            t_span=span,
            y0=deviation_from_goal,
            args=(self.A, b, K),
            method='RK45',
            t_eval=span
            )
        # Optimal Trajectory and Control
        optimal_trajectory = sol.y
        optimal_control = np.matmul(-K, optimal_trajectory)
        # Only take the first action
        v_clippled = np.clip(optimal_control[0, [0]], self.v_min, self.v_max)
        w_clipped = np.clip(optimal_control[1, [0]], self.omega_min, self.omega_max)
        self.controls[:, [idx]] = np.array([v_clippled, w_clipped], dtype=float)
        # Apply control to current state

        return self.dT*np.matmul(b, self.controls[:, [idx]])

    def compute_objective_func(self):
        #TODO: This function may not be needed but need to find a way to compute objective for RL
        obj = 0
        for idx in range(self.horizon):
            state_error = self.state[:, [idx]] - self.prediction_state.T[:, self.num_states:]
            control = self.controls[:, idx]
            xQx = self.quadratic_cost(state_error, self.Q)
            uRu = self.quadratic_cost(control, self.R)
            obj += .5*(xQx + uRu)
        print("---OBJECTIVE COST---")
        print(LA.norm(obj))
        print("----------")
        return LA.norm(obj)

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

    def plot_state_and_estimate(self, state, idx):

        self.plot.set_state(state[self.xidx], state[self.yidx])
        self.plot.animate(idx)
        self.plot.run()


class Plotter:
    # TODO: fix animation class
    # Plot attributes
    xlim, ylim = 20, 20
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
        #anim.save('basic.mp4', fps=30)
        plt.show()


if __name__ == '__main__':

    # Define tolerance
    max_tolerance = .5
    # Define gains
    kx = np.sqrt(.4)
    ky = np.sqrt(.2)
    ktheta = np.sqrt(np.pi/10)
    kv = .99
    komega = 1.2

    # Choose horizon
    horizon = 5
    start = np.array([[-8], [-9], [0]], dtype=np.float)
    goal = np.array([[8], [9], [0]], dtype=np.float)
    start_control = np.array([[.5], [0]], dtype=np.float)

    Q = np.array([[kx, 0, 0],
                  [0, ky, 0],
                  [0, 0, ktheta]], dtype=np.float)

    R = np.array([[kv, .001], [.001, komega]], dtype=np.float)

    counter = 0
    solution = []

    # Start MPC

    dummy = MPC(start, goal, start_control, horizon, Q, R)
    while dummy.get_time_elapsed() < dummy.tf:

        xo = dummy.compute_state()
        print("----UPDATING INITIAL STATE----")
        new_initial = xo[:, [1]]
        print(new_initial)
        dummy.update_initial(new_initial)
        if LA.norm(new_initial-goal) <= max_tolerance:
            print("---REACHED MAX TOLERANCE---")
            print(max_tolerance)
            break
    dummy.troubleshoot()

