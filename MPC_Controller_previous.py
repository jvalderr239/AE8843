import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import integrate
from scipy import linalg as LA

from BicycleModel import BicycleModel


class MPC:

    # State indices
    xidx, yidx, thetaidx = 0, 1, 2
    # Define parameter weights
    qx = np.sqrt(.4)
    qy = np.sqrt(.2)
    qtheta = np.sqrt(np.pi/10)
    qv = .99
    qomega = 1.2
    # Define time
    t0, dT, tf = 0, .5,  50
    num_pts =int(1/dT)
    time = np.linspace(t0, tf, num_pts)
    time_elapsed = 0

    def __init__(self, vehicle, N):

        self.horizon = N
        self.vehicle = vehicle
        self.num_states = vehicle.state.size
        self.num_controls = vehicle.B.shape[1]
        self.predicted_state = np.zeros((self.num_states, self.horizon+1))
        self.controls = np.zeros((self.num_controls, 1))
        self.Q = np.array([[self.qx, 0, 0],
                  [0, self.qy, 0],
                  [0, 0, self.qtheta]], dtype=np.float)
        self.R = np.array([[self.qv, 0],
                           [0, self.qomega]], dtype=np.float)
        self.time_elapsed = 0
        # Initialize animation
        self.plot = Plotter(self.vehicle.state[self.xidx],
                            self.vehicle.state[self.yidx],
                            self.dT)

    def update_time(self):
        self.time_elapsed += self.dT

    def get_time_elapsed(self):
        return self.time_elapsed

    def compute_state(self):
        """
        This method propagates the state of the system by the horizon
        :return: Null
        """

        self.predicted_state[:, [0]] = self.vehicle.state
        u = []
        for idx in range(1, self.horizon+1):
            dstate, control = self.propagate(self.predicted_state[:, [idx-1]])

            self.predicted_state[:, [idx]] = np.matmul(self.vehicle.system_matrix(), self.predicted_state[:, [idx-1]])\
                                             + dstate
            u.append(control)
        self.update_time()
        print("----------------------------------")
        print("Current initial state and horizon: \n")
        print(self.predicted_state[self.xidx, :], "\n", self.predicted_state[self.yidx, :])
        print("----------------------------------")
        current_initial = self.predicted_state[:self.num_states, [1]]

        self.vehicle.update(current_initial, u[0])
        return self.predicted_state

    def troubleshoot(self):

        "Plot history of the vehicle"
        color = 'red'
        x_history = [item[self.xidx]for item in self.vehicle.state_history]
        y_history = [item[self.yidx]for item in self.vehicle.state_history]

        plt.plot(x_history,y_history, 'bx', label='Path Traveled')
        plt.plot(self.vehicle.goal[self.xidx], self.vehicle.goal[self.yidx], 'gx', label='Goal')
        plt.plot(self.predicted_state[self.xidx, :], self.predicted_state[self.yidx, :], 'r--',
                 color=color, label='Predicted Trajectory')
        plt.legend()
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.grid(axis='both', color='0.95')
        plt.show()

    def propagate(self, vector):
        """
        :param vector: the current deviation of the goal state
        :param idx: the current horizon at which the dynamics are estimated
        :return: the next state of the system and an update to control matrix
        """
        deviation_from_goal = vector.T[0]-self.vehicle.goal.T[0]
        theta = vector[-1]

        b = self.vehicle.control_matrix(theta)
        A = self.vehicle.system_matrix()
        P = LA.solve_discrete_are(A, b, self.Q, self.R)
        K = np.linalg.inv(self.R).dot(np.dot(b.T, P))
        eigVals, eigVecs = LA.eig(A-np.matmul(b, K))

        # Project the solver
        span = np.linspace(
            self.t0,
            self.tf,
            self.num_pts
        )
        # Drive deviation to zero by solving xdot = (A-BK)x
        sol = integrate.solve_ivp(
            fun=self.sys_func,
            t_span=span,
            y0=deviation_from_goal,
            args=(A, b, K),
            method='RK45',
            t_eval=span
            )
        # Optimal Trajectory and Control
        optimal_trajectory = sol.y
        optimal_control = np.matmul(-K, optimal_trajectory)
        # Only take the first action and limit the entries
        v_clipped = np.clip(optimal_control[0, [0]], self.vehicle.v_min, self.vehicle.v_max)
        omega_clipped = np.clip(optimal_control[1, [0]], self.vehicle.omega_min, self.vehicle.omega_max)

        # Apply control to current state
        control = np.array([v_clipped, omega_clipped], dtype=float)
        du = self.dT * np.matmul(b, control)
        return du, control

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
    max_tolerance = 1e-02

    # Choose horizon
    horizon = 5
    start = np.array([[-8], [-9], [np.pi/2]], dtype=np.float)
    goal = np.array([[8], [5], [np.pi/4]], dtype=np.float)

    # Initialize the vehicle
    test_vehicle = BicycleModel(start, goal)

    counter = 0

    # Start MPC

    rldmpc = MPC(test_vehicle, horizon)
    while rldmpc.get_time_elapsed() < rldmpc.tf:

        xo = rldmpc.compute_state()
        print("----UPDATING INITIAL STATE----")
        new_initial = xo[:, [1]]
        print(new_initial)
        rldmpc.troubleshoot()
        if LA.norm(new_initial-goal) <= max_tolerance:
            print("---REACHED MAX TOLERANCE---")
            print(max_tolerance)
            break
