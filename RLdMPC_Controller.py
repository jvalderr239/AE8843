import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import integrate
from scipy import linalg as LA

# generate seed for random generator
random.seed(239)


class RLdMPC:

    # State indices
    xidx, yidx, thetaidx = 0, 1, 2
    xmax, ymax, thetamax = 20, 20, np.pi/2

    # Define parameter weights
    qx = np.sqrt(.4)
    qy = np.sqrt(.2)
    qtheta = np.sqrt(np.pi/10)
    qv = 1
    qomega = 1
    # Define time
    t0, dT, tf = 0, .5,  100
    num_pts = int(1/dT)
    time = np.linspace(t0, tf, num_pts)
    time_elapsed = 0
    mean, sq_sigma = 0, 1

    def __init__(self, vehicle, C, N):

        self.horizon = N
        self.vehicle = vehicle
        self.num_states = vehicle.state.size
        self.num_controls = vehicle.B.shape[1]
        self.y = np.zeros((self.num_states, self.horizon+1))
        self.controls = np.zeros((self.num_controls, 1))
        self.R = np.array([[self.qv, 0],
                           [0, self.qomega]], dtype=np.float)
        self.noise = np.random.normal(
            self.mean, self.sq_sigma, size=self.vehicle.state.shape)
        self.C = C
        self.C_est = np.zeros(self.C.shape)
        self.CCov = 1000*np.eye(self.num_states)
        self.G = 0
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
        u = []
        dev_state = []
        predictors = []
        uncertainty = []

        self.y[:, [0]] = np.matmul(self.C, self.vehicle.state) + self.noise
        Rt = np.linalg.inv(self.CCov)
        param_uncertainty = np.linalg.inv(Rt)
        uncertainty.append(param_uncertainty)
        for idx in range(1, self.horizon+1):

            # Step 2 of RLdMPC
            dstate, control = self.propagate(self.y[:, [idx-1]],
                                             uncertainty[idx-1])

            phi = np.matmul(
                self.vehicle.system_matrix(), self.y[:, [idx - 1]]) + dstate
            yhat = np.matmul(self.C_est, phi)
            # RLS method for 3rd order
            print(yhat)
            #Rt += np.matmul(phi, phi.T)
            param_uncertainty = np.linalg.inv(Rt)

            if not np.allclose(param_uncertainty, param_uncertainty.T, rtol=1e-05, atol=1e-08):
                print('Whoops')

            self.y[:, [idx]] = yhat  # converted from dev state

            # Store values
            u.append(control)
            dev_state.append(phi)
            predictors.append(yhat)
            uncertainty.append(param_uncertainty)

        print("----------------------------------")
        print("Current initial state and horizon: \n")
        print(self.y[self.xidx, :], "\n", self.y[self.yidx, :])
        print("----------------------------------")
        # Step 3 of RLdMPC

        new_phi = dev_state[0]
        new_output = np.matmul(self.C, new_phi)
        new_output_est = predictors[0]
        new_deviation = np.subtract(new_output, new_output_est)
        self.vehicle.update(new_phi, u[0])
        G = np.matmul(np.matmul(self.CCov, new_phi),
                      np.linalg.inv(
                          self.sq_sigma +
                          self.quadratic_cost(new_phi, self.CCov)))

        self.CCov -= np.matmul(np.matmul(G, new_phi.T), self.CCov)
        self.C_est += np.matmul(G, new_deviation.T)
        self.update_time()
        return self.predicted_state

    def propagate(self, vector, Q):
        """
        :param vector: the current deviation of the goal state
        :param idx: the current horizon at which the dynamics are estimated
        :return: the next state of the system and an update to control matrix
        """
        # add noise to the output
        print(self.C_est, vector)
        y = np.matmul(self.C_est, vector)
        initial_dev = np.subtract(y, self.vehicle.goal)
        theta = vector[self.thetaidx]

        deviation_from_goal = initial_dev.T[0]
        b = self.vehicle.control_matrix(theta)
        A = self.vehicle.system_matrix()
        print('-------Q-------')
        print(Q)
        print('---------------')
        P = LA.solve_continuous_are(A, b, Q, self.R)
        K = np.matmul(np.linalg.inv(self.R), (np.dot(b.T, P)))
        eigVals, eigVecs = LA.eig(A-np.matmul(b, K))

        # Project the solver
        span = np.linspace(
            self.t0,
            self.tf,
            self.num_pts
        )
        # Drive deviation to zero by solving xdot = (A-BK)x
        # Output the noisy signal
        sol = integrate.solve_ivp(
            fun=self.sys_func,
            t_span=span,
            y0=deviation_from_goal,
            args=(A, b, K),
            method='RK45',
            t_eval=span
            )
        # Optimal Trajectory and Control which are used for the Certainty Equivalence strategy
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
        :param param_est: output matrix
        :return: output estimation for each time, t, evaluated with sys dynamics
        """
        return np.matmul(a - b.dot(k), x.reshape(self.num_states, 1)).T

    @staticmethod
    def quadratic_cost(x, Q): return np.matmul(np.matmul(x.T, Q), x)

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

        self.plot_control_history()

    def plot_control_history(self):
        v_history = [item[self.xidx][0] for item in self.vehicle.control_history]
        omega_history = [item[self.yidx][0] for item in self.vehicle.control_history]
        plt.figure()
        tt = np.arange(self.t0, self.get_time_elapsed(), self.dT)
        plt.plot(tt, v_history, 'm--', label='Lateral Velocity')
        plt.plot(tt, omega_history, 'y--', label='Angular Velocity')
        plt.xlabel('Time(s)')
        plt.ylabel('u(t)')
        plt.legend()
        plt.xlim([0, self.get_time_elapsed()])
        plt.ylim([-10, 10])
        plt.grid(axis='both', color='0.95')
        plt.show()

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




