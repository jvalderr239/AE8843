import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import integrate
from scipy import linalg as LA
import cvxpy as cp
# generate seed for random generator
random.seed(239)


class RLdMPC:

    # State indices
    xidx, yidx, thetaidx = 0, 1, 2
    vidx, omegaidx = 0, 1
    xmax, ymax, thetamax = 10, 10, 10

    # Define parameter weights
    qv = np.sqrt(1)
    qomega = np.sqrt(1)
    # Define time
    t0, tf = 0, 20
    num_pts = int(1/.01)
    time = np.linspace(t0, tf, num_pts+1)
    dT = time[1]-time[0]
    time_elapsed = 0
    mean, sq_sigma = 0, 1
    yhat_idx, u_idx, phi_idx, z_idx, r_idx = 0, 1, 2, 3, 4

    def __init__(self, vehicle, C_est, N, R=1):

        self.horizon = N
        self.vehicle = vehicle
        self.num_states = vehicle.state.size
        self.num_controls = vehicle.B.shape[1]
        self.R = R
        self.noise = np.random.normal(
            self.mean, self.sq_sigma, size=(self.num_states*self.num_pts, 1))
        self.C = self.vehicle.output_matrix()
        self.C_est = C_est
        self.CCov = 1000*np.eye(self.num_states)

        self.time_elapsed = 0
        self.y = [np.matmul(self.C, self.vehicle.state) + self.noise[0]]
        self.phi = [self.vehicle.state]
        self.P = [self.CCov]
        self.params = [self.C_est]
        self.controls = [np.zeros((self.num_controls, 1))]
        self.Kstar = np.zeros((self.num_states, self.num_states))

        # Initialize animation
        self.plot = Plotter(self.vehicle.state[self.xidx],
                            self.vehicle.state[self.yidx],
                            self.dT)

    def update_time(self):
        self.time_elapsed += 1

    def get_time_elapsed(self):
        return self.time_elapsed

    def make_vector(self, item):
        """
        :param item: item to repeat
        :return: return list of repeated items
        """
        return np.array([item]*self.horizon)

    def compute_state_with_opt_mu(self):
        """
        This method propagates the state of the system by the horizon
        :return: Null
        """
        # Initialization
        # Step 1
        obj = 0
        prior = 0
        #self.vehicle.arrived()
        for t in range(self.num_pts+1):

            # minimize
            A = self.vehicle.system_matrix()
            b = self.vehicle.control_matrix()
            phit = self.vehicle.state
            yt = self.y[t]
            print(yt)
            Q = self.CCov
            rt = np.linalg.pinv(Q)
            utk = []
            try:
                self.Kstar = LA.solve_continuous_are(A, b, Q, self.R)
            except ValueError or np.linalg.LinAlgError:
                print('Using previous P')
            for idx in range(self.horizon):

                utk.append(self.propagate(phit.T[0], A, b, Q, self.Kstar))
                phit = np.matmul(A*self.dT + np.eye(3), phit) \
                       + utk[idx]
                yt = np.matmul(self.C_est, phit)
                rt += np.matmul(phit, phit.T)
                Q = np.linalg.pinv(rt)

            #utk = self.opt(t, A, b)

            print('OPTIMAL')
            control = np.array(utk[0], dtype=float)
            print(control)


            # Step 3 of RLdMPC

            self.vehicle.update(control)
            new_phi = self.vehicle.state

            new_output = np.matmul(self.C, new_phi) + self.get_noise(t)

            new_deviation = np.subtract(new_output,
                                        np.matmul(self.C_est, new_phi))

            G = np.matmul(np.matmul(self.CCov, new_phi),
                          np.linalg.pinv(
                              np.matmul(self.C_est, self.C_est.T) +
                              self.quadratic_cost(new_phi, self.CCov)))

            self.CCov -= np.matmul(np.eye(self.num_states) - np.matmul(G, new_phi.T), self.CCov)

            self.C_est += np.matmul(G, new_deviation.T).T
            # Store new values
            self.controls.append(control)
            self.phi.append(new_phi)
            self.y.append(new_output)
            self.P.append(self.CCov)
            self.params.append(self.C_est)
            print("----------------------------------")
            print("Next initial state: \n")
            print(self.vehicle.state[self.xidx], "\n", self.vehicle.state[self.yidx])
            print("----------------------------------")
            obj += new_output*new_output + self.R*control*control + np.matmul(np.matmul(new_phi.T, self.CCov), new_phi)
            if np.linalg.norm(new_phi) <= 1e-3:
                break

            self.update_time()
        return obj

    def compute_state_with_rldmpc(self):
        def compute_state_with_opt_mu(self):
            """
            This method propagates the state of the system by the horizon
            :return: Null
            """
            # Initialization
            # Step 1
            obj = 0
            prior = 0
            # self.vehicle.arrived()
            for t in range(self.num_pts + 1):

                # minimize
                A = self.vehicle.system_matrix()
                b = self.vehicle.control_matrix()
                phit = self.vehicle.state
                yt = self.y[t]
                print(yt)
                Q = self.CCov
                rt = np.linalg.pinv(Q)
                try:
                    self.Kstar = LA.solve_continuous_are(A, b, Q, self.R)
                except ValueError or np.linalg.LinAlgError:
                    print('Using previous P')
                utk = self.opt(t, A, b)

                print('OPTIMAL')
                control = np.array([[utk[0]]], dtype=float)
                print(control)

                # Step 3 of RLdMPC

                self.vehicle.update(control)
                new_phi = self.vehicle.state

                new_output = np.matmul(self.C, new_phi) + self.get_noise(t)

                new_deviation = np.subtract(new_output,
                                            np.matmul(self.C_est, new_phi))

                G = np.matmul(np.matmul(self.CCov, new_phi),
                              np.linalg.pinv(
                                  np.matmul(self.C_est, self.C_est.T) +
                                  self.quadratic_cost(new_phi, self.CCov)))

                self.CCov -= np.matmul(np.eye(self.num_states) - np.matmul(G, new_phi.T), self.CCov)

                self.C_est += np.matmul(G, new_deviation.T).T
                # Store new values
                self.controls.append(control)
                self.phi.append(new_phi)
                self.y.append(new_output)
                self.P.append(self.CCov)
                self.params.append(self.C_est)
                print("----------------------------------")
                print("Next initial state: \n")
                print(self.vehicle.state[self.xidx], "\n", self.vehicle.state[self.yidx])
                print("----------------------------------")
                self.update_time()
    def opt(self, t, A, b):
        obj = 0
        print('MINIMIZING')
        phit = self.vehicle.state. \
            reshape(self.num_states, )
        yt = self.y[t]. \
            reshape(self.num_controls, )
        rt = np.linalg.pinv(self.CCov).flatten()
        zt = np.matmul(self.CCov, phit)

        # Form and solve control problem
        phitk = cp.Variable((self.num_states, self.horizon + 1))
        utk = cp.Variable((self.num_controls, self.horizon))
        ytk = cp.Variable((self.num_states, self.horizon))
        ztk = cp.Variable((self.num_states, self.horizon)
                          )
        rbartk = cp.Variable((self.num_states ** 2, self.horizon),
                             nonneg=True)
        Rtk = cp.Variable((self.num_states, self.num_states))

        # Parameters
        Amat = cp.Parameter((self.num_states, self.num_states))
        bmat = cp.Parameter((self.num_states, self.num_controls))
        Rmat = cp.Parameter()
        Cparam = cp.Parameter((self.num_controls, self.num_states))
        Kstar = cp.Parameter((self.num_states, self.num_states),
                             PSD=True)
        Eye = cp.Parameter((self.num_states, self.num_states),
                           PSD=True)

        # Initialize params
        Amat.value = A
        bmat.value = b
        Rmat.value = self.R
        Cparam.value = self.C_est
        Kstar.value = self.Kstar
        Eye.value = np.eye(self.num_states)
        constr = []
        stateshape = self.vehicle.state.shape
        controlshape = (self.num_controls, 1)
        rshape = self.CCov.shape
        rflatshape = (self.num_states ** 2,)
        zero = np.zeros(stateshape).reshape(self.num_states, )
        input = np.zeros(controlshape)
        alpha = 1

        for idx in range(self.horizon):
            print('---------------')
            print(idx)
            print('---------------')

            # Define objective function
            # Stage Cost
            obj += cp.multiply(self.R, cp.sum_squares(utk[:, idx])) \
            + cp.quad_form(phitk[:, idx], Eye)\
                   #+ cp.sum_squares(ytk[:, idx]) \
                # + cp.sum(phitk[:, idx].T @ ztk[:, idx])
            # + phiz
            print(obj)
            constr += [
                # Lower Input Bounds
                cp.norm(-utk[self.vidx, idx], 'inf') <= -self.vehicle.v_min,
                # Upper Input Bounds
                cp.norm(utk[self.vidx, idx], 'inf') <= self.vehicle.v_max,

                # Lower Input Bounds
                 #cp.norm(-ytk[self.xidx, idx], 'inf') <= -self.xmax,
                # Upper Input Bounds
                # cp.norm(ytk[self.xidx, idx], 'inf') <= self.xmax,

            ]
            constr += [

                # Simulate next value
                phitk[:, idx + 1] == Amat @ phitk[:, idx] + bmat @ utk[:, idx],
                #ytk[:, idx] == Cparam @ phitk[:, idx],

            ]

        # terminal cost
        obj += cp.quad_form(phitk[:, self.horizon], Kstar)
        constr += [
            phitk[:, 0] == phit,
            #ytk[:, 0] == yt,
            # rbartk[:, 0] == rt,
            # ztk[:, 0] == zt
            #utk[:, idx] == ut
        ]

        ocp = cp.Problem(cp.Minimize(obj), constr)
        ocp.solve(solver=cp.CVXOPT, verbose=True)

        return utk.value[0]

    def get_noise(self, t):
        idx = 2*t + self.num_controls
        return self.noise[idx:idx + self.num_controls]

    def objective(self, utk, t, A, b, P, Kstar):

        return self.calculate_finite_horizon(t, utk, A,b,P,Kstar)

    def calculate_finite_horizon(self, t, ut, A, b, P, Kstar):

        phi = self.vehicle.state
        utk = ut.reshape(2, 1)
        print('state')
        print(phi)
        r = np.linalg.inv(P)
        # Calculate finite horizon
        obj = 0
        for idx in range(self.horizon):
            print("Horizon: ", idx, "for time: ", t)

            if idx == 0:

                phi = np.matmul(A, phi) \
                      + np.matmul(self.vehicle.control_matrix(phi[self.thetaidx]), utk
                )
                y = self.y[t]
                z = np.matmul(self.P[t], phi)
                u = utk
                r = r

            else:
                control = self.propagate(
                    phi,
                    A, b,
                    np.linalg.inv(r),
                    self.Kstar
                )
                # update per horizon
                # phi
                phi = np.matmul(A, phi) \
                      + np.matmul(self.vehicle.control_matrix(phi[self.thetaidx]), control)
                # u
                u = control
                # y
                y = np.matmul(self.C_est, phi)
                # R
                r += np.matmul(phi.T, phi)
                ptk = np.linalg.inv(r.reshape(self.CCov.shape))
                z = np.matmul(ptk, phi)
            print(phi)
            print(y)
            print(u)
            print(z)

            obj += self.cost(phi, y, u, z, self.R)
        obj += self.quadratic_cost(phi, Kstar)

        return obj

    def cost(self, phi, y, u, z, R):

        y_squared = np.matmul(y.T,y)
        u_squared = self.quadratic_cost(u, R)
        phiz = np.matmul(phi.T, z)
        cost = np.sum([y_squared, u_squared, phiz ])
        print(cost)
        return cost

    def propagate(self, vector, A, b, Q, Kstar):
        """
        :param vector: the current deviation of the goal state
        :param idx: the current horizon at which the dynamics are estimated
        :return: the next state of the system and an update to control matrix
        """
        # add noise to the output

        try:
            self.Kstar = LA.solve_continuous_are(A, b, Q, self.R)
        except ValueError or np.linalg.LinAlgError:
            print('Using previous P')
        btk = np.matmul(b.T, Kstar)
        K = np.matmul(
            np.linalg.inv(
            np.matmul(
                btk,
                b
            )
            + self.R
            ),
            np.matmul(
                btk, A
            )

        )
        delta_pts = int(1/10)
        # Project the solver
        span = np.linspace(
            self.get_time_elapsed(),
            self.get_time_elapsed() + self.horizon,
        )
        # Drive deviation to zero by solving xdot = (A-BK)x
        # Output the noisy signal
        sol = integrate.solve_ivp(
            fun=self.sys_func,
            t_span=[span[0], span[-1]],
            y0=vector,
            args=(A, b, K),
            method='LSODA',
            t_eval=span
            )
        # Optimal Trajectory and Control which are used for the Certainty Equivalence strategy
        optimal_trajectory = sol.y
        optimal_control = np.matmul(-K, optimal_trajectory)
        # Only take the first action and limit the entries
        v_clipped = np.clip(optimal_control[0, [0]], self.vehicle.v_min, self.vehicle.v_max)

        # Apply control to current state
        control = np.array([v_clipped], dtype=float)
        return control

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

    def output_func(self, t, x, a, b, k, C):
        """
        :param t: Current time at which the dynamics are evaluated
        :param x: Current state of the system
        :param a: the system matrix
        :param b: the control matrix
        :param k: the gain matrix
        :param param_est: output matrix
        :return: output estimation for each time, t, evaluated with sys dynamics
        """

        xx = np.matmul(a - b.dot(k), x.reshape(self.num_states, 1))

        return np.matmul(C, xx).T

    @staticmethod
    def quadratic_cost(x, Q): return np.matmul(np.matmul(x.T, Q), x)

    def process_model(self, y, t, u, K, tau):
        # arguments
        #  y   = outputs
        #  t   = time
        #  u   = input value
        #  K   = process gain
        #  tau = process time constant

        # calculate derivative
        dydt = (-y + K * u) / (tau)

        return dydt

    @staticmethod
    def plot_state_history(time, state_history, output):

        "Plot history of the vehicle"
        color = 'red'
        time = np.arange(time)
        x_history = [item[0]for item in state_history]
        y_history = [item[1]for item in state_history]
        theta_history = [item[2] for item in state_history]
        output_history = [item for item in output]

        plt.plot(time, x_history, 'b--', label=r'$\phi_{0}$')
        plt.plot(time, y_history, 'm--', label=r'$\phi_{1}$')
        plt.plot(time, theta_history, 'k--', label=r'$\phi_{2}$')
        plt.legend()
        plt.ylabel("State Variables")
        plt.xlabel('Time Index')
        plt.xlim([time[0], time[-1]])
        plt.ylim([-10, 10])
        plt.grid(axis='both', color='0.95')
        plt.show()

        plt.figure()
        plt.plot(time, output_history, 'k--', label='x')
        plt.xlim([time[0], time[-1]])
        plt.ylabel('Output y(t)')
        plt.xlabel('Time Index')
        plt.grid(axis='both', color='0.95')
        plt.show()

    @staticmethod
    def plot_control_history(time, control_history):
        tt = np.arange(time-1)
        v_history = [item[0] for item in control_history]
        plt.figure()
        plt.plot(tt, v_history, 'm--')
        #plt.plot(tt, omega_history, 'y--', label='Angular Velocity')
        plt.xlabel('Time Index')
        plt.ylabel('Input u(t)')
        plt.xlim([tt[0], tt[-1]])
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
