import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy import integrate
from scipy import linalg as LA
from scipy.optimize import minimize as min

# generate seed for random generator
random.seed(239)


class RLdMPC:

    # State indices
    xidx, yidx, thetaidx = 0, 1, 2
    xmax, ymax, thetamax = 20, 20, np.pi/2

    # Define parameter weights
    qx = 1
    qy = 1
    qtheta = np.sqrt(np.pi / 100)
    qv = np.sqrt(1)
    qomega = np.sqrt(100)
    # Define time
    t0, tf = 0, 100
    num_pts = 100
    time = np.linspace(t0, tf, num_pts+1)
    dT = time[1]-time[0]
    time_elapsed = 0
    mean, sq_sigma = 0, 1
    yhat_idx, u_idx, phi_idx, z_idx, r_idx = 0, 1, 2, 3, 4

    def __init__(self, vehicle, C, N):

        self.horizon = N
        self.vehicle = vehicle
        self.num_states = vehicle.state.size
        self.num_controls = vehicle.B.shape[1]
        self.R = np.array([[self.qv, 0],
                           [0, self.qomega]], dtype=np.float)
        self.noise = np.random.normal(
            self.mean, self.sq_sigma, size=self.vehicle.state.shape)
        self.C = C
        self.C_est = np.zeros(self.C.shape)
        self.CCov = 1000*np.eye(self.num_states)
        self.G = 0
        self.time_elapsed = 0
        self.y = [np.matmul(self.C, self.vehicle.state) + self.noise]
        self.phi = [self.vehicle.state]
        self.P = [self.CCov]
        self.params = [self.C_est]
        self.controls = [np.zeros((self.num_controls, 1))]
        self.Kstar = np.zeros((self.num_states, self.num_states))

        # Initialize animation
        self.plot = Plotter(self.vehicle.state[self.xidx],
                            self.vehicle.state[self.yidx],
                            self.dT)
        self.U0 = np.zeros((self.num_controls, self.horizon+1))

        # Define constraints
        self.cons = [
            {'type': 'eq',
             'fun':
                 lambda state: np.matmul(state[self.r_idx], state[self.z_idx]) - state[self.phi_idx]},  # equality constraint
            {'type': 'ineq',
             'fun': lambda state: np.matmul(state[self.phi_idx].T, state[self.z_idx])}  # inequality constraint
         ]


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

    def compute_state(self):
        """
        This method propagates the state of the system by the horizon
        :return: Null
        """
        # Initialization
        phitk = self.vehicle.state
        ytk = np.zeros((self.num_states, 1))
        Rtk = np.zeros((self.num_states*self.num_states, 1))
        ztk = np.zeros((self.num_states, 1))
        ut = np.zeros((self.num_controls, 1))

        state = np.concatenate((ytk, ut, phitk, ztk, Rtk), axis=0)

        # Step 1
        for t in range(0, self.num_pts):
            A = self.vehicle.system_matrix()
            b = self.vehicle.control_matrix(
                self.vehicle.state[self.thetaidx]
            )
            try:
                self.Kstar = LA.solve_continuous_are(A, b, self.CCov, self.R)
            except ValueError or np.linalg.LinAlgError:
                print('Using previous P')

            for idx in range(0, self.horizon-1):

                if idx > 0:
                    control = self.propagate(
                        state[self.phi_idx, [idx - 1]],
                        A, b,
                        state[self.r_idx, [idx]],
                        self.Kstar
                    )
                    # update per horizon
                    # phi
                    phitk = np.matmul(
                        A,
                        phitk
                    ) + np.matmul(self.vehicle.control_matrix(
                        phitk[self.thetaidx]
                    ), control
                    )
                    # u
                    utk = control
                    # y
                    ytk = np.matmul(self.C_est, phitk)
                    # R
                    rtk += np.matmul(phitk.T, phitk).flatten()
                    ztk = np.matmul(rtk, phitk)
                else:
                    phitk = np.matmul(A, self.vehicle.state)

                    ytk = self.y[t]
                    rtk = np.linalg.inv(self.P[t]).flatten()
                    ztk = np.matmul(self.P[t], phitk)
                    utk = self.controls[t]
                    new_state = np.concatenate((ytk, ut, phitk, ztk, Rtk), axis=0)
                    state = np.concatenate(state, new_state, axis=1)
            optimal_solution = min(self.objective,
                                   state,
                                   constraints=self.cons,
                                   bounds=
                                   [
                                       (-10, 10),  # y bounds
                                       (self.vehicle.get_min(),
                                        self.vehicle.get_max()),  # u bounds
                                       (None, None),
                                       (None, None),
                                       (None, None)
                                   ],
                                   method='SLSQP',
                                   args=(self.R,
                                         self.Kstar,
                                         self.horizon))
            print(optimal_solution)
            optimal_control = optimal_solution.x



            print("----------------------------------")
            print("Current initial state and horizon: \n")
            print(self.y[self.xidx, :], "\n", self.y[self.yidx, :])
            print("----------------------------------")
            # Step 3 of RLdMPC

            new_dev = np.matmul(
                    self.vehicle.system_matrix(),
                    self.vehicle.state
                ) + np.matmul(self.vehicle.control_matrix(
                        self.vehicle.state[self.thetaidx]
                    ), optimal_control[:, [0]]
                )
            self.vehicle.update(new_dev, optimal_control[:, [0]])
            new_phi = self.vehicle.state
            new_output = np.matmul(self.C, self.vehicle.state)

            new_deviation = np.subtract(new_output,
                                        np.matmul(self.C_est, new_phi))
            G = np.matmul(np.matmul(self.CCov, new_phi),
                          np.linalg.inv(
                              self.sq_sigma +
                              self.quadratic_cost(new_phi, self.CCov)))

            self.CCov -= np.matmul(np.matmul(G, new_phi.T), self.CCov)
            self.C_est += np.matmul(G, new_deviation.T)
            # Store new values
            self.controls.append(optimal_control[:, [0]])
            self.phi.append(new_phi)
            self.y.append(new_output)
            self.P.append(self.CCov)
            self.params.append(self.C_est)
            self.update_time()

    def objective(self, state, R, Kstar, N):

        y = state[self.yhat_idx, [0]][0:N-1]
        phi = state[self.phi_idx, [0]][0:N-1]
        u = state[self.u_idx, [0]][0:N-1]
        z = state[self.z_idx, [0]][0:N-1]

        phi_final = state[self.phi_idx, [0]][N]
        y_squared = y.T.dot(y)
        u_squared = self.quadratic_cost(u, R)
        phiz = np.matmul(phi.T, z)
        terminal = self.quadratic_cost(phi_final, Kstar)

        return y_squared + u_squared + phiz + terminal


    """def objective(self, state,A,B,R,P,N):

        x0 = state[self.phi_idx]
        y0 = state[self.yhat_idx]
        Q = state[self.r_idx]
        sx = np.eye(A.ndim)
        su = np.zeros((A.ndim, B.shape[1] * N))

        # calc sx,su
        for i in range(N):
            # generate sx
            An = np.linalg.matrix_power(A, i + 1)
            sx = np.r_[sx, An]

            # generate su
            tmp = None
            for ii in range(i + 1):
                tm = np.linalg.matrix_power(A, ii) * B
                if tmp is None:
                    tmp = tm
                else:
                    tmp = np.c_[tm, tmp]

            for ii in np.arange(i, N - 1):
                tm = np.zeros(B.shape)
                if tmp is None:
                    tmp = tm
                else:
                    tmp = np.c_[tmp, tm]

            su = np.r_[su, tmp]

        tm1 = np.eye(N + 1)
        tm1[N, N] = 0
        tm2 = np.zeros((N + 1, N + 1))
        tm2[N, N] = 1
        Qbar = np.kron(tm1, Q) + np.kron(tm2, P)
        Rbar = np.kron(np.eye(N), R)

        uopt = -(su.T * Qbar * su + Rbar).I * su.T * Qbar * sx * x0
        #  print(uBa)
        costx = x0.T * (sx.T * Qbar * sx - sx.T * Qbar * su * (su.T * Qbar * su + Rbar).I * su.T * Qbar * sx) * x0
        costy = 

        #  print(costBa)

        return uopt, costBa"""

    def check_obsv(self, A, theta):

        pass

    def opt(self, vector, C,  P, Q):
        """
        :param vector: the current deviation of the goal state
        :param idx: the current horizon at which the dynamics are estimated
        :return: the next state of the system and an update to control matrix
        """
        # add noise to the output
        initial_dev = np.subtract(vector, self.vehicle.goal)
        theta = vector[self.thetaidx]

        deviation_from_goal = initial_dev.T[0]
        b = self.vehicle.control_matrix(theta)
        A = self.vehicle.system_matrix()
        print('-------Q-------')
        print(Q)
        print('---------------')
        try:
            Pt = LA.solve_continuous_are(A, b, Q, self.R)
        except ValueError:
            print('Using previous P')
            Pt = P

        K = np.matmul(np.linalg.inv(self.R), (np.dot(b.T, P)))
        eigVals, eigVecs = LA.eig(A-np.matmul(b, K))

        # Project the solver
        # Drive deviation to zero by solving xdot = (A-BK)x
        # Output the noisy signal
        sol = integrate.solve_ivp(
            fun=self.output_func,
            t_span=[self.get_time_elapsed(),
                    self.tf],
            y0=deviation_from_goal,
            args=(A, b, K, C),
            method='RK45',
            t_eval=[self.get_time_elapsed(),
                    self.tf],
            )
        # Optimal Trajectory and Control which are used for the Certainty Equivalence strategy
        optimal_trajectory = sol.y
        optimal_control = np.matmul(-K, optimal_trajectory)
        print(optimal_control)
        # Only take the first action and limit the entries
        v_clipped = np.clip(optimal_control[0, [0]], self.vehicle.v_min, self.vehicle.v_max)
        omega_clipped = np.clip(optimal_control[1, [0]], self.vehicle.omega_min, self.vehicle.omega_max)

        # Apply control to current state
        control = np.array([v_clipped, omega_clipped], dtype=float)
        du = self.dT * np.matmul(b, control)
        return du, control, Pt

    def propagate(self, vector, A, b, Q, Kstar):
        """
        :param vector: the current deviation of the goal state
        :param idx: the current horizon at which the dynamics are estimated
        :return: the next state of the system and an update to control matrix
        """
        # add noise to the output
        initial_dev = np.subtract(vector, self.vehicle.goal)
        theta = vector[self.thetaidx]

        deviation_from_goal = initial_dev.T[0]
        b = self.vehicle.control_matrix(theta)
        A = self.vehicle.system_matrix()
        K = np.matmul(np.linalg.inv(self.R), (np.dot(b.T, Kstar)))

        # Project the solver
        span = np.linspace(
            self.get_time_elapsed(),
            self.tf,
            self.tf - self.get_time_elapsed()
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

    def troubleshoot(self):

        "Plot history of the vehicle"
        color = 'red'
        x_history = [item[self.xidx]for item in self.vehicle.state_history]
        y_history = [item[self.yidx]for item in self.vehicle.state_history]

        plt.plot(x_history,y_history, 'bx', label='Path Traveled')
        plt.plot(self.vehicle.goal[self.xidx], self.vehicle.goal[self.yidx], 'gx', label='Goal')
        plt.plot(self.y[self.xidx, :], self.y[self.yidx, :], 'r--',
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




