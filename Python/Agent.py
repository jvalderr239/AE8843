import numpy as np


class Agent(object):
    """
    Class representing the state of a vehicle
    : param state : (float) numpy array [x-coordinate, y-coordinate, yaw-angle].T
    """
    """
    phik+1 = A*phik + dt Bu
    yk+1 = Cxk+1 + dt
    xdot = vcos(theta)
    ydot = vsin(theta)
    thetadot = thetadot 
    xk+1 = xk + dt * vcos(theta)
    yk+1 = yk + dt* vsin(theta)
    thetak+1 = thetak + dt* thetadot
    A = [1 0 0; 
        0 1 0; 
        0 0 1];
    B = [cos(theta) 0]
        [sin(theta) 0][v]
        [0          1][thetadot]
    C = [1 0 0]
        [0 1 0]
        [0 0 1]
    Chat = [chat1 0 0]
            [0 chat2 0]
            [0 0 chat3]
    phi = [x]
        [y]
        [theta]
    phi_augmented = [ x  ]
                    [y    ]
                    [theta]
                    [chat1]
                    [chat2]
                    [chat3]
    phit, theta_hat, Pt = 100*sigma * eye
    y = theta * phit + noise
    """
    # state indices
    xidx, yidx, thetaidx = 0, 1, 2
    # control max vals
    wheel_base = .7
    near_zero = 1e-05
    def __init__(self, initial_state,
                 u_max=5,
                 maxTol=5):
        self.state_history = [initial_state]
        self.control_history = []
        self.state = initial_state
        self.maxTol = maxTol

        # Max Values
        self.v_max = u_max
        self.v_min = -u_max

        self.num_states = self.state.size
        self.num_controls = 1

        self.A = self.system_matrix()
        self.B = self.control_matrix()

    def update(self, control):
        self.state = np.matmul(self.system_matrix(),
                               self.state) \
                     + np.matmul(self.control_matrix(), control)
        self.state_history.append(self.state)
        self.control_history.append(control)

    def get_min(self):
        return np.array([self.v_min], dtype=float)

    def get_max(self):
        return np.array([self.v_max], dtype=float)

    def arrived(self):
        close = self.ssd(self.state[0:2])

        if close <= self.maxTol:
            print('---------------------------------------------')
            print('WELCOME, YOU HAVE ARRIVED TO YOU DESTINATION!')
            print('You are this:', close, " close")
            print('---------------------------------------------')
            return True
        print('You are this:', close, " close")
        return False

    def system_matrix(self):
        return np.array([[0, 1, -0.1],
                         [-2/3, -1/3, 1/3],
                         [0, 1/7, -3/7]], dtype=np.float)

    @staticmethod
    def control_matrix():
        return np.array([[0],
                         [-1/3],
                         [0]], dtype=np.float)

    @staticmethod
    def output_matrix():
        return np.array([5, 2, -1], dtype=np.float)

    @staticmethod
    def ssd(A, B):
        dif = np.subtract(A, B)
        print(dif)
        return np.dot(dif.T, dif)
