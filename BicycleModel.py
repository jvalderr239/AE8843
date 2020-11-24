import numpy as np


class BicycleModel(object):
    """
    Class representing the state of a vehicle
    : param state : (float) numpy array [x-coordinate, y-coordinate, yaw-angle].T
    """
    # state indices
    xidx, yidx, thetaidx = 0, 1, 2
    # control max vals
    wheel_base = .7
    near_zero = 1e-05
    def __init__(self, initial_state, goal, v_max=3, omega_max=5):

        self.state_history = [initial_state]
        self.control_history = []
        self.state = initial_state
        self._goal = goal

        #Max Values
        self.v_max = v_max
        self.v_min = -v_max
        self.omega_max = omega_max
        self.omega_min = -omega_max

        self.A = self.system_matrix()
        self.B = self.control_matrix(initial_state[self.thetaidx])

    def update(self, state, control):
        self.state = state
        self.state_history.append(self.state)
        self.control_history.append(control)

    def move(self, controls, dT):

        self.state += self.get_throttle(controls, dT)
        self.state_history.append(self.state)
        return self.get_throttle(controls, dT)

    def get_throttle(self, controls, dT):
        return dT*np.matmul(self.B, controls)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, destination):
        self._goal = destination

    def system_matrix(self):

        return np.array([[1, self.near_zero, self.near_zero],
                        [self.near_zero, 1, self.near_zero],
                        [self.near_zero, self.near_zero, 1]], dtype=np.float)

    @staticmethod
    def control_matrix(theta):
        return np.array([[np.cos(theta), 0],
                         [np.sin(theta), 0],
                        [0, 1]], dtype=np.float)

