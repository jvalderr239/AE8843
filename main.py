import numpy as np
from scipy import linalg as LA

from BicycleModel import BicycleModel
from RLdMPC_Controller import RLdMPC

# Define tolerance
max_tolerance = .001

# Choose horizon
horizon = 5
start = np.array([[-8], [-9], [0]], dtype=np.float)
goal = np.array([[8], [9], [0]], dtype=np.float)
output_param = np.random.rand(3, 3)
# Initialize the vehicle
test_vehicle = BicycleModel(start, goal)

# Start MPC

rldmpc = RLdMPC(test_vehicle, output_param, horizon)
while rldmpc.get_time_elapsed() < rldmpc.tf:

    xo = rldmpc.compute_state()
    print("----UPDATING INITIAL STATE----")
    new_initial = xo[:, [1]]
    print(new_initial)

    if LA.norm(new_initial-goal) <= max_tolerance:
        print("---REACHED MAX TOLERANCE---")
        print(max_tolerance)
        break
rldmpc.troubleshoot()