import numpy as np
from scipy import linalg as LA

from Agent import Agent
from RLdMPC import RLdMPC
np.random.seed(239)
# Define tolerance


# Choose horizon
mean, sq_sigma = 0, 1
horizon = 3
start = np.random.normal(
            mean, sq_sigma, size=(3, 1))
output_param = np.zeros((1, 3))

# Initialize the vehicle
agent_optimal_mu = Agent(start)
agent_rldmpc = Agent(start)
# Start MPC

# Optimal mu solution
"""
opt_mu = RLdMPC(agent_optimal_mu, output_param, horizon)
opt_mu.compute_state_with_opt_mu()
agent_optimal_states = agent_optimal_mu.state_history
agent_optimal_control = agent_optimal_mu.control_history
tt_opt = opt_mu.get_time_elapsed()+2
agent_optimal_output = opt_mu.y

RLdMPC.plot_state_history(tt_opt, agent_optimal_states, agent_optimal_output)
RLdMPC.plot_control_history(tt_opt, agent_optimal_control)
"""
# Primal Control
opt_rldmpc = RLdMPC(agent_rldmpc, output_param, horizon)
opt_rldmpc.compute_state_with_rldmpc()
agent_optimal_states = agent_optimal_mu.state_history
agent_optimal_control = agent_optimal_mu.control_history
tt_opt = opt_rldmpc.get_time_elapsed()+1
agent_optimal_output = opt_rldmpc.y

RLdMPC.plot_state_history(tt_opt, agent_optimal_states, agent_optimal_output)
RLdMPC.plot_control_history(tt_opt, agent_optimal_control)
