# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 1: Reinforcement Learning in a Discrete Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 4 System Identification: 

## IMPORTS ##
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from section3 import MDP, REWARDS, ACTIONS_ALLOWED, TRANSLATION, GAMMA, N, NB_LINES, NB_COLUMNS, PROB_STOCHASTIC, N_ITERATIONS, transition, translater_tuple_action
from section2 import domain, agent

N_ITERATIONS = 10 #to estimate the Q function

## DERIVED CONSTANTS ##

det_dom = domain(REWARDS, GAMMA, False)
sto_dom = domain(REWARDS, GAMMA, True)

det_mdp = MDP(False)
sto_mdp = MDP(True)

det_mdp.compute_Q(N_ITERATIONS)
sto_mdp.compute_Q(N_ITERATIONS)


# dict to convert the action tuple to an index, usefull in the p and r estimation
dict_actions = {(1,0): 0, (-1,0): 1, (0,1): 2, (0,-1): 3}



def compute_trajectory(domain, agent, N):
    """
    Computes the trajectory of an agent in a given domain.

    Args:
        domain (Domain): The domain in which the agent operates.
        agent (Agent): The agent that takes actions in the domain.
        N (int): The number of steps in the trajectory.

    Returns:
        list: A list of tuples representing the trajectory. Each tuple contains the state, action, reward, and next state at a given step.
    """
    lines = domain.nb_lines
    columns = domain.nb_columns
    trajectory = []
    state = (3,0)
    for _ in range(N):
        action = agent.chose_action()
        next_state = domain.dynamic(state, action, lines, columns)
        reward = domain.reward(next_state)
        trajectory.append((state, action, reward, next_state))
        state = next_state
    return trajectory


def estimate_r_p(trajectory):
    """
    Estimates the reward and transition probabilities based on the given trajectory.

    Args:
        trajectory (list): A list of tuples representing the trajectory. Each tuple contains the state, action, reward, and next state.

    Returns:
        tuple: A tuple containing the estimated reward matrix and transition probability matrix.

    """

    #Initialization of the reward and transition probability
    p_estimated = np.zeros((NB_LINES * NB_COLUMNS, 4, NB_LINES * NB_COLUMNS))
    r_estimated = np.zeros((NB_LINES * NB_COLUMNS, 4))
    r_estimated_count = np.zeros((NB_LINES * NB_COLUMNS, 4))

    #Estimation of the reward and transition probability
    for state, action, reward, next_state in trajectory:
        s = state[0] * NB_COLUMNS + state[1]
        a = dict_actions[action]
        s_prime = next_state[0] * NB_COLUMNS + next_state[1]
        r_estimated[s, a] += reward
        r_estimated_count[s, a] += 1
        p_estimated[s, a, s_prime] += 1

    #normalization of the reward and transition probability
    for i in range(NB_LINES * NB_COLUMNS):
        for j in range(4):
            if r_estimated_count[i, j] != 0:
                r_estimated[i, j] = r_estimated[i, j] / r_estimated_count[i, j]
    for i in range(NB_LINES * NB_COLUMNS):
        for j in range(4):
            s = np.sum(p_estimated[i, j])
            if s != 0:
                p_estimated[i, j] = p_estimated[i, j] / s
            else:
                p_estimated[i, j] = np.ones(NB_LINES * NB_COLUMNS) / NB_LINES / NB_COLUMNS

    return r_estimated, p_estimated

#Initialization of the true reward and transition probability
true_p_det = np.zeros((NB_LINES * NB_COLUMNS, 4, NB_LINES * NB_COLUMNS))
true_r_det = np.zeros((NB_LINES * NB_COLUMNS, 4))

true_p_sto = np.zeros((NB_LINES * NB_COLUMNS, 4, NB_LINES * NB_COLUMNS))
true_r_sto = np.zeros((NB_LINES * NB_COLUMNS, 4))

#Computation of the true reward and transition probability
for i in range(NB_LINES):
    for j in range(NB_COLUMNS):
        for k in range(4):
            for l in range(NB_LINES):
                for m in range(NB_COLUMNS):
                    true_p_det[i * NB_COLUMNS + j, k, l * NB_COLUMNS + m] = det_mdp.p_sp_s_a((i, j), ACTIONS_ALLOWED[k], (l, m))
                    true_p_sto[i * NB_COLUMNS + j, k, l * NB_COLUMNS + m] = sto_mdp.p_sp_s_a((i, j), ACTIONS_ALLOWED[k], (l, m))
            true_r_det[i * NB_COLUMNS + j, k] = det_mdp.r_s_a((i, j), ACTIONS_ALLOWED[k])
            true_r_sto[i * NB_COLUMNS + j, k] = sto_mdp.r_s_a((i, j), ACTIONS_ALLOWED[k])

#Initialization of the norm infinity and the number of steps for the computation of the trajectory        
N = [10, 100, 1000, 10000, 100000, 1000000]
norm_inf_det = np.zeros((len(N), 2))
norm_inf_sto = np.zeros((len(N), 2))


for i,n in enumerate(N):
    trajectory = compute_trajectory(det_dom, agent(ACTIONS_ALLOWED), n)
    r_estimated, p_estimated = estimate_r_p(trajectory)
    norm_inf_det[i] = [np.max(np.abs(r_estimated - true_r_det)), np.max(np.abs(p_estimated - true_p_det))]

    trajectory = compute_trajectory(sto_dom, agent(ACTIONS_ALLOWED), n)
    r_estimated, p_estimated = estimate_r_p(trajectory)
    norm_inf_sto[i] = [np.max(np.abs(r_estimated - true_r_sto)), np.max(np.abs(p_estimated - true_p_sto))]

plt.plot(N, norm_inf_det[:, 0], label = "r")
plt.plot(N, norm_inf_det[:, 1], label = "p")

plt.xlabel("N")
plt.ylabel("||true - estimated||")
plt.legend()
plt.title("Deterministic domain")

plt.show()

plt.plot(N, norm_inf_sto[:, 0], label = "r")
plt.plot(N, norm_inf_sto[:, 1], label = "p")

plt.xlabel("N")
plt.ylabel("||true - estimated||")
plt.legend()
plt.title("Stochastic domain")

plt.show()


# Constants from section3.py
REWARDS = np.matrix([
    [-3, 1, -5, 0, 19],
    [6, 3, 8, 9, 10],
    [5, -8, 4, 1, -8],
    [6, -9, 4, 19, -5],
    [-20, -17, -4, -3, 9]
])
ACTIONS_ALLOWED = [(1,0), (-1,0), (0,1), (0,-1)]
GAMMA = 0.99
N = 10000

# Derived constants
n_states = REWARDS.shape[0] * REWARDS.shape[1]
n_actions = len(ACTIONS_ALLOWED)

# Initialize the estimates
br = np.zeros((n_states, n_actions))
bp = np.zeros((n_states, n_actions, n_states))
bQ = np.zeros((n_states, n_actions))
counts = np.zeros((n_states, n_actions))

# Generate a trajectory using a random uniform policy
policy = np.ones((n_states, n_actions)) / n_actions
T = 1000
trajectory = compute_trajectory(domain, agent,T)

# Estimate r(s, a) and p(s'|s, a)
for t in range(T - 1):
    s, a, r, s_next = trajectory[t]
    counts[s, a] += 1
    br[s, a] += (r - br[s, a]) / counts[s, a]
    bp[s, a, s_next] += (1 - bp[s, a, s_next]) / counts[s, a]

# Compute bQ using br and bp
for s in range(n_states):
    for a in range(n_actions):
        for s_next in range(n_states):
            bQ[s, a] += bp[s, a, s_next] * (br[s, a] + GAMMA * np.max(bQ[s_next]))

# Derive bμ* from bQ
bmu_star = np.argmax(bQ, axis=1)

true_Q = np.zeros((n_states, n_actions))
for s in range(n_states):
    for a in range(n_actions):
        for s_next in range(n_states):
            true_Q[s, a] = (sto_mdp.Q_d[s],sto_mdp.Q_r[s],sto_mdp.Q_u[s],sto_mdp.Q_l[s])

# Calculate Jbμ*N and Jμ*N for each state
Jbmu_star_N = np.sum(bQ[np.arange(n_states), bmu_star])
Jmu_star_N = np.sum(true_Q[np.arange(n_states), bmu_star])

# Display the results
print("bQ:\n", bQ)
print("bmu_star:\n", bmu_star)
print("Jbmu_star_N:\n", Jbmu_star_N)
print("Jmu_star_N:\n", Jmu_star_N)

# Plot the convergence of bp and br
plt.figure()
plt.plot(np.linalg.norm(bp - true_p_sto, ord=np.inf, axis=2))
plt.plot(np.linalg.norm(br - true_r_sto, ord=np.inf, axis=1))
plt.legend(["bp", "br"])
plt.xlabel("Time step")
plt.ylabel("Infinite norm")
plt.show()