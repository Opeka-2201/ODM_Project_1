# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 1: Reinforcement Learning in a Discrete Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 3: Optimal policy

## IMPORTS ##
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


from section3 import MDP, REWARDS, ACTIONS_ALLOWED, TRANSLATION, GAMMA, N, NB_LINES, NB_COLUMNS, PROB_STOCHASTIC, N_ITERATIONS, transition, translater_tuple_action
from section2 import domain, agent

det_dom = domain(False)
sto_dom = domain(True)

def compute_trajectory(domain, agent, N):
    #return list of tuples (state, action, reward,next_state)
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

def compute_J(domain, agent, N, N_runs):
    return domain.function_j(agent, N, N_runs, mean = True)


def traj_to_dict(trajectory):
    #return dictionary with keys (states, actions) and values (rewards, next_states)
    r_dict = {}
    p_dict = {}
    for state, action, reward, next_state in trajectory:
        r_dict[(state, action)] = reward
        p_dict[(state, action)] = next_state

    return r_dict, p_dict

#Implement a routine which estimates r (s, a) and p (s, a) from a trajectory
def estimate_r_p(trajectory):
    #return dictionary with keys (states, actions) and values (rewards, next_states)
    r_dict = {}
    p_dict = {}
    for state, action, reward, next_state in trajectory:
        if (state, action) in r_dict:
            r_dict[(state, action)] += reward
            p_dict[(state, action)].append(next_state)
        else:    
            r_dict[(state, action)] = reward
            p_dict[(state, action)] = [next_state]

    for key in r_dict:
        r_dict[key] = r_dict[key] / len(p_dict[key])
        # i want in each key of p_dict a list of next states associated with a probability
        # i.e. if i have 2 next states, i want to have 2 probabilities

        # i will use the following formula to compute the probability of each next state : sum of given next state / total number of next states
        # this is equivalent to the number of times the next state appears in the list of next states / total number of next states

        for next_state in p_dict[key]:
            p_dict[key][p_dict[key].index(next_state)] = p_dict[key].count(next_state) / len(p_dict[key])

    return r_dict, p_dict


det_MDP = MDP(False)
sto_MDP = MDP(True)

det_psa = det_MDP.p_sp_s_a
sto_psa = sto_MDP.p_sp_s_a

def compute_Q(r_dict, p_dict, gamma, N):
    Q = {}
    for state, action in r_dict:
        Q[(state, action)] = r_dict[(state, action)]
        for next_state in p_dict[(state, action)]:
            Q[(state, action)] += gamma * p_dict[(state, action)][p_dict[(state, action)].index(next_state)] * N[next_state]
    return Q

def compute_N(Q, N, N_runs):
    for _ in range(N_runs):
        N_new = {}
        for state in N:
            N_new[state] = max([Q[(state, action)] for action in ACTIONS_ALLOWED])
        N = N_new
    return N

def main():
    ag = agent(ACTIONS_ALLOWED)
    det_dm = domain(REWARDS, GAMMA, False)
    sto_dm = domain(REWARDS, GAMMA, True, PROB_STOCHASTIC)

    j_det = det_dm.function_j(ag, N, 100) # normalizing because of the random nature of the agent
    j_det_mean = np.mean(j_det, axis=0)
    print(j_det_mean.shape)

    trajectory = compute_trajectory(det_dm, ag, 100)
    r_dict, p_dict = traj_to_dict(trajectory)
    r_dict_est, p_dict_est = estimate_r_p(trajectory)

    Q = compute_Q(r_dict, p_dict, GAMMA, j_det_mean)
    N = compute_N(Q, j_det_mean, 100)
    print(N)

    j_det = det_dm.function_j(ag, N, 100) # normalizing because of the random nature of the agent
    j_det_mean = np.mean(j_det, axis=0)
    print(j_det_mean.shape)



    print("Deterministic MDP")
    deterministic_mdp = MDP(False)
    deterministic_mdp.compute_policies_iteratively(N_ITERATIONS)
    print("Policy:\n", translater_tuple_action(deterministic_mdp.policy))
    j_det = deterministic_mdp.function_j(N)
    print("Expected return:\n", j_det)

    print("Stochastic MDP")
    stochastic_mdp = MDP(True)
    stochastic_mdp.compute_policies_iteratively(N_ITERATIONS)
    print("Policy:\n", translater_tuple_action(stochastic_mdp.policy))
    j_sto = stochastic_mdp.function_j(N)
    print("Expected return:\n", j_sto)

    plt.imshow(j_det_mean, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(j_det, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(j_sto, cmap='hot', interpolation='nearest')
    plt.show()
