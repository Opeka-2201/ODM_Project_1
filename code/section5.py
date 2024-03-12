import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

from section4 import Agent, Domain, REWARDS, ACTIONS_ALLOWED, START_STATE, GAMMA, generate_trajectory
from section3 import translater_tuple_action, TRANSLATION

dom_sto = Domain(bool_stochastic=True)
dom_det = Domain(bool_stochastic=False)
ag = Agent()

T = 100000 # length of the trajectory

# We will generate a trajectory of 1000 steps
traj_sto = generate_trajectory(ag, dom_sto, T)
traj_det = generate_trajectory(ag, dom_det, T)


def Q_hat(trajectory, lr = 0.05, domain = dom_det, agent = ag):
    nb_states = domain.nb_columns * domain.nb_lines
    nb_actions = len(agent.actions_allowed)
    Q = np.zeros((nb_states, nb_actions))
    for step in tqdm(trajectory[:-1]):
        state, action, reward, next_state = step
        state_idx = state[0] * domain.nb_columns + state[1]
        action_idx = agent.actions_allowed.index(action)
        next_state_idx = next_state[0] * domain.nb_columns + next_state[1]
        Q[state_idx, action_idx] = (1-lr) * Q[state_idx, action_idx] + lr * (reward + GAMMA * np.max(Q[next_state_idx, :]))
    return Q

def deterministic_action(Q, state):
    state_idx = state[0] * dom_det.nb_columns + state[1]
    return np.argmax(Q[state_idx, :])

def policy_grid(Q, domain):
    pol = np.zeros((domain.nb_lines, domain.nb_columns))
    for i in range(domain.nb_lines):
        for j in range(domain.nb_columns):
            pol[i, j] = deterministic_action(Q, (i, j))
    return pol

Q_sto = Q_hat(traj_sto)
Q_det = Q_hat(traj_det)

# We will now compare the policies obtained with the deterministic and stochastic domains
pol_sto = policy_grid(Q_sto, dom_sto)
pol_det = policy_grid(Q_det, dom_det)

TRANSLATION = {0: "↓", 1: "↑", 2: "→", 3: "←"}


def policy_translater(policy, dict_translator = TRANSLATION):
    translated = policy.copy().astype(str)
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            print(policy[i, j])
            translated[i, j] = dict_translator[policy[i, j]]
    return translated

pol_sto_tr = policy_translater(pol_sto)
pol_det_tr = policy_translater(pol_det)



print("Policy for the stochastic domain")
print(pol_sto_tr)
print("Policy for the deterministic domain")
print(pol_det_tr)
print('#'*40)
print()
print()
print()
J_sto = dom_sto.function_j(pol_sto,N=1000)
print('Computing J for the deterministic domain')
J_det = dom_det.function_j(pol_det,N=1000)
print('J for the stochastic domain')
print(J_sto)
print('J for the deterministic domain')
print(J_det)


