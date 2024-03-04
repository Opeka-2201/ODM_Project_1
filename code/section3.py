# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 1: Reinforcement Learning in a Discrete Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 2: Optimal policy

## IMPORTS ##
import numpy as np
from tqdm import tqdm

## CONSTANTS ##
REWARDS = np.matrix([
    [-3, 1, -5, 0, 19],
    [6, 3, 8, 9, 10],
    [5, -8, 4, 1, -8],
    [6, -9, 4, 19, -5],
    [-20, -17, -4, -3, 9]
])
ACTIONS_ALLOWED = [(1,0), (-1,0), (0,1), (0,-1)]
TRANSLATION = {(1,0): "↓", (-1,0): "↑", (0,1): "→", (0,-1): "←"}
GAMMA = 0.99
N = 10000
N_RUNS_STOCHASTIC = 10
N_ITERATIONS = 19
PROB_STOCHASTIC = 0.5

## DERIVED CONSTANTS ##
NB_LINES = REWARDS.shape[0]
NB_COLUMNS = REWARDS.shape[1]

## FUNCTIONS ##
def transition(state, action):
    return (min(max(0, state[0] + action[0]), NB_LINES - 1), min(max(0, state[1] + action[1]), NB_COLUMNS - 1))

def translater_tuple_action(policy):
    translated = policy.copy()
    for i in range (NB_LINES):
        for j in range (NB_COLUMNS):
            translated[i, j] = TRANSLATION[policy[i, j]]
    return translated

## CLASSES ##
class MDP:
    def __init__(self, stochastic_behavior):
        self.stochastic_behavior = stochastic_behavior
        self.policy = None
        self.Q_u = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
        self.Q_d = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
        self.Q_r = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
        self.Q_l = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)

    def compute_policies_iteratively(self, N):
        current = None

        for n in range(1, N+1):
            previous = current
            self.Q_u, self.Q_d, self.Q_r, self.Q_l = self.compute_Q(n)
            self.compute_policy()
            current = self.policy
            print("Iteration:", n, " \tEqual policies:", np.array_equal(previous, current))

    def compute_Q(self, N):
        Q_u = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
        Q_d = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
        Q_r = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
        Q_l = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)

        for _ in range(N):
            temp_u = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
            temp_d = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
            temp_r = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)
            temp_l = np.zeros((NB_LINES, NB_COLUMNS), dtype=float)

            for i in range(NB_LINES):
                for j in range(NB_COLUMNS):
                    temp_u = self.reward_function(temp_u, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[0])
                    temp_d = self.reward_function(temp_d, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[1])
                    temp_r = self.reward_function(temp_r, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[2])
                    temp_l = self.reward_function(temp_l, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[3])
            
            # Swapping
            Q_u = temp_u
            Q_d = temp_d
            Q_r = temp_r
            Q_l = temp_l

        return Q_u, Q_d, Q_r, Q_l
    
    def reward_function(self, temp, Q_u, Q_d, Q_r, Q_l, state, action):
        state_prime = transition(state, action)
        r_s_a = self.r_s_a(state, action)
        p_sp_s_a = self.p_sp_s_a(state_prime, state, action)
        reward_function = p_sp_s_a * max(Q_u[state_prime[0], state_prime[1]], Q_d[state_prime[0], state_prime[1]], Q_r[state_prime[0], state_prime[1]], Q_l[state_prime[0], state_prime[1]])

        if self.stochastic_behavior:
            p_stoch = self.p_sp_s_a((0, 0), state, action)
            reward_function += p_stoch * max(Q_u[0, 0], Q_d[0, 0], Q_r[0, 0], Q_l[0, 0])

        temp[state[0], state[1]] = r_s_a + GAMMA * reward_function
        return temp

    def r_s_a(self, state, action):
        state_prime = transition(state, action)
        if self.stochastic_behavior:
            return PROB_STOCHASTIC * REWARDS[0, 0] + (1 - PROB_STOCHASTIC) * REWARDS[state_prime[0], state_prime[1]]
        else:
            return REWARDS[state_prime[0], state_prime[1]]

    def p_sp_s_a(self, state_prime, state, action):
        visited = transition(state, action)
        if self.stochastic_behavior:
            return PROB_STOCHASTIC * (1 if state_prime == visited else 0) + (1 - PROB_STOCHASTIC) * (1 if state_prime == (0, 0) else 0)
        else:
            return 1 if state_prime == visited else 0

    def compute_policy(self):
        self.policy = np.zeros((NB_LINES, NB_COLUMNS), dtype=object)
        for i in range(NB_LINES):
            for j in range(NB_COLUMNS):
                self.policy[i, j] = ACTIONS_ALLOWED[np.argmax([self.Q_u[i, j], self.Q_d[i, j], self.Q_r[i, j], self.Q_l[i, j]])]

    def policy_in_state(self, state):
        if self.policy is None:
            self.compute_policy()
        return self.policy[state[0], state[1]]

    def function_j(self, N):
        J_N = np.zeros((NB_LINES, NB_COLUMNS))
        for _ in range(N):
            temp = np.zeros((NB_LINES, NB_COLUMNS))

            for i in range(NB_LINES):
                for j in range(NB_COLUMNS):
                    state = (i, j)
                    action = self.policy_in_state(state)
                    state_prime = transition(state, action)
                    r_s_a = self.r_s_a(state, action)

                    if self.stochastic_behavior:
                        temp[state[0], state[1]] = r_s_a + GAMMA * (PROB_STOCHASTIC * J_N[state_prime] + (1 - PROB_STOCHASTIC) * J_N[0, 0])
                    else:
                        temp[state[0], state[1]] = r_s_a + GAMMA * J_N[state_prime]
            J_N = temp
        return J_N
    
## MAIN ##
def main():
    print("Deterministic MDP")
    deterministic_mdp = MDP(False)
    deterministic_mdp.compute_policies_iteratively(N_ITERATIONS)
    print("Policy:\n", translater_tuple_action(deterministic_mdp.policy))
    print("Expected return:\n", deterministic_mdp.function_j(N))

    print("Stochastic MDP")
    stochastic_mdp = MDP(True)
    stochastic_mdp.compute_policies_iteratively(N_ITERATIONS)
    print("Policy:\n", translater_tuple_action(stochastic_mdp.policy))
    print("Expected return:\n", stochastic_mdp.function_j(N))

if __name__ == "__main__":
    main()