# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 1: Reinforcement Learning in a Discrete Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 2: Expected return of a policy

## IMPORTS ##
import random
import numpy as np
import matplotlib.pyplot as plt
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
GAMMA = 0.99
N = 10000
N_RUNS_STOCHASTIC = 10
PROB_STOCHASTIC = 0.5

## CLASSES ##
class agent:
    def __init__(self, actions_allowed):
        self.actions_allowed = actions_allowed

    def chose_action(self):
        return random.choice(self.actions_allowed)
    
class domain:
    def __init__(self, rewards, gamma, bool_stochastic, prob_stochastic = 0, actions_allowed = ACTIONS_ALLOWED, N = N, N_runs = N_RUNS_STOCHASTIC):
        self.rewards = rewards
        self.gamma = gamma
        self.bool_stochastic = bool_stochastic
        self.nb_lines, self.nb_columns = rewards.shape
        self.prob_stochastic = prob_stochastic
        self.estimated_p = self.estimated_p(N)
        self.estimated_r = self.estimated_r(N)
        self.estimated_Q = self.estimated_Q(N)
        self.actions_allowed = actions_allowed

        
    def reward(self, visited):
        return self.rewards[visited[0], visited[1]]
    
    def function_j(self, agent, N, N_runs):
        lines = self.nb_lines
        columns = self.nb_columns
        J = np.zeros((N_runs, lines, columns))

        print("Computing J for domain:", "stochastic" if self.bool_stochastic else "deterministic")
        for i in tqdm(range(N_runs)):
            J_run = np.zeros((lines, columns))
            for _ in range(N):
                J_new = np.zeros((lines, columns))
                
                for i in range(lines):
                    for j in range(columns):
                        state = self.dynamic((i,j), agent.chose_action(), lines, columns)
                        if self.bool_stochastic:
                            J_new[i,j] = self.prob_stochastic * (self.reward(state) + self.gamma * J_run[state[0], state[1]]) + \
                                         (1 - self.prob_stochastic) * (self.reward((0,0)) + self.gamma * J_run[0,0])
                        else:
                            J_new[i,j] = self.reward(state) + self.gamma * J_run[state[0], state[1]]

                J_run = J_new
            J[i] = J_run
        return J

    def estimated_p(self, N):
        lines = self.nb_lines
        columns = self.nb_columns
        P = np.zeros((lines, columns, len(self.actions_allowed)))

        print("Computing P for domain:", "stochastic" if self.bool_stochastic else "deterministic")
        for i in tqdm(range(N)):
            for i in range(lines):
                for j in range(columns):
                    for a in range(len(self.actions_allowed)):
                        state = self.dynamic((i,j), self.actions_allowed[a], lines, columns)
                        P[i,j,a] = self.prob_stochastic * (self.reward(state) + self.gamma * self.J[state[0], state[1]]) + \
                                    (1 - self.prob_stochastic) * (self.reward((0,0)) + self.gamma * self.J[0,0])
        return P
    
    def estimated_r(self, N):
        lines = self.nb_lines
        columns = self.nb_columns
        R = np.zeros((lines, columns, len(self.actions_allowed)))

        print("Computing R for domain:", "stochastic" if self.bool_stochastic else "deterministic")
        for i in tqdm(range(N)):
            for i in range(lines):
                for j in range(columns):
                    for a in range(len(self.actions_allowed)):
                        state = self.dynamic((i,j), self.actions_allowed[a], lines, columns)
                        R[i,j,a] = self.reward(state)
        return R
    
    def estimated_Q(self, N):
        #using estimated_r and estimated_p
        lines = self.nb_lines
        columns = self.nb_columns
        Q = np.zeros((lines, columns, len(self.actions_allowed)))

        print("Computing Q for domain:", "stochastic" if self.bool_stochastic else "deterministic")
        for i in tqdm(range(N)):
            for i in range(lines):
                for j in range(columns):
                    for a in range(len(self.actions_allowed)):
                        Q[i,j,a] = self.estimated_r[i,j,a] + self.gamma * np.max([self.estimated_p[i,j,a] * self.J[state[0], state[1]] for state in self.dynamic((i,j), self.actions_allowed[a], lines, columns)])
        return Q
    

    @staticmethod
    def dynamic(state, action, nb_lines, nb_columns):
        return (min(max(state[0] + action[0], 0), nb_lines - 1), min(max(state[1] + action[1], 0), nb_columns - 1))
    

## MAIN ##
# Create the agent and the domain
a = agent(ACTIONS_ALLOWED)
d = domain(REWARDS, GAMMA, False)
d_sto = domain(REWARDS, GAMMA, True, PROB_STOCHASTIC)

def main():
    # Compute J for the deterministic domain
    J_det = d.function_j(a, N, N_RUNS_STOCHASTIC)
    J_sto = d_sto.function_j(a, N, N_RUNS_STOCHASTIC)
    return J_det, J_sto

J_det, J_sto = main()

# Display the results
print("J_det:\n", J_det)
print("J_sto:\n", J_sto)
print("J_det.shape:", J_det.shape)
print("J_sto.shape:", J_sto.shape)
print("J_det[0]:\n", J_det[0])
print("J_sto[0]:\n", J_sto[0])
print("J_det[0].shape:", J_det[0].shape)
print("J_sto[0].shape:", J_sto[0].shape)
print("J_det[0][0]:\n", J_det[0][0])
print("J_sto[0][0]:\n", J_sto[0][0])
print("J_det[0][0].shape:", J_det[0][0].shape)
print("J_sto[0][0].shape:", J_sto[0][0].shape)
print("J_det[0][0][0]:\n", J_det[0][0][0])
print("J_sto[0][0][0]:\n", J_sto[0][0][0])
print("J_det[0][0][0].shape:", J_det[0][0][0].shape)
