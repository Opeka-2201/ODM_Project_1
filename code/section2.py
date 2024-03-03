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
    def __init__(self, rewards, gamma, bool_stochastic, prob_stochastic = 0):
        self.rewards = rewards
        self.gamma = gamma
        self.bool_stochastic = bool_stochastic
        self.nb_lines, self.nb_columns = rewards.shape
        self.prob_stochastic = prob_stochastic
    
    def reward(self, visited):
        return self.rewards[visited[0], visited[1]]
    
    def function_j(self, agent, N, N_runs):
        lines = self.nb_lines
        columns = self.nb_columns
        J = np.zeros((lines, columns))

        print("Computing J for domain:", "stochastic" if self.bool_stochastic else "deterministic")
        for _ in tqdm(range(N_runs)):
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
            J += J_run / N_runs
        return J
        
    @staticmethod
    def dynamic(state, action, nb_lines, nb_columns):
        return (min(max(state[0] + action[0], 0), nb_lines - 1), min(max(state[1] + action[1], 0), nb_columns - 1))
    
## MAIN ##
def main():
    ag = agent(ACTIONS_ALLOWED)
    det_dm = domain(REWARDS, GAMMA, False)
    sto_dm = domain(REWARDS, GAMMA, True, PROB_STOCHASTIC)

    j_det = det_dm.function_j(ag, N, N_RUNS_STOCHASTIC) # normalizing because of the random nature of the agent
    j_sto = sto_dm.function_j(ag, N, N_RUNS_STOCHASTIC)

    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(j_det, cmap="viridis", interpolation="nearest")
    axs[0].set_title("Deterministic domain")
    axs[0].set_xlabel("Columns")
    axs[0].set_ylabel("Rows")
    axs[0].set_xticks(np.arange(j_det.shape[1]))
    axs[0].set_yticks(np.arange(j_det.shape[0]))
    axs[0].set_xticklabels(np.arange(j_det.shape[1]))
    axs[0].set_yticklabels(np.arange(j_det.shape[0]))
    
    for i in range(j_det.shape[0]):
        for j in range(j_det.shape[1]):
            _ = axs[0].text(j, i, f'{j_det[i, j]:.2f}', ha="center", va="center", color="black")
    
    axs[1].imshow(j_sto, cmap="viridis", interpolation="nearest")
    axs[1].set_title("Stochastic domain")
    axs[1].set_xlabel("Columns")
    axs[1].set_ylabel("Rows")
    axs[1].set_xticks(np.arange(j_sto.shape[1]))
    axs[1].set_yticks(np.arange(j_sto.shape[0]))
    axs[1].set_xticklabels(np.arange(j_sto.shape[1]))
    axs[1].set_yticklabels(np.arange(j_sto.shape[0]))
    
    for i in range(j_sto.shape[0]):
        for j in range(j_sto.shape[1]):
            _ = axs[1].text(j, i, f'{j_sto[i, j]:.2f}', ha="center", va="center", color="black")
    
    plt.suptitle("$J^\mu_N$ for $N = " + str(N) + "$ and $N_{runs} = " + str(N_RUNS_STOCHASTIC) + "$")
    plt.show()

if __name__ == "__main__":
    main()