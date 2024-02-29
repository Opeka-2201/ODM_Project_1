# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 1: Reinforcement Learning in a Discrete Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 1: Implementation of the domain

## IMPORTS ##
import random
import numpy as np

## CONSTANTS ##
REWARDS = np.matrix([
    [-3, 1, -5, 0, 19],
    [6, 3, 8, 9, 10],
    [5, -8, 4, 1, -8],
    [6, -9, 4, 19, -5],
    [-20, -17, -4, -3, 9]
])
ACTIONS_ALLOWED = [(1,0), (-1,0), (0,1), (0,-1)]
START_STATE = (3, 0)
GAMMA = 0.99
NB_STEPS = 10
PROB_STOCHASTIC = 0.5

## CLASSES ##
class agent:
    def __init__(self, actions_allowed):
        self.actions_allowed = actions_allowed

    def chose_action(self):
        return random.choice(self.actions_allowed)
    
class domain:
    def __init__(self, rewards, start_state, gamma, bool_stochastic, prob_stochastic = 0):
        self.rewards = rewards
        self.state = start_state
        self.gamma = gamma
        self.bool_stochastic = bool_stochastic
        self.nb_lines, self.nb_columns = rewards.shape
        self.prob_stochastic = prob_stochastic
    
    def reward(self, visited):
        return self.rewards[visited[0], visited[1]]

    def step(self, action):
        state = self.state

        visited = self.dynamic(state, action, self.nb_lines, self.nb_columns)
        if self.bool_stochastic and random.random() > self.prob_stochastic:
            visited = (0, 0)

        reward = self.reward(visited)
        self.state = visited

        return (state, action, reward, visited)
    
    @staticmethod
    def dynamic(state, action, nb_lines, nb_columns):
        return (min(max(state[0] + action[0], 0), nb_lines - 1), min(max(state[1] + action[1], 0), nb_columns - 1))
    
## MAIN ##
def main():
    ag = agent(ACTIONS_ALLOWED)
    det_dm = domain(REWARDS, START_STATE, GAMMA, False)
    sto_dm = domain(REWARDS, START_STATE, GAMMA, True, PROB_STOCHASTIC)

    print("Deterministic domain\t\t\t Stochastic domain")

    for _ in range(NB_STEPS):
        action_det = ag.chose_action()
        action_sto = ag.chose_action()
        print(det_dm.step(action_det), "\t\t", sto_dm.step(action_sto))

if __name__ == "__main__":
    main()