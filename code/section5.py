# INFO8003-1: Optimal Decision Making for Complex Problems
# Project 1: Reinforcement Learning in a Discrete Domain
# Authors: Romain LAMBERMONT, Arthur LOUIS
# Section 5: Q-Learning in a Batch Setting

# Importing relevant functions and classes from the previous sections
from section3 import transition, translater_tuple_action, MDP, REWARDS, ACTIONS_ALLOWED, NB_LINES, NB_COLUMNS, GAMMA, PROB_STOCHASTIC
from section3 import GAMMA as GAMMA_3

## IMPORTS
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

## CONSTANTS ##
INITIAL_STATE = (3,0)
LEARNING_RATE = 0.05
TRAJECTORY_LENGTH = 1_000_000
EPSILON = 0.5
EPOCHS = 100
STEPS = 1000

## DERIVED CONSTANTS ##
NB_LINES = REWARDS.shape[0]
NB_COLUMNS = REWARDS.shape[1]

# CLASSES
class Q_Learning:
    def __init__(self, stochastic_behaviour, trajectory_length, initial_state, epsilon, gamma):
        self.stochastic_behaviour = stochastic_behaviour
        self.trajectory = self.generate_trajectory(trajectory_length, initial_state)
        self.trajectory_length = trajectory_length
        self.initial_state = initial_state
        self.epsilon = epsilon
        self.gamma = gamma

        self.estimated_Q = self.offline_learning()
        self.estimated_policy = self.compute_policy()

        self.estimated_policy_offline = self.estimated_policy

    def compute_policy(self):
        # Compute the policy from the Q values by fetching the action with the highest Q value
        self.policy = np.zeros((NB_LINES, NB_COLUMNS), dtype=object)
        for i in range(NB_LINES):
            for j in range(NB_COLUMNS):
                self.policy[i, j] = ACTIONS_ALLOWED[np.argmax(self.estimated_Q[i, j])]

        return self.policy

    def policy_in_state(self, state):
        # Return the action in a given state to follow the policy
        if self.policy is None:
            self.compute_policy()
        return self.policy[state[0], state[1]]

    def function_j(self, N):
        # Compute the expected return for a given policy (same as before but with an optimal policy instead of a random one)
        J_N = np.zeros((NB_LINES, NB_COLUMNS))
        for _ in range(N):
            temp = np.zeros((NB_LINES, NB_COLUMNS))

            for i in range(NB_LINES):
                for j in range(NB_COLUMNS):
                    state = (i, j)
                    action = self.policy_in_state(state)
                    state_prime = transition(state, action)
                    reward = REWARDS[state_prime[0], state_prime[1]]

                    if self.stochastic_behaviour:
                        temp[state[0], state[1]] = reward + GAMMA * (PROB_STOCHASTIC * J_N[state_prime] + (1 - PROB_STOCHASTIC) * J_N[0, 0])
                    else:
                        temp[state[0], state[1]] = reward + GAMMA * J_N[state_prime]
            J_N = temp
        return J_N

    def generate_trajectory(self, N_steps, initial_state):
        trajectory = list()
        trajectory.append(initial_state)
        x, y = initial_state
        print("Generating trajectory...")
        for _ in tqdm(range(N_steps)):
            a = random.choice(ACTIONS_ALLOWED)
            x, y = transition((x, y), a)

            if self.stochastic_behaviour:
                if random.uniform(0, 1) > 1/2:
                    x, y = (0, 0)

            r = REWARDS[x, y]

            trajectory.append(a)
            trajectory.append(r)
            trajectory.append((x, y))

        return trajectory
    
    def offline_learning(self):
        estimated_Q = np.zeros((NB_LINES, NB_COLUMNS, len(ACTIONS_ALLOWED)))
        print("\nLearning Q-values offline for the beahviour:", "stoachastic" if self.stochastic_behaviour else "deterministic")
        for i in tqdm(range(0, len(self.trajectory)-1, 3)):
            state = self.trajectory[i]
            action = self.trajectory[i+1]
            reward = self.trajectory[i+2]
            state_prime = self.trajectory[i+3]

            action_taken = ACTIONS_ALLOWED.index(action)
            estimated_Q[state[0], state[1], action_taken] = LEARNING_RATE * (reward + self.gamma * np.max(estimated_Q[state_prime[0], state_prime[1], :])) + \
                                                            (1 - LEARNING_RATE) * estimated_Q[state[0], state[1], action_taken]
            
        return estimated_Q
    
    def first_online_learning(self):
        infinite_norm = np.zeros(EPOCHS)
        infinite_norm_2 = np.zeros(EPOCHS)
        mdp = MDP(self.stochastic_behaviour)
        mdp.compute_policies_iteratively(10)
        j_mu = mdp.function_j(5000)
        mdp_Q = np.array([mdp.Q_u, mdp.Q_d, mdp.Q_r, mdp.Q_l])
        mdp_Q = np.moveaxis(mdp_Q, 0, -1)

        estimated_Q = np.zeros((NB_LINES, NB_COLUMNS, len(ACTIONS_ALLOWED)))
        self.estimated_Q = estimated_Q
        self.estimated_policy = self.compute_policy()

        x, y = self.initial_state

        print("\nFirst method of online learning for the behaviour:", "stoachastic" if self.stochastic_behaviour else "deterministic")
        for epoch in tqdm(range(EPOCHS)):
            for _ in range(STEPS):
                action = self.policy_in_state((x, y)) if random.uniform(0, 1) > self.epsilon else random.choice(ACTIONS_ALLOWED)
                state_prime = transition((x, y), action)
                action_taken = ACTIONS_ALLOWED.index(action)
                r = REWARDS[state_prime[0], state_prime[1]]

                estimated_Q[x, y, action_taken] = LEARNING_RATE * (r + self.gamma * np.max(estimated_Q[state_prime[0], state_prime[1], :])) + \
                                                (1 - LEARNING_RATE) * estimated_Q[x, y, action_taken]
                
                x, y = state_prime
                self.estimated_Q = estimated_Q
                self.estimated_policy = self.compute_policy()

            estimated_j_mu = self.function_j(5000)
            infinite_norm[epoch] = np.max(np.abs(j_mu - estimated_j_mu))
            infinite_norm_2[epoch] = np.max(np.abs(mdp_Q - estimated_Q))

        if GAMMA == 0.9:
            plt.figure()
            plt.plot(range(EPOCHS), infinite_norm)
            plt.xlabel("Epochs")
            plt.ylabel("Infinite norm")
            plt.title("Convergence of the expected return for the 1st method\n with behaviour: " + ("stochastic " if self.stochastic_behaviour else "deterministic ") + "and " + r"$\gamma =$" + str(GAMMA_3))
            plt.savefig("figures/first_online_learning_" + ("stochastic_" if self.stochastic_behaviour else "deterministic_") + "gamma_" + str(GAMMA_3) + ".png")
            plt.close()

        if GAMMA == 0.4:
            plt.figure()
            plt.plot(range(EPOCHS), infinite_norm_2)
            plt.xlabel("Epochs")
            plt.ylabel("Infinite norm")
            plt.title(r"Convergence of $\hat{Q}$ " + "for the 1st method\n with behaviour: " + ("stochastic " if self.stochastic_behaviour else "deterministic ") + "and " + r"$\gamma =$" + str(GAMMA_3))
            plt.savefig("figures/first_online_learning_" + ("stochastic_" if self.stochastic_behaviour else "deterministic_") + "gamma_" + str(GAMMA_3) + ".png")
            plt.close()

    
    def second_online_learning(self):
        infinite_norm = np.zeros(EPOCHS)
        infinite_norm_2 = np.zeros(EPOCHS)
        mdp = MDP(self.stochastic_behaviour)
        mdp.compute_policies_iteratively(10)
        j_mu = mdp.function_j(5000)
        mdp_Q = np.array([mdp.Q_u, mdp.Q_d, mdp.Q_r, mdp.Q_l])
        mdp_Q = np.moveaxis(mdp_Q, 0, -1)

        estimated_Q = np.zeros((NB_LINES, NB_COLUMNS, len(ACTIONS_ALLOWED)))
        self.estimated_Q = estimated_Q
        self.estimated_policy = self.compute_policy()

        x, y = self.initial_state

        print("\nSecond method of online learning for the behaviour:", "stoachastic" if self.stochastic_behaviour else "deterministic")
        for epoch in tqdm(range(EPOCHS)):
            lr = LEARNING_RATE
            for _ in range(STEPS):
                action = self.policy_in_state((x, y)) if random.uniform(0, 1) > self.epsilon else random.choice(ACTIONS_ALLOWED)
                state_prime = transition((x, y), action)
                action_taken = ACTIONS_ALLOWED.index(action)
                r = REWARDS[state_prime[0], state_prime[1]]

                estimated_Q[x, y, action_taken] = lr * (r + self.gamma * np.max(estimated_Q[state_prime[0], state_prime[1], :])) + \
                                                (1 - lr) * estimated_Q[x, y, action_taken]
                
                x, y = state_prime
                self.estimated_Q = estimated_Q
                self.estimated_policy = self.compute_policy()
                lr = lr * 0.8

            estimated_j_mu = self.function_j(5000)
            infinite_norm[epoch] = np.max(np.abs(j_mu - estimated_j_mu))
            infinite_norm_2[epoch] = np.max(np.abs(mdp_Q - estimated_Q))

        if GAMMA == 0.9:
            plt.figure()
            plt.plot(range(EPOCHS), infinite_norm)
            plt.xlabel("Epochs")
            plt.ylabel("Infinite norm")
            plt.title("Convergence of the expected return for the 2nd method\n with behaviour: " + ("stochastic " if self.stochastic_behaviour else "deterministic ") + "and " + r"$\gamma =$" + str(GAMMA_3))
            plt.savefig("figures/second_online_learning_" + ("stochastic_" if self.stochastic_behaviour else "deterministic_") + "gamma_" + str(GAMMA_3) + ".png")
            plt.close()

        if GAMMA == 0.4:
            plt.figure()
            plt.plot(range(EPOCHS), infinite_norm_2)
            plt.xlabel("Epochs")
            plt.ylabel("Infinite norm")
            plt.title(r"Convergence of $\hat{Q}$ " + "for the 2nd method\n with behaviour: " + ("stochastic " if self.stochastic_behaviour else "deterministic ") + "and " + r"$\gamma =$" + str(GAMMA_3))
            plt.savefig("figures/second_online_learning_" + ("stochastic_" if self.stochastic_behaviour else "deterministic_") + "gamma_" + str(GAMMA_3) + ".png")
            plt.close()

    def third_online_learning(self):
        infinite_norm = np.zeros(EPOCHS)
        infinite_norm_2 = np.zeros(EPOCHS)
        mdp = MDP(self.stochastic_behaviour)
        mdp.compute_policies_iteratively(10)
        j_mu = mdp.function_j(5000)
        mdp_Q = np.array([mdp.Q_u, mdp.Q_d, mdp.Q_r, mdp.Q_l])
        mdp_Q = np.moveaxis(mdp_Q, 0, -1)

        estimated_Q = np.zeros((NB_LINES, NB_COLUMNS, len(ACTIONS_ALLOWED)))
        self.estimated_Q = estimated_Q
        self.estimated_policy = self.compute_policy()

        x, y = self.initial_state
        buffer = []

        print("\nThird method of online learning for the behaviour:", "stoachastic" if self.stochastic_behaviour else "deterministic")
        for epoch in tqdm(range(EPOCHS)):
            for _ in range(STEPS):
                action = self.policy_in_state((x, y)) if random.uniform(0, 1) > self.epsilon else random.choice(ACTIONS_ALLOWED)
                buffer.append(action)
                for _ in range(10):
                    action_buff = random.choice(buffer)
                    state_prime = transition((x, y), action_buff)
                    action_taken = ACTIONS_ALLOWED.index(action_buff)
                    r = REWARDS[state_prime[0], state_prime[1]]

                    estimated_Q[x, y, action_taken] = LEARNING_RATE * (r + self.gamma * np.max(estimated_Q[state_prime[0], state_prime[1], :])) + \
                                                    (1 - LEARNING_RATE) * estimated_Q[x, y, action_taken]
                    
                    x, y = state_prime
                
                self.estimated_Q = estimated_Q
                self.estimated_policy = self.compute_policy()

            estimated_j_mu = self.function_j(5000)
            infinite_norm[epoch] = np.max(np.abs(j_mu - estimated_j_mu))
            infinite_norm_2[epoch] = np.max(np.abs(mdp_Q - estimated_Q))

        if GAMMA == 0.9:
            plt.figure()
            plt.plot(range(EPOCHS), infinite_norm)
            plt.xlabel("Epochs")
            plt.ylabel("Infinite norm")
            plt.title("Convergence of the expected return for the 3rd method\n with behaviour: " + ("stochastic " if self.stochastic_behaviour else "deterministic ") + "and " + r"$\gamma =$" + str(GAMMA_3))
            plt.savefig("figures/third_online_learning_" + ("stochastic_" if self.stochastic_behaviour else "deterministic_") + "gamma_" + str(GAMMA_3) + ".png")
            plt.close()

        if GAMMA == 0.4:
            plt.figure()
            plt.plot(range(EPOCHS), infinite_norm_2)
            plt.xlabel("Epochs")
            plt.ylabel("Infinite norm")
            plt.title(r"Convergence of $\hat{Q}$ " + "for the 3rd method\n with behaviour: " + ("stochastic " if self.stochastic_behaviour else "deterministic ") + "and " + r"$\gamma =$" + str(GAMMA_3))
            plt.savefig("figures/third_online_learning_" + ("stochastic_" if self.stochastic_behaviour else "deterministic_") + "gamma_" + str(GAMMA_3) + ".png")
            plt.close()

def main():
    deter = Q_Learning(False, TRAJECTORY_LENGTH, INITIAL_STATE, EPSILON, GAMMA)
    print("\nPolicy for deterministic domain:\n", translater_tuple_action(deter.estimated_policy_offline))
    #print(pd.DataFrame(translater_tuple_action(deter.estimated_policy)).to_latex())
    print("\nExpected return for deterministic domain:\n", deter.function_j(5000))
    #print(pd.DataFrame(deter.function_j(5000)).to_latex())

    deter.first_online_learning()
    deter.second_online_learning()
    deter.third_online_learning()

    stocha = Q_Learning(True, TRAJECTORY_LENGTH, INITIAL_STATE, EPSILON, GAMMA)
    print("\nPolicy for stochastic domain:\n", translater_tuple_action(stocha.estimated_policy_offline))
    #print(pd.DataFrame(translater_tuple_action(stocha.estimated_policy)).to_latex())
    print("\nExpected return for stochastic domain:\n", stocha.function_j(5000))
    #print(pd.DataFrame(stocha.function_j(5000)).to_latex())

    stocha.first_online_learning()
    stocha.second_online_learning()
    stocha.third_online_learning()

if __name__ == "__main__":
    main()