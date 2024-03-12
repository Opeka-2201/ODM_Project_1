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
ACTIONS_ALLOWED = [(1, 0), (-1, 0), (0, 1), (0, -1)]
START_STATE = (3, 0)
GAMMA = 0.99
NB_STEPS = 10


class Agent:
    def __init__(self, actions_allowed=ACTIONS_ALLOWED):
        self.actions_allowed = actions_allowed

    def chose_action(self):
        return random.choice(self.actions_allowed)


class Domain:
    """
    Represents a domain for a reinforcement learning problem.

    Args:
        rewards (numpy.ndarray): The rewards matrix for the domain.
        start_state (tuple): The starting state for the agent.
        gamma (float): The discount factor for future rewards.
        bool_stochastic (bool): Whether the domain is stochastic or not.
        prob_stochastic (float): The probability of a stochastic action.

    Attributes:
        rewards (numpy.ndarray): The rewards matrix for the domain.
        state (tuple): The current state of the agent.
        gamma (float): The discount factor for future rewards.
        bool_stochastic (bool): Whether the domain is stochastic or not.
        nb_lines (int): The number of lines in the rewards matrix.
        nb_columns (int): The number of columns in the rewards matrix.
        prob_stochastic (float): The probability of a stochastic action.
    """

    def __init__(self, rewards=REWARDS, start_state=START_STATE, gamma=GAMMA, bool_stochastic=False, prob_stochastic=0.5):
        self.rewards = rewards
        self.state = start_state
        self.default_start = start_state
        self.gamma = gamma
        self.bool_stochastic = bool_stochastic
        self.nb_lines, self.nb_columns = rewards.shape
        self.prob_stochastic = prob_stochastic

    def reward(self, visited):
        """
        Returns the reward for a given state.

        Args:
            visited (tuple): The state for which to get the reward.

        Returns:
            float: The reward for the given state.
        """
        return self.rewards[visited[0], visited[1]]

    def step(self, action):
        """
        Takes a step in the domain.

        Args:
            action (tuple or list): The action to take.

        Returns:
            tuple: A tuple containing the previous state, the action taken, the reward received, and the new state.
        """
        state = self.state

        if self.bool_stochastic and random.random() > self.prob_stochastic:
            # If w < 1/2 the agent will be teleported to (0, 0)
            visited = (0, 0)
        else:
            visited = self.dynamic(state, action)

        reward = self.reward(visited)
        self.state = visited

        return (state, action, reward, visited)

    def estimated_Q(self, N):
        """
        Computes the Q-values iteratively from N = 1 to N.

        Args:
            N (int): The number of iterations.

        Returns:
            Q (ndarray): The Q-values.

        """
        Q = np.zeros((self.nb_lines, self.nb_columns, 4), dtype=float)
        for n in range(1, N+1):
            for i in range(self.nb_lines):
                for j in range(self.nb_columns):
                    for k in range(4):
                        action = ACTIONS_ALLOWED[k]
                        state = (i, j)
                        state_prime = self.dynamic(state, action)
                        reward = self.reward(state_prime)
                        Q[i, j, k] = reward + self.gamma * max(Q[state_prime[0], state_prime[1]])
        return Q

    def function_j(self, policy, N = 1000):
        # Computation of the expected return of the rule-based policy starting from N = 0
        lines = self.nb_lines
        columns = self.nb_columns

        print("Computing J for domain:", "stochastic" if self.bool_stochastic else "deterministic")
        J_run = np.zeros((lines, columns))
        for _ in range(N):
            J_new = np.zeros((lines, columns))
            
            for i in range(lines):
                for j in range(columns):
                    state = self.dynamic((i,j), ACTIONS_ALLOWED[int(policy[i,j])], lines, columns)
                    if self.bool_stochastic:
                        J_new[i,j] = self.prob_stochastic * (self.reward(state) + self.gamma * J_run[state[0], state[1]]) + \
                                        (1 - self.prob_stochastic) * (self.reward((0,0)) + self.gamma * J_run[0,0])
                    else:
                        J_new[i,j] = self.reward(state) + self.gamma * J_run[state[0], state[1]]

            J_run = J_new
        return J_run


    @staticmethod
    def dynamic(state, action, nb_lines=5, nb_columns=5):
        """
        Computes the new state based on the current state and action.

        Args:
            state (tuple): The current state.
            action (tuple or list): The action to take.
            nb_lines (int): The number of lines in the domain.
            nb_columns (int): The number of columns in the domain.

        Returns:
            tuple or list: The new state(s) after taking the action.
        """
        if not isinstance(action, list):
            return (min(max(state[0] + action[0], 0), nb_lines - 1), min(max(state[1] + action[1], 0), nb_columns - 1))
        else:
            to_return = list()
            actions = action
            for action in actions:
                to_return.append(min(max(state[0] + action[0], 0), nb_lines - 1), min(
                    max(state[1] + action[1], 0), nb_columns - 1))
            return to_return


class MDP:
    """
    Markov Decision Process (MDP) class for computing policies iteratively.

    Attributes:
        domain (Domain): The domain of the MDP.
        nb_lines (int): The number of lines in the domain.
        nb_columns (int): The number of columns in the domain.
        stochastic_behavior (bool): Flag indicating whether the MDP has stochastic behavior.
        prob_stochastic (float): The probability of stochastic behavior.
        policy (ndarray): The policy of the MDP.
        Q (ndarray): The Q-values of the MDP.

    Methods:
        __init__(self, domain: Domain): Initializes the MDP with the given domain.
        compute_policies_iteratively(self, N): Computes the policies iteratively starting from N = 1.
        compute_Q(self, N): Computes the Q-values iteratively from N = 1 to N.
        reward_function(self, temp, Q_u, Q_d, Q_r, Q_l, state, action): Computes the reward function for a given state and action using the Q-values.
        r_s_a(self, state, action): Computes the expected reward for a given state and action depending on the stochastic behavior.
        p_sp_s_a(self, state_prime, state, action): Computes the probability of reaching a state_prime from state using action depending on the stochastic behavior.
        compute_policy(self): Computes the policy from the Q-values by fetching the action with the highest Q-value.
        policy_in_state(self, state): Returns the action in a given state to follow the policy.
        function_j(self, N): Computes the expected return for a given policy.

    """

    def __init__(self, domain: Domain):
        """
        Initializes the MDP with the given domain.

        Args:
            domain (Domain): The domain of the MDP.

        """
        self.domain = domain
        self.nb_lines, self.nb_columns = domain.nb_lines, domain.nb_columns
        self.stochastic_behavior = domain.bool_stochastic
        self.prob_stochastic = domain.prob_stochastic
        self.policy = None
        self.Q = np.zeros((self.nb_lines, self.nb_columns, 4),
                          dtype=float)  # [u, d, r, l]

    def compute_policies_iteratively(self, N):
        """
        Computes the policies iteratively starting from N = 1.

        Args:
            N (int): The number of iterations.

        """
        current = None

        for n in range(1, N+1):
            previous = current
            # update Q values
            self.Q = self.compute_Q(n)
            # update policy
            self.compute_policy()
            # swap + check convergence
            current = self.policy
            print("Iteration:", n, " \tEqual policies:",
                  np.array_equal(previous, current))

    def compute_Q(self, N):
        """
        Computes the Q-values iteratively from N = 1 to N.

        Args:
            N (int): The number of iterations.

        Returns:
            Q (ndarray): The Q-values.

        """
        Q_u = np.zeros((self.nb_lines, self.nb_columns), dtype=float)
        Q_d = np.zeros((self.nb_lines, self.nb_columns), dtype=float)
        Q_r = np.zeros((self.nb_lines, self.nb_columns), dtype=float)
        Q_l = np.zeros((self.nb_lines, self.nb_columns), dtype=float)

        # compute the Q values iteratively from N = 1 to N
        for _ in range(N):
            # temporary Q values for swapping later
            temp_u = np.zeros((self.nb_lines, self.nb_columns), dtype=float)
            temp_d = np.zeros((self.nb_lines, self.nb_columns), dtype=float)
            temp_r = np.zeros((self.nb_lines, self.nb_columns), dtype=float)
            temp_l = np.zeros((self.nb_lines, self.nb_columns), dtype=float)

            for i in range(self.nb_lines):
                for j in range(self.nb_columns):
                    # compute the Q values for each action using the reward function
                    temp_u = self.reward_function(
                        temp_u, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[0])
                    temp_d = self.reward_function(
                        temp_d, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[1])
                    temp_r = self.reward_function(
                        temp_r, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[2])
                    temp_l = self.reward_function(
                        temp_l, Q_u, Q_d, Q_r, Q_l, (i, j), ACTIONS_ALLOWED[3])

            # Swapping
            Q_u = temp_u
            Q_d = temp_d
            Q_r = temp_r
            Q_l = temp_l

        Q = np.zeros((self.nb_lines, self.nb_columns, 4), dtype=float)
        Q[:, :, 0] = Q_u
        Q[:, :, 1] = Q_d
        Q[:, :, 2] = Q_r
        Q[:, :, 3] = Q_l
        return Q

    def reward_function(self, temp, Q_u, Q_d, Q_r, Q_l, state, action):
        """
        Computes the reward function for a given state and action using the Q-values.

        Args:
            temp (ndarray): The temporary Q-values.
            Q_u (ndarray): The Q-values for the action 'up'.
            Q_d (ndarray): The Q-values for the action 'down'.
            Q_r (ndarray): The Q-values for the action 'right'.
            Q_l (ndarray): The Q-values for the action 'left'.
            state (tuple): The current state.
            action (str): The current action.

        Returns:
            temp (ndarray): The updated temporary Q-values.

        """
        # Probabilities and expected rewards
        state_prime = self.domain.dynamic(state, action)
        r_s_a = self.r_s_a(state, action)
        p_sp_s_a = self.p_sp_s_a(state_prime, state, action)

        # Compute the reward function
        reward_function = p_sp_s_a * max(Q_u[state_prime[0], state_prime[1]], Q_d[state_prime[0],
                                         state_prime[1]], Q_r[state_prime[0], state_prime[1]], Q_l[state_prime[0], state_prime[1]])

        # Stochastic behavior
        if p_sp_s_a < 1 and self.stochastic_behavior:
            p_stoch = self.p_sp_s_a((0, 0), state, action)
            reward_function += p_stoch * \
                max(Q_u[0, 0], Q_d[0, 0], Q_r[0, 0], Q_l[0, 0])

        # Update the temporary Q values
        temp[state[0], state[1]] = r_s_a + GAMMA * reward_function
        return temp

    def r_s_a(self, state, action):
        """
        Computes the expected reward for a given state and action depending on the stochastic behavior.

        Args:
            state (tuple): The current state.
            action (str): The current action.

        Returns:
            reward (float): The expected reward.

        """
        state_prime = self.domain.dynamic(state, action)
        if self.stochastic_behavior:
            return self.prob_stochastic * self.domain.rewards[0, 0] + (1 - self.prob_stochastic) * self.domain.rewards[state_prime[0], state_prime[1]]
        else:
            return self.domain.rewards[state_prime[0], state_prime[1]]

    def p_sp_s_a(self, state_prime, state, action):
        """
        Computes the probability of reaching a state_prime from state using action depending on the stochastic behavior.

        Args:
            state_prime (tuple): The next state.
            state (tuple): The current state.
            action (str): The current action.

        Returns:
            probability (float): The probability.

        """
        visited = self.domain.dynamic(state, action)
        if self.stochastic_behavior:
            return self.prob_stochastic * (1 if state_prime == visited else 0) + (1 - self.prob_stochastic) * (1 if state_prime == (0, 0) else 0)
        else:
            return 1 if state_prime == visited else 0

    def compute_policy(self):
        """
        Computes the policy from the Q-values by fetching the action with the highest Q-value.

        """
        self.policy = np.zeros((self.nb_lines, self.nb_columns), dtype=object)
        for i in range(self.nb_lines):
            for j in range(self.nb_columns):
                self.policy[i, j] = ACTIONS_ALLOWED[np.argmax(
                    self.Q[i, j])]

    def policy_in_state(self, state):
        """
        Returns the action in a given state to follow the policy.

        Args:
            state (tuple): The current state.

        Returns:
            action (str): The action to follow.

        """
        if self.policy is None:
            self.compute_policy()
        return self.policy[state[0], state[1]]

    def function_j(self, N):
        """
        Computes the expected return for a given policy.

        Args:
            N (int): The number of iterations.

        Returns:
            J_N (ndarray): The expected return.

        """
        J_N = np.zeros((self.nb_lines, self.nb_columns))
        for _ in range(N):
            temp = np.zeros((self.nb_lines, self.nb_columns))

            for i in range(self.nb_lines):
                for j in range(self.nb_columns):
                    state = (i, j)
                    action = self.policy_in_state(state)
                    state_prime = self.domain.dynamic(state, action)
                    r_s_a = self.r_s_a(state, action)

                    if self.stochastic_behavior:
                        temp[state[0], state[1]] = r_s_a + GAMMA * (
                            self.prob_stochastic * J_N[state_prime] + (1 - self.prob_stochastic) * J_N[0, 0])
                    else:
                        temp[state[0], state[1]] = r_s_a + \
                            GAMMA * J_N[state_prime]
            J_N = temp
        return J_N
    

def generate_trajectory(agent, domain, N):
    trajectory = list()
    state = domain.default_start
    for _ in range(N):
        action = agent.chose_action()
        next_state = domain.dynamic(state, action)
        reward = domain.reward(next_state)
        trajectory.append((state, action, reward, next_state))
        state = next_state
    return trajectory

def traj_to_matrix(trajectory, domain):
    r_hat = np.zeros((domain.nb_lines* domain.nb_columns, len(ACTIONS_ALLOWED)))
    r_hat_count = np.zeros((domain.nb_lines* domain.nb_columns, len(ACTIONS_ALLOWED)))
    p_hat = np.zeros((domain.nb_lines* domain.nb_columns, len(ACTIONS_ALLOWED), domain.nb_lines* domain.nb_columns))

    for (state, action, reward, next_state) in trajectory:
        state_index = state[0] * domain.nb_columns + state[1]
        action_index = ACTIONS_ALLOWED.index(action)
        next_state_index = next_state[0] * domain.nb_columns + next_state[1]
        r_hat[state_index, action_index] += reward
        r_hat_count[state_index, action_index] += 1
        p_hat[state_index, action_index, next_state_index] += 1

    for i in range(domain.nb_lines* domain.nb_columns):
        for j in range(len(ACTIONS_ALLOWED)):
            if np.sum(p_hat[i, j]) > 0:
                p_hat[i, j] /= np.sum(p_hat[i, j])
            else :
                p_hat[i, j] = np.ones((domain.nb_lines* domain.nb_columns)) / (domain.nb_lines* domain.nb_columns)

    # replace 0 with 1 for division
    r_hat_count[r_hat_count == 0] = 1
    r_hat /= r_hat_count
    return r_hat, p_hat

def true_r(mdp):
    r = np.zeros((mdp.nb_lines* mdp.nb_columns, len(ACTIONS_ALLOWED)))
    for i in range(mdp.nb_lines):
        for j in range(mdp.nb_columns):
            state = (i, j)
            for k in range(len(ACTIONS_ALLOWED)):
                action = ACTIONS_ALLOWED[k]
                next_state = mdp.domain.dynamic(state, action)
                r[i*mdp.nb_columns + j, k] = mdp.r_s_a(state, action)
    return r

def true_p(mdp):
    p = np.zeros((mdp.nb_lines* mdp.nb_columns, len(ACTIONS_ALLOWED), mdp.nb_lines* mdp.nb_columns))
    for i in range(mdp.nb_lines):
        for j in range(mdp.nb_columns):
            state = (i, j)
            for k in range(len(ACTIONS_ALLOWED)):
                action = ACTIONS_ALLOWED[k]
                for l in range(mdp.nb_lines):
                    for m in range(mdp.nb_columns):
                        next_state = (l, m)
                        p[i*mdp.nb_columns + j, k, l*mdp.nb_columns + m] = mdp.p_sp_s_a(next_state, state, action)
    return p

def estimate_Q(r_hat, p_hat, gamma, N):
    Q = np.zeros((r_hat.shape[0], r_hat.shape[1]))
    for _ in range(N):
        temp = np.zeros((r_hat.shape[0], r_hat.shape[1]))
        for i in range(r_hat.shape[0]):
            for j in range(r_hat.shape[1]):
                temp[i, j] = r_hat[i, j] + gamma * np.sum(p_hat[i, j] * np.max(Q, axis=1))
        Q = temp
    return Q

def inf_norm(Q1, Q2):
    return np.max(np.abs(Q1 - Q2))

## MAIN ##
def main():
    ag = Agent()
    det_dom = Domain()
    sto_dom = Domain(bool_stochastic=True, prob_stochastic=0.5)
    mdp_det = MDP(det_dom)
    mdp_sto = MDP(sto_dom)

    mdp_det.compute_Q(100)
    mdp_sto.compute_Q(100)


    true_p_det = true_p(mdp_det)
    true_r_det = true_r(mdp_det)
    true_p_sto = true_p(mdp_sto)
    true_r_sto = true_r(mdp_sto)
    true_Q_det = mdp_det.Q.copy()
    true_Q_sto = mdp_sto.Q.copy()

    inf_norm_Q_det = list()
    inf_norm_r_det = list()
    inf_norm_p_det = list()

    inf_norm_Q_sto = list()
    inf_norm_r_sto = list()
    inf_norm_p_sto = list()


    test_N = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
    for N in tqdm(test_N):
        trajectory = generate_trajectory(ag, det_dom, N)
        r_hat, p_hat = traj_to_matrix(trajectory, det_dom)
        Q_hat = estimate_Q(r_hat, p_hat, GAMMA, 100)
        Q_hat = Q_hat.reshape((5, 5, 4))
        inf_norm_Q_det.append(inf_norm(Q_hat, true_Q_det))
        inf_norm_r_det.append(inf_norm(r_hat, true_r_det))
        inf_norm_p_det.append(inf_norm(p_hat, true_p_det))

        trajectory = generate_trajectory(ag, sto_dom, N)
        r_hat, p_hat = traj_to_matrix(trajectory, sto_dom)
        Q_hat = estimate_Q(r_hat, p_hat, GAMMA, 100)
        Q_hat = Q_hat.reshape((5, 5, 4))
        inf_norm_Q_sto.append(inf_norm(Q_hat, true_Q_sto))
        inf_norm_r_sto.append(inf_norm(r_hat, true_r_sto))
        inf_norm_p_sto.append(inf_norm(p_hat, true_p_sto))

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].plot(test_N, inf_norm_Q_det, label="Q")
    axs[0, 0].set_title('Q Deterministic')
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_yscale('log')

    axs[0, 1].plot(test_N, inf_norm_r_det, label="R")
    axs[0, 1].set_title('R Deterministic')
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_yscale('log')

    axs[0, 2].plot(test_N, inf_norm_p_det, label="P")
    axs[0, 2].set_title('P Deterministic')
    axs[0, 2].set_xscale('log')
    axs[0, 2].set_yscale('log')

    axs[1, 0].plot(test_N, inf_norm_Q_sto, label="Q")
    axs[1, 0].set_title('Q Stochastic')
    axs[1, 0].set_xscale('log')
    axs[1, 0].set_yscale('log')

    axs[1, 1].plot(test_N, inf_norm_r_sto, label="R")
    axs[1, 1].set_title('R Stochastic')
    axs[1, 1].set_xscale('log')
    axs[1, 1].set_yscale('log')

    axs[1, 2].plot(test_N, inf_norm_p_sto, label="P")
    axs[1, 2].set_title('P Stochastic')
    axs[1, 2].set_xscale('log')
    axs[1, 2].set_yscale('log')

    for ax in axs.flat:
        ax.legend()

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
