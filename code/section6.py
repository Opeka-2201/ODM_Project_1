import numpy as np
import matplotlib.pyplot as plt



# Initialize the estimates
br = np.zeros((n_states, n_actions))
bp = np.zeros((n_states, n_actions, n_states))
bQ = np.zeros((n_states, n_actions))
counts = np.zeros((n_states, n_actions))

# Generate a trajectory using a random uniform policy
policy = np.ones((n_states, n_actions)) / n_actions
T = 1000
trajectory = generate_trajectory(policy, T)

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

# Calculate Jbμ*N and Jμ*N for each state
Jbmu_star_N = np.sum(bQ[np.arange(n_states), bmu_star])
Jmu_star_N = np.sum(true_Q()[np.arange(n_states), bmu_star])

# Display the results
print("bQ:\n", bQ)
print("bmu_star:\n", bmu_star)
print("Jbmu_star_N:\n", Jbmu_star_N)
print("Jmu_star_N:\n", Jmu_star_N)

# Plot the convergence of bp and br
plt.figure()
plt.plot(np.linalg.norm(bp - true_p(), ord=np.inf, axis=2))
plt.plot(np.linalg.norm(br - true_r(), ord=np.inf, axis=1))
plt.legend(["bp", "br"])
plt.xlabel("Time step")
plt.ylabel("Infinite norm")
plt.show()