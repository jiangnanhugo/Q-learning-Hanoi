import numpy as np
import pandas as pd


def learn_Q(R, gamma=0.8, alpha=1.0, N_episodes=1000):
    Q = np.zeros(R.shape)
    states = list(range(R.shape[0]))
    for n in range(N_episodes):
        state = np.random.choice(states)  # Randomly select initial state
        next_states = np.where(R[state, :] >= 0)[0]  # Generate a list of possible next states
        next_state = np.random.choice(next_states)  # Randomly select next state from the list of possible next states
        # Update Q-values
        Q[state, next_state] = (1 - alpha) * Q[state, next_state] + \
                               alpha * (R[state, next_state] + gamma * np.max(Q[next_state, :]))
    if np.max(Q) > 0:
        Q /= np.max(Q)  # Normalize Q to its maximum value
    return Q


def get_policy(Q, R):
    Q_allowed = pd.DataFrame(Q)[pd.DataFrame(R) >= 0].values
    policy = []
    for i in range(Q_allowed.shape[0]):
        row = Q_allowed[i, :]
        sorted_vals = np.sort(row)
        sorted_vals = sorted_vals[~np.isnan(sorted_vals)][::-1]
        sorted_args = row.argsort()[np.where(~np.isnan(sorted_vals))][::-1]
        max_args = [sorted_args[i] for i, val in enumerate(sorted_vals) if val == sorted_vals[0]]
        policy.append(max_args)
    return policy