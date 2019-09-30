import numpy as np


class History:
    def __init__(self):
        self.observations = []
        self.L1As = []
        self.L2As = []
        self.probs = []
        self.actions = []
        self.rewards = []

    def append(self, observation, L1A, L2A, probs, action, reward):
        self.observations.append(observation)
        self.L1As.append(L1A)
        self.L2As.append(L2A)
        self.probs.append(probs)
        self.actions.append(action)
        self.rewards.append(reward)

    def stack(self):
        self.observations = np.column_stack(self.observations)
        self.L1As = np.column_stack(self.L1As)
        self.L2As = np.column_stack(self.L2As)
        self.probs = np.column_stack(self.probs)
        self.actions = np.column_stack(self.actions)
        self.rewards = np.column_stack(self.rewards)

    def size(self):
        return self.observations.shape[0]
