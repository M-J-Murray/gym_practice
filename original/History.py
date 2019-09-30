import numpy as np
from Tools import discount_rewards


class History:
    def __init__(self):
        self.ep_X = []
        self.ep_f = []
        self.ep_y = []
        self.ep_r = []

    def append(self, observation, f, action, reward):
        self.ep_X.append(observation)
        self.ep_f.append(f)
        self.ep_y.append(action)
        self.ep_r.append(reward)

    def compile(self):
        self.ep_X = np.column_stack(self.ep_X)
        self.ep_f = np.column_stack(self.ep_f)
        self.ep_y = np.column_stack(self.ep_y)
        self.ep_r = np.column_stack(self.ep_r)

        discounted_rewards = discount_rewards(self.ep_r)
        discounted_rewards -= np.mean(discounted_rewards)
        self.ep_r /= np.std(discounted_rewards)
