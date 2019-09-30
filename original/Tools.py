import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.shape[1])):
        running_add = running_add * 0.99 + r[0, t]
        discounted_r[0, t] = running_add
    return discounted_r