import numpy as np
from Tools import sigmoid


class Agent(object):

    def __init__(self, height, input_space, eta):
        self.W1 = np.random.randn(height, input_space)
        self.b1 = np.zeros((height, 1))
        self.W2 = np.random.randn(1, height)
        self.b2 = 0
        self.eta = eta

    def determine_action(self, observation):
        g = np.dot(self.W1, observation) + self.b1
        f = sigmoid(np.dot(self.W2, np.maximum(g, 0)) + self.b2)
        action = 0 if f[0][0] < 1-f[0][0] else 1
        return action, f

    def update_weights(self, history):
        # Learning
        d = -(np.multiply(history.ep_y, (1 - history.ep_f)) - np.multiply(1 - history.ep_y, history.ep_f)) * history.ep_r

        # -ve log - likelihood gradients
        d_W2 = np.dot(d, np.maximum(np.dot(self.W1, history.ep_X) + self.b1, 0).T)
        d_b2 = np.sum(d, axis=1)
        d_W1 = np.dot(np.multiply(d, np.multiply(np.dot(self.W1, history.ep_X) + self.b1 > 0, self.W2.T)), history.ep_X.T)
        d_b1 = np.sum(np.multiply(d, np.multiply(np.dot(self.W1, history.ep_X) + self.b1 > 0, self.W2.T)), axis=1)

        # update weights
        self.W1 -= self.eta * d_W1
        self.b1 -= self.eta * d_b1[:, None]
        self.W2 -= self.eta * d_W2
        self.b2 -= self.eta * d_b2

