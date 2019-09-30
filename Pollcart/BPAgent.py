import numpy as np


def softmax(L2A):
    exp_scores = np.exp(L2A)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def policy_forward(network, observation):
    L1A = np.dot(network['W1'], observation) + network['b1']
    L1A[L1A < 0] = 0
    L2A = np.dot(network['W2'], L1A) + network['b2']
    return L1A, L2A


def policy_backward(network, history):
    num_examples = history.size()
    dL2A = history.probs
    dL2A[history.actions.T, range(num_examples)] -= 1
    dL2A /= num_examples
    dL2A *= history.rewards

    dW2 = np.dot(dL2A, history.L1As.T)
    db2 = np.sum(dL2A, axis=1, keepdims=True)

    dL1A = np.dot(network['W2'].T, dL2A)
    dL1A[history.L1As <= 0] = 0

    dW1 = np.dot(dL1A, history.observations.T)
    db1 = np.sum(dL1A, axis=1, keepdims=True)

    return {'W1': dW1, 'b1': db1.flatten(), 'W2': dW2, 'b2': db2.flatten()}


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * 0.99 + r[0, t]
        discounted_r[0, t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


def select_action(probs):
    prob = np.random.uniform(0, 1)
    prob_sum = 0
    for i in range(0, len(probs)):
        prob_sum += probs[i]
        if prob <= prob_sum:
            return i


class BPAgent(object):

    def __init__(self, height, dimensionality, actions, learning_rate, seed):
        np.random.seed(seed)
        self.learning_rate = learning_rate
        self.network = {
            "W1": np.random.randn(height, dimensionality) / np.sqrt(dimensionality),
            "b1": np.zeros(height),
            "W2": np.random.randn(actions, height),
            "b2": np.zeros(actions)}

    def determine_action(self, observation):
        L1A, L2A = policy_forward(self.network, observation)
        probs = softmax(L2A)
        action = select_action(probs)
        return action, L1A, L2A, probs

    def update_params(self, history):
        history.stack()
        history.rewards = discount_rewards(history.rewards)
        grad = policy_backward(self.network, history)
        for k, v in self.network.iteritems():
            self.network[k] -= self.learning_rate * grad[k]
