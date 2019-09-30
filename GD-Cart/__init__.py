import gym
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.shape[1])):
        running_add = running_add * 0.99 + r[0, t]
        discounted_r[0, t] = running_add
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)
    return discounted_r


def forward_prop(x, network):
    L1A = np.maximum(np.dot(network['W1'], x)[:, None] + network['b1'], 0)
    L2A = sigmoid(np.dot(network['W2'], L1A) + network['b2'])
    return L1A, L2A


def back_prop(history, network):
    d = -(history['ep_y']/history['ep_r'] - history['ep_f'])
    ep_g = np.maximum(np.dot(network['W1'], history['ep_X']) + network['b1'], 0)
    D_W2 = np.dot(d, ep_g.T)
    D_b2 = np.sum(d)
    D_W1 = np.dot(np.multiply(d, np.multiply(ep_g > 0, network['W2'].T)), history['ep_X'].T)
    D_b1 = np.sum(np.multiply(d, np.multiply(ep_g > 0, network['W2'].T)), axis=1)

    return {'W1': D_W1, 'b1': D_b1[:, None], 'W2': D_W2, 'b2': D_b2}


env = gym.make('CartPole-v0')
observation = env.reset()
# seed = 50
# np.random.seed(seed)
# env.seed(seed)

learning_rate = 1e-4
input_space = 4
height = 160

network = {
    'W1': np.random.randn(height, input_space),
    'b1': np.zeros((height, 1)),
    'W2': np.random.randn(1, height),
    'b2': 0}

running_reward = None
episode = 0
best_score = None

while True:
    episode += 1

    history = {
        'ep_X': [],
        'ep_f': [],
        'ep_y': [],
        'ep_r': []}

    done = False
    # Run game
    while not done:



        history['ep_X'].append(observation)
        g, f = forward_prop(observation, network)
        history['ep_f'].append(f[0][0])

        if f[0][0] > np.random.uniform():
            action = 0
        else:
            action = 1
        history['ep_y'].append(action)

        observation, reward, done, info = env.step(action)

        history['ep_r'].append(reward)

    sum_reward = np.sum(history['ep_r'])
    if best_score is None:
        best_score = sum_reward
    elif sum_reward > best_score:
        best_score = sum_reward
    running_reward = sum_reward if running_reward is None else running_reward * 0.99 + sum_reward * 0.01
    if episode % 100 == 0:
        print "episode {:4.0f} complete - average reward = {:3.0f}, best score is = {:3.0f}".format(episode, running_reward, best_score)

    # Arrays to matrices
    history['ep_X'] = np.column_stack(history['ep_X'])
    history['ep_f'] = np.column_stack(history['ep_f'])
    history['ep_y'] = np.column_stack(history['ep_y'])
    history['ep_r'] = np.column_stack(history['ep_r'])

    # Adjust rewards
    history['ep_r'] = discount_rewards(history['ep_r'])
    history['ep_r'] -= np.mean(history['ep_r'])
    history['ep_r'] /= np.std(history['ep_r'])

    grad = back_prop(history, network)

    # update step
    for k, v in network.iteritems():
        network[k] -= learning_rate * grad[k]

    observation = env.reset()
