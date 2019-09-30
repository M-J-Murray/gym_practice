import gym
import numpy as np

def softmax(f):
    exp_f = np.exp(f)
    return exp_f / np.sum(exp_f, axis=0, keepdims=True)

def policy_forward(network, observation):
    g = np.maximum(0, np.dot(network['W1'], observation) + network['b1'])
    f = softmax(np.dot(network['W2'], g) + network['b2'])
    return g, f

def policy_backward(network, history, reg):
    dW2 = np.dot(history['ep_dlogp'], history['ep_g'].T)
    db2 = np.sum(history['ep_dlogp'], axis=1, keepdims=True)

    dg = np.dot(network['W2'].T, history['ep_dlogp'])
    dg[history['ep_g'] <= 0] = 0

    dW1 = np.dot(dg, history['ep_X'].T)
    db1 = np.sum(dg, axis=1, keepdims=True)

    # Reg
    dW1 += reg * network['W1']
    dW2 += reg * network['W2']

    return {'W1': dW1, 'b1': db1.flatten(), 'W2': dW2, 'b2': db2.flatten()}

def discount_rewards(r, gamma):
    discounted_r = np.zeros_like(r)
    running_add = np.sum(r)
    for t in range(0, r.shape[1]):
        running_add = running_add * gamma
        discounted_r[0, t] = running_add
    discounted_r -= np.percentile(discounted_r, 70)
    discounted_r /= np.std(discounted_r)
    return discounted_r

env = gym.make('CartPole-v0')
observation = env.reset()

learning_rate = 1e-5
decay_rate = 0.99
reg = 1e-3
gamma = 0.99
input_space = 4
height = 70
actions = 2

network = {
    'W1': np.random.randn(height, input_space) / np.sqrt(input_space),
    'b1': np.zeros(height),
    'W2': np.random.randn(actions, height) / np.sqrt(height),
    'b2': np.zeros(actions)}
grad_buffer = {k: np.zeros_like(v) for k, v in network.items()}  # update buffers that add up gradients over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in network.items()}

running_reward = None
episode = 0
best_score = None

history = {
        'ep_X': [],
        'ep_f': [],
        'ep_g': [],
        'ep_dlogp': [],
        'ep_r': []}

while episode < 1000000:

    # if episode % 120 == 0:
    #     env.render()

    history['ep_X'].append(observation)
    g, f = policy_forward(network, observation)
    history['ep_g'].append(g)
    history['ep_f'].append(f)

    action = 0 if f[0] > f[1] else 1
    f[action] -= 1
    history['ep_dlogp'].append(f)

    observation, reward, done, info = env.step(action)

    history['ep_r'].append(reward)

    if done:
        episode += 1
        env.render()
        sum_reward = np.sum(history['ep_r'])
        if best_score is None:
            best_score = sum_reward
        elif sum_reward > best_score:
            best_score = sum_reward
        running_reward = sum_reward if running_reward is None else running_reward * 0.99 + sum_reward * 0.01
        if episode % 100 == 0:
            print("episode {:4.0f} complete - average reward = {:3.0f}, best score is = {:3.0f}".format(episode, running_reward, best_score))

        # Arrays to matrices
        history['ep_X'] = np.column_stack(history['ep_X'])
        history['ep_f'] = np.column_stack(history['ep_f'])
        history['ep_g'] = np.column_stack(history['ep_g'])
        history['ep_dlogp'] = np.column_stack(history['ep_dlogp'])
        history['ep_r'] = np.column_stack(history['ep_r'])

        # Adjust rewards
        history['ep_r'] = discount_rewards(history['ep_r'], gamma)
        history['ep_dlogp'] /= history['ep_r'].shape[1]
        history['ep_dlogp'] *= history['ep_r']

        grad = policy_backward(network, history, reg)
        for k in network: grad_buffer[k] += grad[k]
        # update step
        for k, v in network.items():
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * grad_buffer[k] ** 2
            network[k] -= learning_rate * grad[k] / (np.sqrt(rmsprop_cache[k]) + 1e-5)
            grad_buffer[k] = np.zeros_like(v)

        env.reset()
        history = {
            'ep_X': [],
            'ep_f': [],
            'ep_g': [],
            'ep_dlogp': [],
            'ep_r': []}