import gym
import numpy as np

from original import sigmoid, discount_rewards

env = gym.make('CartPole-v0')

seed = 50
np.random.seed(seed)
env.seed(seed)

learning_rate = 1e-5
input_space = 4
height = 200

W1 = np.random.randn(height, input_space)
R1 = np.zeros_like(W1)
b1 = np.zeros((height, 1))
W2 = np.random.randn(1, height)
R2 = np.zeros_like(W2)
b2 = 0

running_reward = None
episode = 0
best_score = None

while True:
    episode += 1
    observation = env.reset()

    ep_X = []
    ep_f = []
    ep_y = []
    ep_rewards = []

    done = False
    # Run game
    while not done:
        #env.render()

        ep_X.append(observation)
        g = np.maximum(np.dot(W1, observation) + np.dot(R1, np.multiply(observation, observation)) + b1, 0)
        f = sigmoid(np.dot(W2, g) + np.dot(R2, np.multiply(g, g)) + b2)
        ep_f.append(f[0][0])

        if f[0][0] > np.random.uniform():
            action = 0
        else:
            action = 1
        ep_y.append(action)

        observation, reward, done, info = env.step(action)

        ep_rewards.append(reward)

    sum_reward = np.sum(ep_rewards)
    if best_score is None:
        best_score = sum_reward
    elif sum_reward > best_score:
        best_score = sum_reward
    running_reward = sum_reward if running_reward is None else running_reward * 0.99 + sum_reward * 0.01
    if episode % 100 == 0:
        print "episode {:4.0f} complete - average reward = {:3.0f}, best score is = {:3.0f}".format(episode, running_reward, best_score)

    # Arrays to matrices
    ep_X = np.column_stack(ep_X)
    ep_f = np.stack(ep_f)
    ep_y = np.stack(ep_y)

    # Adjust rewards
    discounted_epr = discount_rewards(ep_rewards, best_score)
    discounted_epr -= np.mean(discounted_epr)
    discounted_epr /= np.std(discounted_epr)

    # Learning
    d = -(np.multiply(ep_y, (1 - ep_f)) - np.multiply(1 - ep_y, ep_f)) * discounted_epr

    # -ve log - likelihood gradients
    D_g = np.dot(W1, ep_X) + np.dot(R1, np.multiply(ep_X, ep_X)) + b1
    D_g_max = np.maximum(D_g, 0)
    D_W2 = np.dot(d, D_g_max.T)
    D_R2 = np.dot(d, np.multiply(D_g_max, D_g_max).T)
    D_b2 = np.sum(d, axis=0)
    D_W1 = np.dot(np.multiply(d, np.multiply(D_g > 0, W2.T) + np.multiply(D_g > 0, R2.T)), ep_X.T) * 2
    D_R1 = np.dot(np.multiply(d, np.multiply(D_g > 0, W2.T) + np.multiply(D_g > 0, R2.T)), np.multiply(D_g_max, D_g_max).T) * 2
    D_b1 = np.sum(np.multiply(d, np.multiply(D_g > 0, W2.T) + np.multiply(D_g > 0, R2.T)), axis=1)

    # update step
    W1 = W1 - learning_rate * D_W1
    b1 = b1 - learning_rate * D_b1[:, None]
    W2 = W2 - learning_rate * D_W2
    b2 = b2 - learning_rate * D_b2
