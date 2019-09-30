import gym
import numpy as np
from Agent import Agent

from History import History

env = gym.make('CartPole-v0')

seed = 50
learning_rate = 1e-4
input_space = 4
height = 160
np.random.seed(seed)
env.seed(seed)

agent = Agent(height, input_space, learning_rate)

running_reward = None
episode = 0
best_score = None

while True:
    history = History()
    observation = env.reset()
    episode += 1
    sum_reward = 0
    done = False
    # Run game
    while not done:
        if episode % 100 == 0:
            env.render()
        prev_observation = observation
        action, f = agent.determine_action(observation)
        observation, reward, done, info = env.step(action)
        history.append(prev_observation, f[0][0], action, reward)
        sum_reward += reward

    history.compile()
    agent.update_weights(history)

    if best_score is None:
        best_score = sum_reward
    elif sum_reward > best_score:
        best_score = sum_reward
    running_reward = sum_reward if running_reward is None else running_reward * 0.99 + sum_reward * 0.01
    if episode % 100 == 0:
        print "episode {:4.0f} complete - average reward = {:3.0f}, best score is = {:3.0f}"\
            .format(episode, running_reward, best_score)

