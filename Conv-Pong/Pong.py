import gym
import torch
import random
import numpy as np
from Model import Model
from torch.autograd import Variable
import torch.nn as nn


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return Variable(torch.from_numpy(np.array(I, dtype=">f")).cuda())


def chooseAction(f):
    th = random.uniform(0, 1)
    runSum = 0
    for i in range(f.size()[0]):
        runSum += f.data[i]
        if th < runSum:
            return i


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = Variable(torch.zeros_like(r.data))
    running_add = 0
    for t in reversed(range(r.size(1))):
        if r.data[0, t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * 0.99 + r.data[0, t]
        discounted_r.data[0, t] = running_add
    return discounted_r


env = gym.make("Pong-v0")

seed = 1
random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
observation = prepro(env.reset())

eta = 1e-5
prev_x = None
history = {"X": [], "g": [], "f": [], "dlogp": [], "r": []}
running_reward = None
episode = 0
best_score = None

model = Model()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(model.parameters())


while True:


    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(80, 80)
    prev_x = cur_x

    f = model.forward(x)

    action = chooseAction(f)

    history["x"].append(x)
    history["f"].append(f)
    history["g"].append(g)
    history["dlogp"].append(f[action] - 1)

    X, r, done, info = env.step(action + 2)
    X = prepro(X)

    history["r"].append(Variable(torch.from_numpy([r])))