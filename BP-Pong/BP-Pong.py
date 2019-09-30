import random
from math import sqrt
import gym
import torch
from torch.autograd import Variable
import numpy as np


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return Variable(torch.from_numpy(np.array(I, dtype=">f").reshape(6400)).cuda())


def softmax(f):
    exp_f = torch.exp(f)
    return exp_f / exp_f.sum(0)


def chooseAction(f):
    th = random.uniform(0, 1)
    runSum = 0
    for i in range(f.size()[0]):
        runSum += f.data[i]
        if th < runSum:
            return i


def compileHistory(history):
    history["X"] = torch.stack(history["X"], 1)
    history["g"] = torch.stack(history["g"], 1)
    history["f"] = torch.stack(history["f"], 1)
    history["dlogp"] = torch.stack(history["dlogp"], 1)
    history["r"] = discount_rewards(torch.stack(history["r"], 1))
    return history


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = torch.zeros_like(r).cuda()
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
observation = env.reset()

eta = 1e-4
reg = 1e-3
D = 80*80
decay_rate = 0.99

layerDims = {
    "H1": 200,
    "H2": 2,
}

network = {
    "W1": Variable(torch.randn(layerDims["H1"], D).cuda() / sqrt(D), requires_grad=True),
    "b1": Variable(torch.zeros(layerDims["H1"]).cuda(), requires_grad=True),
    "W2": Variable(torch.randn(layerDims["H2"], layerDims["H1"]).cuda() / sqrt(layerDims["H1"]), requires_grad=True),
    "b2": Variable(torch.zeros(layerDims["H2"]).cuda(), requires_grad=True),
}
grad_buffer = {k: torch.zeros_like(v).cuda() for k, v in network.items()} # update buffers that add up gradients over a batch
rmsprop_cache = {k: torch.zeros_like(v).cuda() for k, v in network.items()} # rmsprop memory

history = {"X": [], "g": [], "f": [], "dlogp": [], "r": []}  # type: dict

running_reward = None
episode = 0
best_score = None
prev_x = None

while True:
    # env.render()
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else Variable(torch.zeros(D).cuda())
    prev_x = cur_x

    g = (torch.mv(network["W1"], x) + network["b1"]).clamp(min=0)
    f = softmax(torch.mv(network["W2"], g) + network["b2"])
    action = chooseAction(f)

    history["X"].append(x)
    history["f"].append(f)
    history["g"].append(g)
    history["dlogp"].append(-torch.log(f[action]))

    observation, r, done, info = env.step(action+2)

    history["r"].append(Variable(torch.FloatTensor([r]).cuda()))

    if done:
        episode += 1

        sum_reward = torch.sum(torch.stack(history["r"])).data[0]
        if best_score is None:
            best_score = sum_reward
        elif sum_reward > best_score:
            best_score = sum_reward
        running_reward = sum_reward if running_reward is None else running_reward * 0.9 + sum_reward * 0.1
        if episode % 1 == 0:
            print("episode {:4.0f} complete - average reward = {:3.0f}, last score was = {:3.0f}, best score is = {:3.0f}".format(episode,
                                                                                                        running_reward,
                                                                                                        sum_reward,
                                                                                                        best_score))
        history = compileHistory(history)

        data_loss = torch.sum(history["dlogp"])/history["f"].size(1)
        reg_loss = 0.5 * reg * torch.sum(network["W1"] * network["W1"]) + 0.5 * reg * torch.sum(network["W2"] * network["W2"])
        loss = data_loss + reg_loss

        loss.backward()

        for k, v in network.items():
            rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * network[k].grad ** 2
            network[k].data -= eta * network[k].grad.data / (torch.sqrt(rmsprop_cache[k].data) + 1e-5)
            grad_buffer[k] = torch.zeros_like(v).cuda()
            network[k].grad.data.zero_()

        X = env.reset()
        prev_x = None
        history = {"X": [], "g": [], "f": [], "dlogp": [], "r": []}
