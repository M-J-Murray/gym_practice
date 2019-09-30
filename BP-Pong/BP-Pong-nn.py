import random
import gym
import torch
from math import sqrt
import numpy as np
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import RMSprop
import torch.multiprocessing as mp

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return Variable(torch.from_numpy(np.array(I, dtype=">f").reshape(1,1,80,80)).cuda())

def chooseAction(f):
    th = random.uniform(0, 1)
    runSum = 0
    for i in range(f.size(1)):
        runSum += f.data[0,i]
        if th < runSum:
            return i

def compileHistory(history):
    history["observation"] = torch.stack(history["observation"])
    history["output"] = torch.stack(history["output"])
    history["reward"] = discount_rewards(torch.stack(history["reward"]))
    history["reward"] -= torch.mean(history["reward"])
    history["reward"] /= torch.std(history["reward"])
    history["dlogp"] = torch.stack(history["dlogp"])

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = torch.zeros_like(r).cuda()
    running_add = 0
    for t in reversed(range(r.size(0))):
        if r.data[t, 0] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * 0.99 + r.data[t, 0]
        discounted_r.data[t, 0] = running_add
    return discounted_r

def run(env, model, optim, results):
    while True:
        observation = env.reset()

        prev_x = None
        history = {"observation": [], "output": [], "dlogp": [], "reward": []}
        done = False

        model.eval()

        while not done:
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else Variable(torch.zeros(1,1,80,80).cuda())
            prev_x = cur_x

            output = model(x)
            action = chooseAction(output)

            history["observation"].append(x)
            history["output"].append(output)
            history["dlogp"].append(torch.log(output[0,action]))

            observation, r, done, info = env.step(action+2)

            history["reward"].append(Variable(torch.FloatTensor([r]).cuda()))
        
        reward = torch.sum(torch.stack(history["reward"])).data[0]
        results.put(reward)

        model.train()

        compileHistory(history)
        updateModel(model, optim, history)
        

def updateModel(model, optim, history):
    optim.zero_grad()
    loss = -torch.sum(history["reward"]*history["dlogp"])
    loss.backward()
    optim.step

def main():
    mp.set_start_method('spawn')
    workers = 2
    learning_rate = 1e-4
    model = Model(2)
    model.cuda()
    model.share_memory()
    optimizer = RMSprop(model.parameters(), lr=learning_rate)
    envs = []
    for i in range(workers):
        envs.append(gym.make("Pong-v0"))

    episode = 0
    best_score = None
    running_reward = None

    results = mp.Queue()
    processes = []

    #Start workers
    for i in range(workers):
        process = mp.Process(target=run, args=(envs[i], model, optimizer, results,))
        process.start()
        processes.append(process)

    while True:
        episode += 1

        reward = results.get()

        if best_score is None:
            best_score = reward
        elif reward > best_score:
            best_score = reward
        running_reward = reward if running_reward is None else running_reward * 0.99 + reward * 0.01
        if episode % 1 == 0:
            print("episode {:4.0f} complete - average reward = {:3.0f}, last score was = {:3.0f}, best score is = {:3.0f}".format(episode,
                                                                                                        running_reward,
                                                                                                        reward,
                                                                                                        best_score))

class Model(nn.Module):

    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU())
        self.fc = nn.Linear(24*7*7, num_classes)
        self.softmax = nn.Softmax(1)

    def forward(self, out):
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.fc(out.view(out.size(0), -1))
        return self.softmax(out)


if __name__ == '__main__':
    main()
