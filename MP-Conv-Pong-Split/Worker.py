#!/home/michael/anaconda3/envs/AIGym/bin/python3.6
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
from torch.autograd import Variable
import numpy as np
import torch.utils.data as td

import torch.nn.functional as F
import traceback
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from Model import Model
import gym

def prepro(I):
    I = I[32:198, 16:144]  # crop
    I = I[::4, ::2, 0]  # downsample by factor of 4
    return torch.cuda.FloatTensor(np.array(I, dtype="uint8").reshape(1,1,42,64))

def chooseAction(f):
    th = torch.cuda.FloatTensor(1).uniform_() 
    runSum = torch.cuda.FloatTensor(1).fill_(0)
    for i in range(f.size(1)):
        runSum += f.data[0,i]
        if th[0] < runSum[0]:
            break
    return i

def discount_rewards(r, gamma = 0.99):
  discounted_r = torch.zeros_like(r)
  running_add = 0
  for t in reversed(range(r.size(0))):
    if r[t] != 0: running_add = 0
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def compileHistory(history):
    history["observation"] = torch.cat(history["observation"])
    history["action"] = torch.cat(history["action"])
    rewards = discount_rewards(torch.cat(history["reward"]))
    history["reward"] = (rewards - rewards.mean())/rewards.std()
    return history

def train(model, criterion, optimizer, history):
    output = model(Variable(history["observation"].cuda()))
    actions = Variable(history["action"].cuda())
    rewards = Variable(history["reward"].cuda())
    optimizer.zero_grad()
    loss = torch.sum(rewards*criterion(output,actions))
    loss.backward()
    optimizer.step()

class Worker(mp.Process):

    def __init__(self, env, model, criterion, optimizer, reward_queue, name):
        super(Worker, self).__init__()
        self.env = env
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.reward_queue = reward_queue
        self.name = name

    def run(self):
        try:
            while True:
                history = {"observation":[], "action": [], "reward": []}
                done = False
                observation = self.env.reset()
                for i in range(22):
                    if i == 21:
                        prev_x = prepro(observation)
                    observation, r, done, info = self.env.step(0)
                
                epoch_reward = 0
                while not done:
                    #self.env.render()
                    cur_x = prepro(observation)
                    x = cur_x - prev_x
                    prev_x = cur_x

                    output = self.model(Variable(x, volatile=True))
                    action = chooseAction(F.softmax(output, dim=1))

                    history["action"].append(torch.LongTensor(1).fill_(action))

                    observation, r, done, info = self.env.step(action+2)

                    epoch_reward += r
                    history["observation"].append(x.cpu())
                    history["reward"].append(torch.FloatTensor(1).fill_(r))

                self.reward_queue.put(epoch_reward)
                history = compileHistory(history)
                train(self.model, self.criterion, self.optimizer, history)
                
        except Exception as identifier:
            self.reward_queue.put(identifier)
            self.reward_queue.put(traceback.format_exc())
            

if __name__ == '__main__':
    mp.set_start_method('spawn')
    learning_rate = 1e-3
    model = Model(2)
    model.cuda()
    model.share_memory()
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    env = gym.make("Pong-v0")
    queue = SimpleQueue()
    worker = Worker(env, model, criterion, optimizer, queue, "test")
    worker.run()
    print(queue.get())