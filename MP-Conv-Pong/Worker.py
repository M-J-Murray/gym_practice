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
    return Variable(torch.cuda.FloatTensor(np.array(I, dtype="uint8").reshape(1,1,42,64)))

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
    if r.data[t] != 0: running_add = 0
    running_add = running_add * gamma + r.data[t]
    discounted_r.data[t] = running_add
  return discounted_r

def compileHistory(history):
    #history["dlogp"] = torch.cat(history["dlogp"])
    history["output"] = torch.cat(history["output"])
    history["action"] = torch.cat(history["action"])
    rewards = discount_rewards(torch.cat(history["reward"]))
    history["reward"] = (rewards - rewards.mean())/rewards.std()
    return history
    

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
                #history = {"dlogp":[], "reward": []}
                history = {"output":[], "action":[], "reward": []}
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

                    output = self.model(x)
                    action = chooseAction(F.softmax(Variable(output.data, volatile=True), dim=1))

                    #history["dlogp"].append(-torch.log(output[0,action]))
                    history["output"].append(output)
                    history["action"].append(Variable(torch.cuda.LongTensor(1).fill_(action)))

                    observation, r, done, info = self.env.step(action+2)

                    epoch_reward += r
                    history["reward"].append(Variable(torch.cuda.FloatTensor(1).fill_(r)))

                self.reward_queue.put(epoch_reward)
                history = compileHistory(history)
                self.optimizer.zero_grad()
                loss = torch.sum(history["reward"]*self.criterion(history["output"],history["action"]))
                #loss = torch.sum(history["reward"]*history["dlogp"])
                loss.backward()
                self.optimizer.step()
                
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
    while True:
        print(queue.get())