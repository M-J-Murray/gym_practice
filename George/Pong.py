#!/home/michael/anaconda3/envs/AIGym/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 14:59:35 2018

@author: vogiatzg
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.distributions import Categorical
import torch.utils.data as td
import gym
import numpy as np
#import datalogger
import multiprocessing

NUM_GAMES_PER_UPDATE = 50
NUM_EPOCHS = 10
USE_CUDA = True

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector
"""
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float)

class RunningAverage:
    def __init__(self):
        self.n=0
        self.tot=0

    def add(self,x):
        self.n += 1
        self.tot += x

    def __call__(self):
        if self.n>0:
            return self.tot/self.n
        else:
            return float('NaN')
if USE_CUDA:
    dtype = torch.cuda.FloatTensor
    dtype_L = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_L = torch.LongTensor


class PolicyNet(nn.Module):
    def __init__(self, input_size=(80,80), act_dim = 3):
        super(PolicyNet, self).__init__()
        self.affine1 = nn.Linear(input_size[0]*input_size[1], 200)
        self.affine2 = nn.Linear(200, act_dim)


    def forward(self, input):
        output = input.view(input.size(0),-1)
        output = F.relu(self.affine1(output))
        output = self.affine2(output)
        return output
        # return output.view(-1, 1).squeeze(1)


#class PolicyNet(nn.Module):
#    def __init__(self, input_size=(80,80), act_dim = 6):
#        super(PolicyNet, self).__init__()
#        self.main = nn.Sequential(
#            # input is (nc) x 32 x 32
#            nn.Conv2d(1, 32, 4, 2, 1),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf) x 16 x 16
#            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*2) x 8 x 8
#            nn.Conv2d(64, 32, 4, 2, 1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#
#            nn.Conv2d(32, 1, 4, 2, 1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*4) x 4 x 4
#        )
##       conduct small experiment to find out desired dimensions :-)
#        out =
#self.main(Variable(torch.randn(1,1,input_size[0],input_size[1])))
#        out = out.view(out.size(0),-1)
#        dim = out.size(1)
#        print(dim)
#        self.linear = nn.Linear(dim, act_dim)
#
#
#    def forward(self, input):
#        output = self.main(input)
#        output = output.view(output.size(0),-1)
#        output = self.linear(output)
#
#        return output.squeeze()
#        # return output.view(-1, 1).squeeze(1)

def sampleFromModel(model, x):
    x = x.squeeze().unsqueeze(0)
    pr = F.softmax(model(Variable(x)),dim=1)
    pr = pr.cpu().data.squeeze().numpy()
    pr = 0.9*pr/pr.sum() + 0.1/len(pr)
    return np.random.choice(len(pr), p=pr)

def discount_rewards(r, gamma = 0.99):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


def generate_data(env, model, n_games = 10):
    obs=[]
    acts=[]
    rwds=[]
    for g in range(n_games):
        done=False
        observation = env.reset()
        prev_x = None
        while not done:
    #        env.render()
            cur_x = prepro(observation)
            x = cur_x - prev_x if prev_x is not None else np.zeros_like(cur_x)
            prev_x = cur_x
            x = torch.from_numpy(x).float().unsqueeze(0).type(dtype)
            action = sampleFromModel(model, x)
            observation, reward, done, info = env.step(action+1)
            obs.append(x)
            acts.append(action)
            rwds.append(reward)
        print('\rPlaying pong games: [%d/%d]'%(g, n_games),
end='')
    obs_tensor = torch.FloatTensor(np.stack(obs))
    acts_tensor = torch.FloatTensor(np.stack(acts))
    rwds_tensor = torch.FloatTensor(discount_rewards(np.stack(rwds)))
    rwds_tensor = (rwds_tensor - rwds_tensor.mean())/rwds_tensor.std()
    target_tensor = torch.stack((acts_tensor, rwds_tensor),dim=1)
    wins = sum(1 if r>0 else 0 for r in rwds)
    losses = sum(1 if r<0 else 0 for r in rwds)
    print("\nwins=%d out of %d points played (%0.1f%%)"%
(wins,wins+losses, wins/(wins+losses)*100))
    return td.TensorDataset(obs_tensor, target_tensor)

def policygradtrain(model, loss, optimizer, dataset, num_epochs=10,
test_set_ratio = 0.2, experiment_name='Policy gradients training'):
    N = len(dataset)
    V = int(N * test_set_ratio)
    test_sampler = td.sampler.RandomSampler(range(V))
    train_sampler = td.sampler.RandomSampler(range(V,N))
    dataloader_test = td.DataLoader(dataset, sampler=test_sampler,
batch_size=64, num_workers=2)
    dataloader_train = td.DataLoader(dataset, sampler=train_sampler ,
batch_size=64, num_workers=2)

    #dl = datalogger.DataLogger(experiment_name)

    for epoch in range(num_epochs):
        train_loss = RunningAverage()
        test_loss = RunningAverage()
        for i,(x,t) in enumerate(dataloader_train):
            x = x.unsqueeze(1).type(dtype)
            a = t[:,0].type(dtype_L) # action
            r = t[:,1].type(dtype) # reward

            model.zero_grad()
            error = torch.mean(Variable(r)*loss(model(Variable(x)), Variable(a)))
            error.backward()
            optimizer.step()
            train_loss.add(error.data[0])
            print('\rTraining: [%d/%d] [%0.2f%%] loss=%0.1f'%(epoch,
num_epochs, 100.0*(i+1)/len(dataloader_train), error.data[0]), end='')
        print()
        if len(dataloader_test)>0:
            for i,(x,t) in enumerate(dataloader_test):
                x = x.unsqueeze(1).type(dtype)
                a = t[:,0].type(dtype_L) # action
                r = t[:,1].type(dtype) # reward
                error = torch.mean(Variable(r)*loss(polnet(Variable(x)), Variable(a)))
                error.backward()
                test_loss.add(error.data[0])
                print('\rTesting: [%d/%d] [%0.2f%%] loss=%0.1f'%(epoch, num_epochs, 100.0*(i+1)/len(dataloader_test), error.data[0]), end='')
        print('\nEpoch summary: [%d/%d] train_loss=%0.3f test_loss=%0.3f' % (epoch,num_epochs,train_loss(),test_loss()))
        #dl.log('train_loss',train_loss())
#        dl.log('test_loss',test_loss())
        #dl.plot('Epoch','loss')


polnet = PolicyNet((80,80))
if USE_CUDA:
    polnet.cuda()
optimizer = optim.RMSprop(polnet.parameters(), lr = 1e-3,
weight_decay=0.99)
loss = nn.CrossEntropyLoss(reduce = False)

mgr = multiprocessing.Manager()
res_obs = mgr.list()
res_tar = mgr.list()

env = gym.make('Pong-v0')

for k in range(800):
    dataset = generate_data(env, polnet, n_games=10)
    policygradtrain(polnet, loss, optimizer, dataset, num_epochs=10,
experiment_name='Pong PG training', test_set_ratio=0.0)