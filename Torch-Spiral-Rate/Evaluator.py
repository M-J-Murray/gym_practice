#!/home/michael/anaconda3/envs/AIGym/bin/python3.6

import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch
from RateModel import RateModel
from BPModel import BPModel
import torch.utils.data as td
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def generate_data(points, classes):
    N = points  # number of points per class
    D = 2  # dimensionality
    K = classes  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y

learning_rate = 1e1
input_space = 2
height = 12
output_space = 3
model = RateModel(input_space, height, output_space)
#model = BPModel(input_space, height, output_space)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# generating labelled training data
N = 1000
K = 3
X, y = generate_data(N, K)
X = Variable(torch.FloatTensor(X))
y = Variable(torch.LongTensor(y))

scale = 1
detail = 100
[xx, yy] = np.meshgrid(np.linspace(-scale, scale, detail), np.linspace(-scale, scale, detail))
XX = Variable(torch.FloatTensor(np.stack([xx.flatten(), yy.flatten()], axis=1)))

fig = plt.figure()
data = np.random.rand(detail, detail, 3)
im = plt.imshow(data)
plt.colorbar(im)
plt.ion()
plt.show()

for j in range(2000):
    model.zero_grad()

    output = model(X)

    loss = criterion(output,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if j % 100 == 0:
        ff = F.softmax(model(XX))
        ff_map = np.reshape(ff.cpu().data.numpy(), [detail, detail, 3])
        im.set_data(ff_map)
        plt.pause(0.0001)
        print(loss.data[0])