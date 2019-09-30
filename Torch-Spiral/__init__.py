import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


def softmax(f):
    exp_f = torch.exp(f)
    return exp_f / exp_f.sum(0)

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

#torch.manual_seed(50)
learning_rate = 1e-1
reg = 1e-3
input_space = 2
height = 100

criterion = nn.CrossEntropyLoss()

# generating labelled training data
N = 500
K = 3
X, y = generate_data(N, K)
X = Variable(torch.FloatTensor(X.T))
y = Variable(torch.LongTensor(y.T))


W1 = Variable(torch.randn(height, input_space), requires_grad=True)
b1 = Variable(torch.zeros(height, 1), requires_grad=True)
W2 = Variable(torch.randn(K, height), requires_grad=True)
b2 = Variable(torch.zeros(K, 1), requires_grad=True)

scale = 1
detail = 100
[xx, yy] = np.meshgrid(np.linspace(-scale, scale, detail), np.linspace(-scale, scale, detail))
XX = Variable(torch.FloatTensor(np.stack([xx.flatten(), yy.flatten()])))

fig = plt.figure()
data = np.random.rand(detail, detail, 3)
im = plt.imshow(data)
plt.colorbar(im)
plt.ion()
plt.show()

for i in range(1, 2000):

    # Forward
    g = (W1.mm(X) + b1).clamp(min=0)
    f = W2.mm(g) + b2

    # compute the loss: average cross-entropy loss and regularization
    #corect_logprobs = -torch.log(f[y, range(N*K)])
    #data_loss = corect_logprobs.sum(0) / N
    data_loss = criterion(f.transpose(1,0), y)
    #reg_loss = 0.5 * reg * torch.sum(W1 * W1) + 0.5 * reg * torch.sum(W2 * W2)
    #loss = data_loss + reg_loss

    data_loss.backward()

    W1.data -= learning_rate * W1.grad.data
    b1.data -= learning_rate * b1.grad.data
    W2.data -= learning_rate * W2.grad.data
    b2.data -= learning_rate * b2.grad.data

    W1.grad.data.zero_()
    b1.grad.data.zero_()
    W2.grad.data.zero_()
    b2.grad.data.zero_()

    if i % 100 == 0:
        gg = (W1.mm(XX) + b1).clamp(min=0)
        ff = softmax(W2.mm(gg) + b2)
        ff_map = np.transpose(np.reshape(ff.data.numpy(), [3, detail, detail]), (1,2,0))
        im.set_data(ff_map)
        plt.pause(0.0001)
        print("iteration %d: loss %f" % (i, data_loss))
