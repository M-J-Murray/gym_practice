import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Datasets import circle_ds


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))  # sigmoid "squashing" function to interval [0,1]


torch.manual_seed(50)
learning_rate = 1e-5
input_space = 2
height = 50

W1 = Variable(torch.randn(height, input_space).cuda(), requires_grad=True)
b1 = Variable(torch.zeros(height, 1).cuda(), requires_grad=True)
W2 = Variable(torch.randn(1, height).cuda(), requires_grad=True)
b2 = Variable(torch.zeros(1, 1).cuda(), requires_grad=True)

# generating labelled training data
N = 1000
dims = 5
X, y = circle_ds(N, dims)

# generating the 2d plot
K = 50
[xx, yy] = np.meshgrid(np.linspace(-dims, dims, K), np.linspace(-dims, dims, K))
XX = Variable(torch.from_numpy(np.vstack([xx.flatten(), yy.flatten()])).type(torch.FloatTensor).cuda())

fig = plt.figure()
data = np.random.rand(50, 50)
im = plt.imshow(data, interpolation='nearest', cmap='plasma')
plt.ion()
plt.show()

for i in xrange(1000):

    g = (torch.mm(W1, X) + b1).clamp(min=0)
    f = sigmoid(torch.mm(W2, g) + b2).clamp(min=1e-5, max=1-1e-5)
    E = torch.sum(-(y*torch.log(f) + (1 - y)*torch.log(1 - f)))
    print E.data[0]

    E.backward()

    W1.data -= W1.grad.data * learning_rate
    b1.data -= b1.grad.data * learning_rate
    W2.data -= W2.grad.data * learning_rate
    b2.data -= b2.grad.data * learning_rate

    W1.grad.data.zero_()
    b1.grad.data.zero_()
    W2.grad.data.zero_()
    b2.grad.data.zero_()

    gg = (torch.mm(W1, XX) + b1).clamp(min=0)
    ff = sigmoid(torch.mm(W2, gg) + b2).clamp(min=1e-5, max=1 - 1e-5)
    grid = np.reshape(ff.cpu().data.numpy(), [50, 50])
    im.set_data(grid)
    plt.pause(0.00001)


