import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
from torch.nn import Softmax
from Datasets import circle_ds


def sigmoid(x):
    return 1.0 / (1.0 + torch.exp(-x))  # sigmoid "squashing" function to interval [0,1]


# def softmax(f):
#     exp_f = torch.exp(f)
#     return exp_f / exp_f.sum(0)


def asNumpy(f):
    return f.cpu().data.numpy()


def isNan(f):
    return np.isnan(asNumpy(f.sum())[0])

torch.manual_seed(2)
learning_rate = 1e-5
input_space = 2
output_space = 2
height = 20

W1 = Variable(torch.randn(height, input_space).cuda(), requires_grad=True)
R1 = Variable(torch.zeros(height, input_space).cuda(), requires_grad=True)
V1 = Variable(torch.zeros(height, 1).cuda(), requires_grad=True)
H1 = Variable(torch.zeros(input_space, 1).cuda(), requires_grad=True)
W2 = Variable(torch.randn(output_space, height).cuda(), requires_grad=True)
R2 = Variable(torch.zeros(output_space, height).cuda(), requires_grad=True)
V2 = Variable(torch.zeros(output_space, 1).cuda(), requires_grad=True)
H2 = Variable(torch.zeros(height, 1).cuda(), requires_grad=True)

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
plt.colorbar(im)
plt.ion()
plt.show()

softmax = Softmax(0)

for i in range(1, 1000):

    g = (V1 + torch.mm(W1, X) + torch.mm(R1, (X + H1)*(X + H1))).clamp(min=0)
    f = softmax(V2 + torch.mm(W2, g) + torch.mm(R2, (g + H2)*(g + H2)))

    corect_logprobs = 1-f[y.cpu().data.numpy(), range(N)]
    data_loss = torch.sum(corect_logprobs)

    data_loss.backward()

    #learning_rate -= 1e-7

    W1.data -= learning_rate * W1.grad.data
    R1.data -= learning_rate * R1.grad.data
    V1.data -= learning_rate * V1.grad.data
    H1.data -= learning_rate * H1.grad.data
    W2.data -= learning_rate * W2.grad.data
    R2.data -= learning_rate * R2.grad.data
    V2.data -= learning_rate * V2.grad.data
    H2.data -= learning_rate * H2.grad.data

    W1.grad.data.zero_()
    R1.grad.data.zero_()
    V1.grad.data.zero_()
    H1.grad.data.zero_()
    W2.grad.data.zero_()
    R2.grad.data.zero_()
    V2.grad.data.zero_()
    H2.grad.data.zero_()

    if i%1==0:
        gg = (V1 + torch.mm(W1, XX) + torch.mm(R1, (XX + H1)*(XX + H1))).clamp(min=0)
        ff = softmax(V2 + torch.mm(W2, gg) + torch.mm(R2, (gg + H2)*(gg + H2)))
        grid = np.reshape(ff.cpu().data.numpy()[1,range(2500)], [50, 50])
        im.set_data(grid)
        plt.pause(0.0001)
