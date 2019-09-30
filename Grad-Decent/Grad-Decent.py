import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]


#np.random.seed(50)
learning_rate = 0.00005
input_space = 2
height = 10

W1 = np.random.randn(height, input_space)
b1 = np.zeros((height, 1))
W2 = np.random.randn(1, height)
b2 = 0

# generating labelled training data
N = 1000
X = (np.random.rand(2, N) * 10)-5
y = (np.sqrt(np.sum(np.multiply(X, X), axis=0)) < 3).reshape(1, N)

for i in range(1, 1000):

    g = np.dot(W1, X) + b1
    f = sigmoid(np.dot(W2, np.maximum(g, 0)) + b2)

    E = np.sum(-(np.multiply(y, np.log(f)) + np.multiply(1 - y, np.log(1 - f))))
    print(E)

    # Learning
    d = -(np.subtract(y, f))

    # -ve log - likelihood gradients
    D_W2 = np.dot(d, np.maximum(g, 0).T)
    D_b2 = np.sum(d, axis=1)
    D_W1 = np.dot(np.multiply(d, np.multiply(g > 0, W2.T)), X.T)
    D_b1 = np.sum(np.multiply(d, np.multiply(g > 0, W2.T)), axis=1)

    # update step
    W1 = W1 - learning_rate * D_W1
    b1 = b1 - learning_rate * D_b1[:, None]
    W2 = W2 - learning_rate * D_W2
    b2 = b2 - learning_rate * D_b2


