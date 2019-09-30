import numpy as np


def circle_ds(points, radius):
    # generating labelled training data
    N = points
    X = (np.random.rand(2, N) * 10)-(radius*1.66)
    y = (np.sqrt(np.sum(np.multiply(X, X), axis=0)) < radius).reshape(1, N)
    return X, y


def spiral_ds(points, classes):
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