import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def generate_data(points, classes):
    N = points  # number of points per class
    D = 2  # dimensionality
    K = classes  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros((N * K, K), dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix, j] = 1

    return X, y


learning_rate = 1e-1
input_space = 2
height = 100

# generating labelled training data
N = 1000
K = 3
X = tf.placeholder(tf.float32, [N*K, 2])
Y = tf.placeholder(tf.float32, [N*K, 3])

layer_1 = tf.layers.Dense(height)
layer_2 = tf.layers.Dense(K)

f = layer_2(tf.nn.relu(layer_1(X)))

# compute the loss: average cross-entropy loss and regularization
loss_sym = tf.losses.softmax_cross_entropy(logits=f, onehot_labels=Y)
train = tf.train.AdamOptimizer(learning_rate).minimize(loss_sym)

scale = 1
detail = 100
XX = tf.placeholder(tf.float32, [detail**2, 2])

ff_sym = tf.nn.softmax(layer_2(tf.nn.relu(layer_1(XX))))

fig = plt.figure()
data = np.random.rand(detail, detail, 3)
im = plt.imshow(data)
plt.colorbar(im)
plt.ion()
plt.show()

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    batch_x, batch_y = generate_data(N, K)
    [xx, yy] = np.meshgrid(np.linspace(-scale, scale, detail), np.linspace(-scale, scale, detail))
    batch_XX = np.stack([xx.flatten(), yy.flatten()], axis=1)

    for i in range(1, 2000):
        loss = sess.run(train, feed_dict={X: batch_x, Y: batch_y})
        loss = sess.run(loss_sym, feed_dict={X: batch_x, Y: batch_y})
        print("Step " + str(i) + ", Loss= {:.4f}".format(loss))

        if i % 100 == 0:
            ff = sess.run(ff_sym, feed_dict={XX: batch_XX})
            ff_map = np.reshape(ff, [detail, detail, 3])
            im.set_data(ff_map)
            plt.pause(0.001)
