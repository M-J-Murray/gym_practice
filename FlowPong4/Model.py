import tensorflow as tf


def init_weights(name, dims, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):
    return tf.get_variable(name, trainable=True, shape=dims, initializer=initializer, regularizer=regularizer)


def init_conv(name, in_channels, out_channels, k=5):
    return init_weights(name, [k, k, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())


def conv2d(inputs, weights, strides=2):
    x = tf.nn.conv2d(inputs, weights, strides=[1, 1, strides, strides], padding='VALID', data_format="NCHW")
    return tf.nn.relu(x)


def maxpool2d(inputs, k=2):
    return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def tail(inputs, weights):
    inputs = tf.reshape(inputs, [-1, weights.get_shape().as_list()[0]])
    return tf.matmul(inputs, weights)


class Model(object):

    def __init__(self, optim, observation_shape=(1, 42, 64), action_classes=2, criterion=tf.losses.softmax_cross_entropy):
        self.conv1 = init_conv("conv1", 1, 12)
        self.conv2 = init_conv("conv2", 12, 24)
        self.conv3 = init_conv("conv3", 24, 24)
        self.fc = init_weights("fc", [24*2*5, action_classes])

        self.observations_sym = tf.placeholder(tf.float32, [1, *observation_shape])
        self.actions_sym = tf.placeholder(tf.float32, [1, action_classes])
        self.rewards_sym = tf.placeholder(tf.float32, [1])

        self.action_out_sym = tf.nn.softmax(self.eval(self.observations_sym), axis=1)
        self.train_step = optim.minimize(self.loss(criterion))

    # Applies forward propagation to the inputs
    def eval(self, inputs):
        out = conv2d(inputs, self.conv1)
        out = conv2d(out, self.conv2)
        out = conv2d(out, self.conv3)
        out = tail(out, self.fc)
        return out

    def loss(self, criterion):
        action_out = self.eval(self.observations_sym)
        loss = criterion(logits=action_out, onehot_labels=self.actions_sym, reduction=tf.losses.Reduction.NONE)
        loss += tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.rewards_sym * loss)
