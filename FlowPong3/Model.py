import tensorflow as tf


def init_weights(name, dims, initializer=tf.contrib.layers.xavier_initializer(), regularizer=tf.contrib.layers.l2_regularizer(scale=1e-3)):
    return tf.get_variable(name, trainable=True, shape=dims, initializer=initializer, regularizer=regularizer)


def init_conv(name, in_channels, out_channels, k=5):
    return init_weights(name, [k, k, in_channels, out_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())


def conv2d(inputs, weights, strides=2):
    x = tf.nn.conv2d(inputs, weights, strides=[1, strides, strides, 1], padding='VALID')
    return tf.nn.relu(x)


def maxpool2d(inputs, k=2):
    return tf.nn.max_pool(inputs, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def tail(inputs, weights):
    inputs = tf.reshape(inputs, [-1, weights.get_shape().as_list()[0]])
    return tf.matmul(inputs, weights)


class Model(object):

    def __init__(self, optim, action_classes=2, criterion=tf.losses.softmax_cross_entropy, batch_size=128):
        self.conv1 = init_conv("conv1", 1, 12)
        self.conv2 = init_conv("conv2", 12, 24)
        self.conv3 = init_conv("conv3", 24, 24)
        self.fc = init_weights("fc", [24*2*5, action_classes])

        self.observations_sym = tf.placeholder(tf.float32, [None, 42, 64, 1])
        self.actions_sym = tf.placeholder(tf.float32, [None, action_classes])
        self.rewards_sym = tf.placeholder(tf.float32, [None])

        self.action_out_sym = tf.nn.softmax(self.eval(self.observations_sym), axis=1)
        self.global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(-1), trainable=False)
        self.zero_ops, self.accum_ops, self.train_step = self.train(optim, criterion)

        self.batch_size = batch_size
        self.buffer_size = tf.placeholder(dtype=tf.int64, shape=())
        dataset = tf.data.Dataset.from_tensor_slices((self.observations_sym, self.actions_sym, self.rewards_sym))
        dataset = dataset.shuffle(buffer_size=self.buffer_size)
        dataset = dataset.batch(batch_size=batch_size)
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    # Applies forward propagation to the inputs
    def eval(self, inputs):
        out = conv2d(inputs, self.conv1)
        out = conv2d(out, self.conv2)
        out = conv2d(out, self.conv3)
        out = tail(out, self.fc)
        return out

    def train(self, optim, criterion):
        grads, lvs = zip(*optim.compute_gradients(self.loss(criterion), var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))
        grads_accum = [tf.Variable(tf.zeros_like(lv.initialized_value())) for lv in lvs]
        zero_ops = [grad_accum.assign(tf.zeros_like(grad_accum)) for grad_accum in grads_accum]
        accum_ops = [grads_accum[i].assign_add(grad) for i, grad in enumerate(grads)]
        train_step = optim.apply_gradients(zip(grads_accum, lvs), self.global_step)
        return zero_ops, accum_ops, train_step

    def loss(self, criterion):
        action_out = self.eval(self.observations_sym)
        loss = criterion(logits=action_out, onehot_labels=self.actions_sym, reduction=tf.losses.Reduction.NONE)
        loss += tf.losses.get_regularization_loss()
        return tf.reduce_sum(self.rewards_sym * loss)
