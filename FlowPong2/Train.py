import gym
from FlowPong2.Model import Model
import FlowPong2.WorkerUtils as wu
from FlowPong2.Statistics import Statistics
from FlowPong2.DataBins import DataBins
import tensorflow as tf
import numpy as np


def train(learning_rate):
    seed = 1
    tf.set_random_seed(seed)
    np.random.seed(seed)

    stats = Statistics()
    data_bins = DataBins("Databins", 1)

    optim = tf.train.AdamOptimizer(learning_rate)
    model = Model(optim, action_classes=2)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("Started Worker Updates")
        env = gym.make("Pong-v0")
        env.seed(seed)

        while True:
            frame = env.reset()

            for i in range(21):
                frame, r, done, info = env.step(0)

            prev_x = wu.prepro(frame)
            frame, r, done, info = env.step(0)

            done = False
            total_reward = 0

            while not done:
                cur_x = wu.prepro(frame)
                x = cur_x - prev_x
                prev_x = cur_x

                action_out = sess.run(model.action_out_sym, feed_dict={model.observations_sym: x})

                action_hot = wu.choose_action(action_out)

                frame, r, done, info = env.step(np.argmax(action_hot) + 2)

                total_reward += r

                data_bins.insert(0, x, action_hot, np.zeros(1) + r)

                if done:
                    wu.train(sess, model, *data_bins.empty_bin(0))
                    stats.update(total_reward)


# spawn must be called inside main
if __name__ == '__main__':
    train(learning_rate=1e-3)
