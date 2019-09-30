import gym
from FlowPong3.Model import Model
import FlowPong3.WorkerUtils as wu
from FlowPong3.Statistics import Statistics
import tensorflow as tf
import numpy as np


def train(learning_rate):
    seed = 1
    tf.set_random_seed(seed)
    np.random.seed(seed)
    stats = Statistics()

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

            history = {"observations": [], "actions": [], "rewards": []}
            done = False
            total_reward = 0

            while not done:
                cur_x = wu.prepro(frame)
                x = cur_x - prev_x
                prev_x = cur_x

                history["observations"].append(x)

                action_out = sess.run(model.action_out_sym, feed_dict={model.observations_sym: x})

                action_hot = wu.choose_action(action_out)

                history["actions"].append(action_hot)

                frame, r, done, info = env.step(np.argmax(action_hot) + 2)

                total_reward += r

                history["rewards"].append(r)

                if done:
                    wu.train(sess, model, history)
                    stats.update(total_reward)


# spawn must be called inside main
if __name__ == '__main__':
    train(learning_rate=1e-3)
