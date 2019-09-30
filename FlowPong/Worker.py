import gym
from FlowPong.Model import Model
import FlowPong.WorkerUtils as wu
import tensorflow as tf
import numpy as np


def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def run(worker_no, learning_rate, cluster, data_bins, stats):
    name = "worker%d" % worker_no
    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % worker_no, cluster=cluster)):
        optim = tf.train.AdamOptimizer(learning_rate)
        Model("global")

    local_model = Model(name, optim=optim)

    update_local_ops = update_target_graph('global', name)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="worker", task_index=worker_no, config=config)
    with tf.train.MonitoredTrainingSession(master=server.target) as sess:
        sess.run(update_local_ops)
        print("Started Worker Updates")
        env = gym.make("Pong-v0")

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

                action_out = sess.run(local_model.action_out_sym, feed_dict={local_model.observations_sym: x})
                action_hot = wu.choose_action(action_out)

                frame, r, done, info = env.step(np.argmax(action_hot) + 2)

                total_reward += r

                data_bins.insert(worker_no, x, action_hot, np.zeros(1)+r)

                if done:
                    wu.train(sess, local_model, *data_bins.empty_bin(worker_no))
                    stats.update(total_reward)
                    sess.run(update_local_ops)
