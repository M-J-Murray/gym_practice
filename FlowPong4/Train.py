import gym
from FlowPong4.Model import Model
import FlowPong4.WorkerUtils as wu
from FlowPong4.Statistics import Statistics
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline


def train(learning_rate):
    seed = 1
    tf.set_random_seed(seed)
    np.random.seed(seed)
    stats = Statistics()

    optim = tf.train.AdamOptimizer(learning_rate)
    model = Model(optim, action_classes=2)

    jobs = {"worker": ["localhost:2222"]}
    cluster = tf.train.ClusterSpec(jobs)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="worker", task_index=0, config=config)
    with tf.train.MonitoredTrainingSession(master=server.target) as sess:
        with tf.contrib.tfprof.ProfileContext('/train_dir') as pctx:
            profiler = tf.profiler.Profiler(sess.graph)
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

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

                    action_out = sess.run(model.action_out_sym,
                                          feed_dict={model.observations_sym: x},
                                          options=options,
                                          run_metadata=run_metadata)
                    profiler.add_step(1, run_metadata)

                    action_hot = wu.choose_action(action_out)

                    history["actions"].append(action_hot)

                    frame, r, done, info = env.step(np.argmax(action_hot) + 2)

                    total_reward += r

                    history["rewards"].append(r)

                    if done:
                        profiler.advise(options)
                        tf.profiler.advise(sess.graph, run_meta=run_metadata)

                        # fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                        # chrome_trace = fetched_timeline.generate_chrome_trace_format()
                        # with open('timeline_01.json', 'w') as f:
                        #     f.write(chrome_trace)
                        exit(1)
                        #wu.train(sess, model, history)
                        #stats.update(total_reward)


# spawn must be called inside main
if __name__ == '__main__':
    train(learning_rate=1e-3)

