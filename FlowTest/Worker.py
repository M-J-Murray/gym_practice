import tensorflow as tf
import numpy as np
import time


def run(worker_count, worker_no, cluster):

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % worker_no, cluster=cluster)):
        with tf.variable_scope("global"):
            gx_sym = tf.placeholder(tf.float32, shape=(worker_count,))
            gvar = tf.get_variable("test", shape=(worker_count,), initializer=tf.initializers.zeros)
            gupdate = gvar.assign_add(gx_sym)

    with tf.variable_scope("worker%d" % worker_no):
        lvar = tf.get_variable("test", shape=(worker_count,), initializer=tf.initializers.zeros)
        lupdate = lvar.assign(gvar)

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="worker", task_index=worker_no, config=config)
    with tf.train.MonitoredTrainingSession(master=server.target) as sess:
        print("Started Worker Updates")
        while True:
            x = np.zeros(worker_count, dtype="float32")
            x[worker_no] += 1
            sess.run(gupdate, feed_dict={gx_sym: x})
            gvar_val = sess.run(gvar)
            sess.run(lupdate)
            lvar_val = sess.run(lvar)
            print("worker%d" % worker_no, gvar_val, lvar_val)
            time.sleep(1)
