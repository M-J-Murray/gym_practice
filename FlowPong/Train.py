from FlowPong.DataBins import DataBins
from FlowPong.Statistics import Statistics
from FlowPong.Worker import run
from multiprocessing import Process
import tensorflow as tf


def run_ps(cluster):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)
    server.join()


def train(learning_rate, worker_count):
    stats = Statistics()
    data_bins = DataBins("/home/michael/dev/fyp/AIGym/FlowPong/Databins", worker_count)

    tasks = ["localhost:" + str(2223 + i) for i in range(worker_count)]
    jobs = {"ps": ["localhost:2222"], "worker": tasks}
    cluster = tf.train.ClusterSpec(jobs)

    ps_proc = Process(target=run_ps, args=[cluster])
    worker_procs = [Process(target=run, args=[i, learning_rate, cluster, data_bins, stats]) for i in range(worker_count)]

    ps_proc.start()
    [proc.start() for proc in worker_procs]

    ps_proc.join()
    [proc.join() for proc in worker_procs]


# spawn must be called inside main
if __name__ == '__main__':
    train(learning_rate=5e-4, worker_count=8)
