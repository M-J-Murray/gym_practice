import FlowTest.Worker as Worker
from multiprocessing import Process
import tensorflow as tf


# Check stats.log for results
def train(worker_count):
    tasks = ["localhost:"+str(2223+i) for i in range(worker_count)]
    jobs = {"ps": ['localhost:2222'], "worker": tasks}
    cluster = tf.train.ClusterSpec(jobs)

    worker_procs = [Process(target=Worker.run, args=[worker_count, i, cluster]) for i in range(worker_count)]
    [proc.start() for proc in worker_procs]


    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    server = tf.train.Server(cluster, job_name="ps", task_index=0, config=config)
    server.join()


# spawn must be called inside main
if __name__ == '__main__':
    train(worker_count=4)
