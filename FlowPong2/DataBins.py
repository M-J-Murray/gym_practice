from pathlib import Path
import numpy as np
import os


def delete_old_bins(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(e)


def generate_bins(path, worker_count):
    workers_bins = []
    for i in range(worker_count):
        full_path = str(Path(path + "/worker%d.bin" % i).absolute())
        bin_file = os.open(full_path, os.O_CREAT | os.O_RDWR)
        workers_bins.append({"file": bin_file, "size": 0})
    return workers_bins


class DataBins(object):

    def __init__(self, path, worker_count, input_metadata=((1, 42, 64, 1), "uint8"), action_metadata=((1, 2), "uint8"), reward_metadata=((1,), "float64")):
        delete_old_bins(path)
        self.path = path
        self.workers_bins = generate_bins(path, worker_count)
        self.input_metadata = input_metadata
        self.action_metadata = action_metadata
        self.reward_metadata = reward_metadata
        self.input_bytes = len(np.zeros(*input_metadata).tobytes())
        self.action_bytes = len(np.zeros(*action_metadata).tobytes())
        self.reward_bytes = len(np.zeros(*reward_metadata).tobytes())
        self.line_bytes = self.input_bytes + self.action_bytes + self.reward_bytes

    def insert(self, worker_no, observation, action, reward):
        data = observation.tobytes()
        if len(data) != self.input_bytes:
            raise IOError("Invalid observation inserted")
        data += action.tobytes()
        if len(data) != self.input_bytes+self.action_bytes:
            raise IOError("Invalid action inserted")
        data += reward.tobytes()
        if len(data) != self.input_bytes + self.action_bytes + self.reward_bytes:
            raise IOError("Invalid reward inserted")

        data_bin = self.workers_bins[worker_no]
        os.write(data_bin["file"], data)
        data_bin["size"] += 1

    def parse_data(self, data_bytes):
        observation = np.frombuffer(data_bytes[0:self.input_bytes], dtype=self.input_metadata[1]).reshape(self.input_metadata[0])
        action = np.frombuffer(data_bytes[self.input_bytes:self.input_bytes + self.action_bytes], dtype=self.action_metadata[1]).reshape(self.action_metadata[0])
        reward = np.frombuffer(data_bytes[self.input_bytes + self.action_bytes:], dtype=self.reward_metadata[1])
        return observation, action, reward

    def empty_bin(self, worker_no):
        data_bin = self.workers_bins[worker_no]
        offset = 0
        os.lseek(data_bin["file"], offset, 0)

        all_observations = np.zeros(shape=[data_bin["size"], *self.input_metadata[0][1:]], dtype=self.input_metadata[1])
        all_actions = np.zeros(shape=[data_bin["size"], *self.action_metadata[0][1:]], dtype=self.action_metadata[1])
        all_rewards = np.zeros(shape=[data_bin["size"]], dtype=self.reward_metadata[1])

        for i in range(data_bin["size"]):
            data_bytes = os.pread(data_bin["file"], self.line_bytes, offset)
            observation, action, reward = self.parse_data(data_bytes)
            all_observations[i] = observation
            all_actions[i] = action
            all_rewards[i] = reward

            offset += self.line_bytes

        data_bin["size"] = 0
        os.ftruncate(data_bin["file"], 0)
        return all_observations, all_actions, all_rewards

    def close(self):
        for data_bin in self.workers_bins:
            os.close(data_bin["file"])