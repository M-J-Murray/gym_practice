from multiprocessing import Value, Lock


class Statistics(object):

    def __init__(self):
        self.lock = Lock()
        self.episode = Value("i", 0)
        self.has_run = Value("i", 0)
        self.best = Value("d", 0)
        self.running = Value("d", 0)

    def update(self, reward):
        with self.lock:
            self.episode.value += 1
            if self.has_run.value == 0:
                self.best.value = reward
            elif reward > self.best.value:
                self.best.value = reward

            self.running.value = reward if self.has_run.value == 0 else self.running.value * 0.99 + reward * 0.01

            self.has_run.value = 1
            print("episode {:5.0f} complete (new:{:5.0f}, avg:{:5.0f}, best:{:5.0f})".format(self.episode.value, reward, self.running.value, self.best.value))
