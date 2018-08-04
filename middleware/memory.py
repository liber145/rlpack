import random
import numpy as np
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


class Memory(object):
    """储存SARS对。

    Method:
        append: 添加一个一个的SARS tuple。期望形式 state, action, reward, next_state, done.
    """

    def __init__(self, capacity):
        self.mem = list()
        self.cnt = 0
        self.capacity = capacity

    def append(self, consecutive_transition):
        """consecutive_transition是sarsd对。"""
        if len(self.mem) < self.capacity:
            self.mem.append(consecutive_transition)
        else:
            self.mem[self.cnt % self.capacity] = consecutive_transition
        self.cnt += (self.cnt + 1) % self.capacity

    def sample(self, batch_size):
        samples = random.sample(self.mem, batch_size)
        return map(np.array, zip(*samples))

    def clear(self):
        self.mem = []

