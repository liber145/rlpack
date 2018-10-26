import random
import numpy as np
from collections import deque, defaultdict
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
        # return map(np.array, zip(*samples))
        return samples

    def clear(self):
        self.mem = []


class Memory2(object):
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def append(self, transition):
        self.mem.append(transition)

    def extend(self, transitions):
        for t in transitions:
            self.append(t)

    def sample(self, batch_size):
        samples = random.sample(self.mem, batch_size)
        return map(np.array, zip(*samples))


class Memory3(object):
    def __init__(self, capacity, n_step=1):
        self.n_step = n_step
        self.traj = defaultdict(list)
        self.mem = deque(maxlen=capacity)

    def append(self, transition):
        pass

    def extend(self, states, actions, rewards, next_states, dones):
        """Store SARSD tuples.

        Args:
            states: current states.
            actions: current actions.
            rewards: received rewards.
            next_states: next states.
            dones: dones.
        """
        for i in range(actions.shape[0]):
            self.traj[i].append((states[i, :], actions[i], rewards[i], next_states[i, :], dones[i]))

            if dones[i] is True or len(self.traj[i]) == self.n_step:
                self.mem.append(self.traj.pop(i))

    def sample(self, batch_size):
        samples = random.sample(self.mem, batch_size)
        return samples

    def clear(self):
        """Empty memory."""
        self.mem.clear()


class Memory4(object):
    def __init__(self, capacity):
        self.mem = deque(maxlen=capacity)

    def put(self, data):
        self.mem.append(data)

    def sample(self, batch_size):
        """均匀采样。"""
        assert batch_size <= len(self.mem), "No enough data to sample"
        samples = random.sample(self.mem, batch_size)

        # 将(s,a,r,s,d)拆成5个矩阵发送出去。
        return map(np.array, zip(*samples))

    def get_full(self):
        return map(np.array, zip(*self.mem))

    def clear(self):
        self.mem.clear()
