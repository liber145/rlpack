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


class Memory5(object):
    def __init__(self, capacity):
        """
        state_queue 存储交互得到的state，假设state是(84*84*4)，有8个env同时传递，那么每个state存储格式为$(8*84*84*4)$。
        """
        self.state_queue = deque(maxlen=capacity)
        self.action_queue = deque(maxlen=capacity)
        self.reward_queue = deque(maxlen=capacity)
        self.done_queue = deque(maxlen=capacity)

    def store_sard(self, state, action, reward, done):
        self.state_queue.append(state)
        self.action_queue.append(action)
        self.reward_queue.append(reward)
        self.done_queue.append(done)

    def get_last_n_step(self, n):
        assert n < self.size, "No enough sample in memory."

        state_batch = np.asarray([self.state_queue[-i-1] for i in reversed(range(n+1))])
        action_batch = np.asarray([self.action_queue[-i-1] for i in reversed(range(1, n+1))])
        reward_batch = np.asarray([self.reward_queue[-i-1] for i in reversed(range(1, n+1))])
        done_batch = np.asarray([self.done_queue[-i-1] for i in reversed(range(1, n+1))])

        n_env = state_batch.shape[1]
        ob_shape = state_batch.shape[2:]

        assert state_batch.shape == (n+1, n_env, *ob_shape)

        state_batch = state_batch.swapaxes(1, 0)
        action_batch = action_batch.swapaxes(1, 0)
        reward_batch = reward_batch.swapaxes(1, 0)
        done_batch = done_batch.swapaxes(1, 0)

        return state_batch, action_batch, reward_batch, done_batch

    def sample_transition(self, n):
        index = np.random.randint(self.size-1, size=n)
        state_batch = np.asarray([self.state_queue[i] for i in index])
        action_batch = np.asarray([self.action_queue[i] for i in index])
        reward_batch = np.asarray([self.reward_queue[i] for i in index])
        done_batch = np.asarray([self.done_queue[i] for i in index])
        next_state_batch = np.asarray([self.state_queue[i+1] for i in index])

        n_env = state_batch.shape[1]
        ob_shape = state_batch.shape[2:]

        state_batch = state_batch.swapaxes(0, 1).reshape(n_env, n, *ob_shape)
        action_batch = action_batch.swapaxes(0, 1).reshape(n_env, n)
        reward_batch = reward_batch.swapaxes(0, 1).reshape(n_env, n)
        done_batch = done_batch.swapaxes(0, 1).reshape(n_env, n)
        next_state_batch = next_state_batch.swapaxes(0, 1).reshape(n_env, n, *ob_shape)

        return state_batch, action_batch, reward_batch, done_batch, next_state_batch

    @property
    def size(self):
        return len(self.done_queue)
