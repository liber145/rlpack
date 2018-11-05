import random
import numpy as np
from collections import deque

class Memory(object):
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
        act_shape = action_batch.shape[2:]

        state_batch = state_batch.swapaxes(0, 1).reshape(n_env * n, *ob_shape)
        action_batch = action_batch.swapaxes(0, 1).reshape(n_env * n, *act_shape)
        reward_batch = reward_batch.swapaxes(0, 1).reshape(n_env * n)
        done_batch = done_batch.swapaxes(0, 1).reshape(n_env * n)
        next_state_batch = next_state_batch.swapaxes(0, 1).reshape(n_env * n, *ob_shape)

        return state_batch, action_batch, reward_batch, done_batch, next_state_batch

    @property
    def size(self):
        return len(self.done_queue)
