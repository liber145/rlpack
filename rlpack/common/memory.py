import pickle
import random
from collections import defaultdict, deque
from typing import List

import numpy as np


class Memory(object):
    def __init__(self, capacity=0):
        """ state_queue 存储交互得到的state，假设state是(84*84*4)，有8个env同时传递，那么每个state存储格式为$(8*84*84*4)$。
        """
        self.state_queue = deque(maxlen=capacity)
        self.action_queue = deque(maxlen=capacity)
        self.reward_queue = deque(maxlen=capacity)
        self.done_queue = deque(maxlen=capacity)

        self.cnt = 0

    def store_sard(self, state, action, reward, done):
        self.state_queue.append(state)
        self.action_queue.append(action)
        self.reward_queue.append(reward)
        self.done_queue.append(done)

        self.cnt += 1
        if self.cnt == 200:
            for j in range(8):
                f = open(f"naive_state{j}.p", "wb")
                pickle.dump([self.state_queue[i][j] for i in range(200)], f)
                f = open(f"naive_reward{j}.p", "wb")
                pickle.dump([self.reward_queue[i][j] for i in range(200)], f)
                f = open(f"naive_done{j}.p", "wb")
                pickle.dump([self.done_queue[i][j] for i in range(200)], f)

    def get_last_n_step(self, n):
        assert n < self.size, "No enough sample in memory."

        state_batch = np.asarray([self.state_queue[-i - 1] for i in reversed(range(n + 1))])
        action_batch = np.asarray([self.action_queue[-i - 1] for i in reversed(range(1, n + 1))])
        reward_batch = np.asarray([self.reward_queue[-i - 1] for i in reversed(range(1, n + 1))])
        done_batch = np.asarray([self.done_queue[-i - 1] for i in reversed(range(1, n + 1))])

        n_env = state_batch.shape[1]
        ob_shape = state_batch.shape[2:]

        assert state_batch.shape == (n + 1, n_env, *ob_shape)

        state_batch = state_batch.swapaxes(1, 0)
        action_batch = action_batch.swapaxes(1, 0)
        reward_batch = reward_batch.swapaxes(1, 0)
        done_batch = done_batch.swapaxes(1, 0)

        return state_batch, action_batch, reward_batch, done_batch

    def sample_transition(self, n):
        index = np.random.randint(self.size - 1, size=n)
        state_batch = np.asarray([self.state_queue[i] for i in index])
        action_batch = np.asarray([self.action_queue[i] for i in index])
        reward_batch = np.asarray([self.reward_queue[i] for i in index])
        done_batch = np.asarray([self.done_queue[i] for i in index])
        next_state_batch = np.asarray([self.state_queue[i + 1] for i in index])

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


class DistributedMemory(object):
    def __init__(self, maxlen: int = 128):
        self.env_wrapper = None
        self.state_queue = defaultdict(lambda: deque(maxlen=maxlen + 1))
        self.action_queue = defaultdict(lambda: deque(maxlen=maxlen))
        self.reward_queue = defaultdict(lambda: deque(maxlen=maxlen))
        self.done_queue = defaultdict(lambda: deque(maxlen=maxlen))

        self.cnt = 0

    def register(self, env_wrapper):
        self.env_wrapper = env_wrapper

    @property
    def _env_ids(self):
        assert self.env_wrapper is not None
        return self.env_wrapper.env_id

    def store_s(self, states):
        for env_id, s in zip(self._env_ids, states):
            self.state_queue[env_id].append(s)

    def store_a(self, actions):
        for env_id, a in zip(self._env_ids, actions):
            self.action_queue[env_id].append(a)

    def store_rds(self, rewards, dones, states):
        for env_id, r, d, s in zip(self._env_ids, rewards, dones, states):
            self.reward_queue[env_id].append(r)
            self.done_queue[env_id].append(d)
            self.state_queue[env_id].append(s)

    def get_last_n_samples(self, n_sample):
        state_batch, action_batch, reward_batch, done_batch = [], [], [], []

        for env_id in self.done_queue.keys():
            state_batch.append(np.asarray([self.state_queue[env_id][-i - 1] for i in reversed(range(n_sample + 1))]))
            action_batch.append(np.asarray([self.action_queue[env_id][-i - 1] for i in reversed(range(n_sample))]))
            reward_batch.append(np.asarray([self.reward_queue[env_id][-i - 1] for i in reversed(range(n_sample))]))
            done_batch.append(np.asarray([self.done_queue[env_id][-i - 1] for i in reversed(range(n_sample))]))

        # self.cnt += 1
        # if self.cnt == 1:
        #     for i in self._env_ids:
        #         f = open(f"state{i}.p", "wb")
        #         pickle.dump(list(self.state_queue[i]), f)
        #         f = open(f"reward{i}.p", "wb")
        #         pickle.dump(list(self.reward_queue[i]), f)
        #         f = open(f"done{i}.p", "wb")
        #         pickle.dump(list(self.done_queue[i]), f)
        #         f = open(f"action{i}.p", "wb")
        #         pickle.dump(list(self.action_queue[i]), f)

        return np.stack(state_batch), np.stack(action_batch), np.stack(reward_batch), np.stack(done_batch)


class NonBlockMemory(object):
    def __init__(self, n_env):
        """
        state_queue 存储交互得到的state，假设state是(84*84*4)，有8个env同时传递，那么每个state存储格式为$(8*84*84*4)$。
        """
        self._n_env = n_env
        self.state_queue = [list() for _ in range(n_env)]
        self.action_queue = [list() for _ in range(n_env)]
        self.reward_queue = [list() for _ in range(n_env)]
        self.done_queue = [list() for _ in range(n_env)]

    def store_tag_rds(self, tag: List[int], reward, done, state):
        for i in range(len(tag)):
            t = tag[i]
            self.reward_queue[t].append(reward[i, ...])
            self.done_queue[t].append(done[i, ...])
            self.state_queue[t].append(state[i, ...])

    def store_tag_s(self, tag: List[int], state):
        for i in range(len(tag)):
            t = tag[i]
            self.state_queue[t].append(state[i, ...])

    def store_tag_a(self, tag: List[int], action):
        for i in range(len(tag)):
            t = tag[i]
            self.action_queue[t].append(action[i, ...])

    def get_trajectory_and_clear(self):
        state_batch, action_batch, reward_batch, done_batch = [], [], [], []
        for i in range(self.n_env):
            trajectory_length = len(self.done_queue[i])
            if trajectory_length == 0:
                continue
            state_batch.append(np.asarray(self.state_queue[i]))
            action_batch.append(np.asarray(self.action_queue[i][:trajectory_length]))
            reward_batch.append(np.asarray(self.reward_queue[i]))
            done_batch.append(np.asarray(self.done_queue[i]))

            # Clear state, action, reward, done queue.
            action_length = len(self.action_queue[i])
            self.state_queue[i] = [self.state_queue[i][-1]]
            self.action_queue[i] = [] if action_length == trajectory_length else [self.action_queue[i][-1]]
            self.reward_queue[i] = []
            self.done_queue[i] = []

        return state_batch, action_batch, reward_batch, done_batch

    @property
    def n_env(self):
        return self._n_env
