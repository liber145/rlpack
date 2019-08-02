import pickle
import random
from collections import defaultdict, deque
from typing import List, Tuple
import math

import numpy as np


class ContinuousActionMemory(object):
    """
    Memory for continuous-action environments.

    Arguments:
        - capacity: replay buffer size.
        - n_env: the number of environments.
        - dim_obs: tuple. the dimension of observaitons, like (16,) or (84, 84, 4).
        - dim_act: int. the dimension of actions.
    """

    def __init__(self, *, capacity=0, n_env: int=1, dim_obs: Tuple=None, dim_act: int=None):
        self.state_queue = np.zeros((capacity, n_env, *dim_obs), dtype=np.float32)
        self.action_queue = np.zeros((capacity, n_env, dim_act), dtype=np.float32)
        self.reward_queue = np.zeros((capacity, n_env), dtype=np.float32)
        self.done_queue = np.zeros((capacity, n_env), dtype=np.float32)
        self.next_state_queue = np.zeros((capacity, n_env, *dim_obs), dtype=np.float32)

        self.ptr, self.size = 0, 0
        self.capacity = capacity
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.n_env = n_env

    def store_sards(self, state, action, reward, done, next_state):

        assert state.shape == (self.n_env, *self.dim_obs)
        assert action.shape == (self.n_env, self.dim_act)
        assert reward.shape == (self.n_env,)
        assert done.shape == (self.n_env,)
        assert next_state.shape == (self.n_env, *self.dim_obs)

        self.state_queue[self.ptr, :] = state
        self.action_queue[self.ptr, :] = action
        self.reward_queue[self.ptr, :] = reward
        self.done_queue[self.ptr, :] = done
        self.next_state_queue[self.ptr, :] = next_state

        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)

    def _get_last_n_index(self, n_sample):

        if self.ptr >= n_sample:
            index = list(range(self.ptr - n_sample, self.ptr))
        else:
            rest = n_sample - self.ptr
            index = list(range(self.capacity - rest, self.capacity)) + list(range(self.ptr))

        return index

    def get_last_n_samples(self, n):
        assert n <= self.size, "No enough sample in memory."

        index = self._get_last_n_index(n)

        state_batch = np.concatenate([self.state_queue[index], self.next_state_queue[index[-1]][np.newaxis, :]], axis=0)
        action_batch = self.action_queue[index]
        reward_batch = self.reward_queue[index]
        done_batch = self.done_queue[index]

        n_env = state_batch.shape[1]
        ob_shape = state_batch.shape[2:]

        assert state_batch.shape == (n + 1, n_env, *ob_shape)

        state_batch = state_batch.swapaxes(1, 0)
        action_batch = action_batch.swapaxes(1, 0)
        reward_batch = reward_batch.swapaxes(1, 0)
        done_batch = done_batch.swapaxes(1, 0)

        return state_batch, action_batch, reward_batch, done_batch

    def sample_transition(self, n):
        idxs = np.random.randint(self.size, size=n)
        state_batch = self.state_queue[idxs, :]
        action_batch = self.action_queue[idxs, :]
        reward_batch = self.reward_queue[idxs, :]
        done_batch = self.done_queue[idxs, :]
        next_state_batch = self.next_state_queue[idxs, :]

        state_batch = state_batch.swapaxes(0, 1)
        action_batch = action_batch.swapaxes(0, 1)
        reward_batch = reward_batch.swapaxes(0, 1)
        done_batch = done_batch.swapaxes(0, 1)
        next_state_batch = next_state_batch.swapaxes(0, 1)

        return state_batch, action_batch, reward_batch, done_batch, next_state_batch


class DiscreteActionMemory(ContinuousActionMemory):
    """
    Memory for discrete-action environments.

    Arguments:
        - capacity: replay buffer size.
        - n_env: the number of environments.
        - dim_obs: tuple. the dimension of observaitons, like (16,) or (84, 84, 4).
    """

    def __init__(self, *, capacity=0, n_env: int=1, dim_obs: Tuple=None, datatype=np.float32):
        self.state_queue = np.zeros((capacity, n_env, *dim_obs), dtype=datatype)
        self.action_queue = np.zeros((capacity, n_env), dtype=np.int32)
        self.reward_queue = np.zeros((capacity, n_env), dtype=np.float32)
        self.done_queue = np.zeros((capacity, n_env), dtype=np.float32)
        self.next_state_queue = np.zeros((capacity, n_env, *dim_obs), dtype=datatype)

        self.ptr, self.size = 0, 0
        self.capacity = capacity
        self.dim_obs = dim_obs
        self.n_env = n_env

    def store_sards(self, state, action, reward, done, next_state):

        assert state.shape == (self.n_env, *self.dim_obs)
        assert action.shape == (self.n_env,)
        assert reward.shape == (self.n_env,)
        assert done.shape == (self.n_env,)
        assert next_state.shape == (self.n_env, *self.dim_obs)

        self.state_queue[self.ptr, :] = state
        self.action_queue[self.ptr, :] = action
        self.reward_queue[self.ptr, :] = reward
        self.done_queue[self.ptr, :] = done
        self.next_state_queue[self.ptr, :] = next_state

        self.ptr = (self.ptr+1) % self.capacity
        self.size = min(self.size+1, self.capacity)


class AsyncContinuousActionMemory(object):
    def __init__(self, maxsize: int = 0, dim_obs: Tuple=None, dim_act: int=None):
        self.env_wrapper = None
        self.state_queue = defaultdict(lambda: np.zeros((maxsize + 1, *dim_obs), dtype=np.float32))
        self.action_queue = defaultdict(lambda: np.zeros((maxsize + 1, dim_act), dtype=np.float32))
        self.reward_queue = defaultdict(lambda: np.zeros((maxsize), dtype=np.float32))
        self.done_queue = defaultdict(lambda: np.zeros((maxsize), dtype=np.float32))
        self.ptr_queue = defaultdict(int)
        self.cnt_queue = defaultdict(int)
        self.s_maxsize, self.a_maxsize, self.r_maxsize, self.d_maxsize = maxsize + 1, maxsize + 1, maxsize, maxsize

    def register(self, env_wrapper):
        self.env_wrapper = env_wrapper

    @property
    def _env_ids(self):
        assert self.env_wrapper is not None, "Not register environment"
        return self.env_wrapper.env_id

    def store_s(self, states):

        for env_id, s in zip(self._env_ids, states):
            self.state_queue[env_id][self.ptr_queue[(env_id, "s")]] = s
            self.ptr_queue[(env_id, "s")] = (self.ptr_queue[(env_id, "s")] + 1) % self.s_maxsize
            self.cnt_queue[(env_id, "s")] += 1

    def store_a(self, actions):
        for env_id, a in zip(self._env_ids, actions):
            self.action_queue[env_id][self.ptr_queue[(env_id, "a")]] = a
            self.ptr_queue[(env_id, "a")] = (self.ptr_queue[(env_id, "a")] + 1) % self.a_maxsize
            self.cnt_queue[(env_id, "a")] += 1

    def store_rds(self, rewards, dones, states):
        for env_id, r, d, s in zip(self._env_ids, rewards, dones, states):
            self.reward_queue[env_id][self.ptr_queue[(env_id, "r")]] = r
            self.done_queue[env_id][self.ptr_queue[(env_id, "d")]] = d
            self.state_queue[env_id][self.ptr_queue[(env_id, "s")]] = s

            self.ptr_queue[(env_id, "r")] = (self.ptr_queue[(env_id, "r")] + 1) % self.r_maxsize
            self.ptr_queue[(env_id, "d")] = (self.ptr_queue[(env_id, "d")] + 1) % self.d_maxsize
            self.ptr_queue[(env_id, "s")] = (self.ptr_queue[(env_id, "s")] + 1) % self.s_maxsize
            self.cnt_queue[(env_id, "r")] += 1
            self.cnt_queue[(env_id, "d")] += 1
            self.cnt_queue[(env_id, "s")] += 1

    def _get_last_n_index(self, env_id, n_sample):
        s_ptr = self.ptr_queue[(env_id, "s")]
        r_ptr = self.ptr_queue[(env_id, "r")]
        d_ptr = self.ptr_queue[(env_id, "d")]
        a_ptr = self.ptr_queue[(env_id, "a")]

        if s_ptr >= n_sample + 1:
            s_index = list(range(s_ptr - n_sample - 1, s_ptr))
        else:
            rest = n_sample + 1 - s_ptr
            s_index = list(range(self.s_maxsize - rest, self.s_maxsize)) + list(range(s_ptr))

        if r_ptr >= n_sample:
            r_index = list(range(r_ptr - n_sample, r_ptr))
        else:
            rest = n_sample - r_ptr
            r_index = list(range(self.r_maxsize - rest, self.r_maxsize)) + list(range(r_ptr))

        if d_ptr >= n_sample:
            d_index = list(range(d_ptr - n_sample, d_ptr))
        else:
            rest = n_sample - d_ptr
            d_index = list(range(self.d_maxsize - rest, self.d_maxsize)) + list(range(d_ptr))

        a_ptr = (a_ptr - 1) % self.a_maxsize if s_ptr == a_ptr else a_ptr
        if a_ptr >= n_sample:
            a_index = list(range(a_ptr - n_sample, a_ptr))
        else:
            rest = n_sample - a_ptr
            a_index = list(range(self.a_maxsize - rest, self.a_maxsize)) + list(range(a_ptr))

        return s_index, a_index, r_index, d_index

    def get_last_n_samples(self, n_sample):
        state_batch, action_batch, reward_batch, done_batch = [], [], [], []

        for env_id in self.done_queue.keys():
            assert self.cnt_queue[(env_id, "d")] >= n_sample, "Not enough warm steps or requiring too many samples."

            s_index, a_index, r_index, d_index = self._get_last_n_index(env_id, n_sample)

            state_batch.append(self.state_queue[env_id][s_index])
            action_batch.append(self.action_queue[env_id][a_index])
            reward_batch.append(self.reward_queue[env_id][r_index])
            done_batch.append(self.done_queue[env_id][d_index])

        return np.stack(state_batch), np.stack(action_batch), np.stack(reward_batch), np.stack(done_batch)

    def _sample_transition_index(self, env_id, n_sample):
        s_ptr = self.ptr_queue[(env_id, "s")]
        r_ptr = self.ptr_queue[(env_id, "r")]
        d_ptr = self.ptr_queue[(env_id, "d")]
        a_ptr = self.ptr_queue[(env_id, "a")]

        s_cnt = self.cnt_queue[(env_id, "s")]
        r_cnt = self.cnt_queue[(env_id, "r")]
        d_cnt = self.cnt_queue[(env_id, "d")]
        a_cnt = self.cnt_queue[(env_id, "a")]

        s_start = 0 if s_ptr == s_cnt else s_ptr
        r_start = 0 if r_ptr == r_cnt else r_ptr
        d_start = 0 if d_ptr == d_cnt else d_ptr
        a_start = 0 if a_ptr == a_cnt else a_ptr

        s_size = min(s_cnt, self.s_maxsize)
        index = np.random.randint(s_size-1, size=n_sample)

        s_index = (index + s_start) % self.s_maxsize
        r_index = (index + r_start) % self.r_maxsize
        d_index = (index + d_start) % self.d_maxsize
        next_s_index = (index + s_start + 1) % self.s_maxsize

        if a_cnt >= self.a_maxsize and a_cnt < s_cnt:
            a_index = (index + a_start + 1) % self.a_maxsize
        else:
            a_index = (index + a_start) % self.a_maxsize

        return s_index, a_index, r_index, d_index, next_s_index

    def sample_transition(self, n):

        state_batch, action_batch, reward_batch, done_batch, next_state_batch = [], [], [], [], []

        env_id_keys = self.done_queue.keys()
        # n_queue = len(env_id_keys)
        # n_not_last = math.ceil(n / n_queue)
        # n_last = n - (n_queue - 1) * n_not_last

        for j, env_id in enumerate(env_id_keys):
            # if j == len(env_id_keys) - 1:
            #     n_sample = n_last
            # else:
            #     n_sample = n_not_last

            s_index, a_index, r_index, d_index, next_s_index = self._sample_transition_index(env_id, n)
            state_batch.append(self.state_queue[env_id][s_index])
            action_batch.append(self.action_queue[env_id][a_index])
            reward_batch.append(self.reward_queue[env_id][r_index])
            done_batch.append(self.done_queue[env_id][d_index])
            next_state_batch.append(self.state_queue[env_id][next_s_index])

        return np.stack(state_batch), np.stack(action_batch), np.stack(reward_batch), np.stack(done_batch), np.stack(next_state_batch)


class AsyncDiscreteActionMemory(AsyncContinuousActionMemory):
    def __init__(self, maxsize: int=0, dim_obs: Tuple=None, datatype=np.float32):
        self.env_wrapper = None
        self.state_queue = defaultdict(lambda: np.zeros((maxsize + 1, *dim_obs), dtype=datatype))
        self.action_queue = defaultdict(lambda: np.zeros((maxsize + 1), dtype=np.int32))
        self.reward_queue = defaultdict(lambda: np.zeros((maxsize), dtype=np.float32))
        self.done_queue = defaultdict(lambda: np.zeros((maxsize), dtype=datatype))
        self.ptr_queue = defaultdict(int)
        self.cnt_queue = defaultdict(int)
        self.s_maxsize, self.a_maxsize, self.r_maxsize, self.d_maxsize = maxsize + 1, maxsize + 1, maxsize, maxsize
