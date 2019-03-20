import argparse
from collections import deque, Counter
import gym
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf

from rlpack.algos import DQN

from utils import AgentWrapper

agent_wrapper = AgentWrapper(port=50000)
agent_wrapper.start()


nb_actions = 6


class Memory(object):
    def __init__(self, capacity: int, dim_obs, dim_act, statetype=np.float32):
        self._state = np.zeros((capacity, *dim_obs), dtype=statetype)
        self._action = np.zeros(capacity, dtype=np.int32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._next_state = np.zeros((capacity, *dim_obs), dtype=statetype)

        self._size = 0
        self._capacity = capacity

    def store_sards(self, state, action, reward, done, next_state):
        ind = self._size % self._capacity
        self._state[ind, ...] = state
        self._action[ind] = action
        self._reward[ind] = reward
        self._done[ind] = done
        self._next_state[ind, ...] = next_state
        self._size += 1

    def sample(self, n: int):
        n_sample = self._size if self._size < self._capacity else self._capacity
        inds = np.random.randint(n_sample, size=n)
        state_batch = self._state[inds, ...]
        action_batch = self._action[inds]
        reward_batch = self._reward[inds]
        done_batch = self._done[inds]
        next_state_batch = self._next_state[inds, ...]
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch


def obs_fn():
    obs = tf.placeholder(shape=[None, 128, 4],
                         dtype=tf.uint8, name="observation")
    obs = tf.to_float(obs) / 255.0
    return obs


def value_fn(obs):
    x = tf.layers.conv1d(obs, filters=32, kernel_size=8,
                         strides=4, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=4,
                         strides=2, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=3,
                         strides=1, activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=nb_actions)
    return x


def main():
    agent = DQN(obs_fn=obs_fn,
                value_fn=value_fn,
                dim_act=nb_actions,
                update_target_freq=100,
                log_freq=10,
                save_path=f"./log/dqn_ramatari/",
                lr=2.5e-4,
                epsilon_schedule=lambda x: max(0.1, (1e6-x) / 1e6),
                train_epoch=1)

    while True:
        for i in tqdm(range(1000)):
            env_ids, states, rewards, dones = agent_wrapper.get_srd_batch(
                batchsize=2)

            actions = agent.get_action(np.asarray(states))
            agent_wrapper.put_a_batch(env_ids, actions)

        s_batch, a_batch, r_batch, d_batch = agent_wrapper.get_episodes()

        # add to memory
if __name__ == "__main__":
    main()
