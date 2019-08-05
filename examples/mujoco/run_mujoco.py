# -*- coding: utf-8 -*-


import argparse
import time
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf

from rlpack.algos import PPO, TRPO, TD3
from rlpack.utils import mlp, mlp_gaussian_policy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env',  type=str, default="Reacher-v2")
args = parser.parse_args()

env = gym.make(args.env)
dim_obs = env.observation_space.shape
dim_act = env.action_space.shape[0]
act_limit = 1
max_ep_len = 1000


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


def policy_fn(x):
    x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    act = act_limit * tf.layers.dense(x, units=dim_act, activation=tf.tanh)
    return act


def value_fn(x, a):
    y = tf.layers.dense(tf.concat([x, a], axis=-1), units=64, activation=tf.nn.relu)
    y = tf.layers.dense(y, units=64, activation=tf.nn.relu)
    v = tf.squeeze(tf.layers.dense(y, units=1, activation=None), axis=1)
    return v


def run_main():

    # agent = TRPO(dim_act=dim_act, dim_obs=dim_obs, policy_fn=trpo_policy_fn, value_fn=trpo_value_fn, delta=0.1, save_path="./log/trpo")
    # agent = PPO(dim_act=dim_act, dim_obs=dim_obs, policy_fn=ppo_policy_fn, value_fn=value_fn, save_path="./log/ppo")
    agent = TD3(dim_act=dim_act, dim_obs=dim_obs, act_limit=act_limit, policy_fn=policy_fn, value_fn=value_fn, save_path="./log/td3")
    replay_buffer = ReplayBuffer(obs_dim=dim_obs, act_dim=dim_act, size=int(1e6))

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(50):
        ep_ret_list, ep_len_list = [], []
        for t in range(1000):
            a = agent.get_action(o[np.newaxis, :])[0]
            nexto, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            replay_buffer.store(o, a, r, nexto, int(d))

            o = nexto

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == 1000-1):

                for _ in range(ep_len):
                    batch = replay_buffer.sample_batch(100)
                    agent.update([batch["obs1"], batch["acts"], batch["rews"], batch["done"], batch["obs2"]])

                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                if terminal:
                    # 当到达完结状态或是最长状态时，记录结果
                    ep_ret_list.append(ep_ret)
                    ep_len_list.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        print(f"{epoch}th epoch. average_return={np.mean(ep_ret_list)}, average_len={np.mean(ep_len_list)}")

        agent.add_scalar("average_return", np.mean(ep_ret_list), epoch*1000)
        agent.add_scalar("average_length", np.mean(ep_len_list), epoch*1000)

    elapsed_time = time.time() - start_time
    print("elapsed time:", elapsed_time)


if __name__ == "__main__":
    run_main()
