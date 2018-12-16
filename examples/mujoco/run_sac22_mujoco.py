# -*- coding: utf-8 -*-


import os
from collections import deque

import gym
import numpy as np
from rlpack.algos import SAC2
from rlpack.environment.mujoco_wrappers import make_mujoco
# from rlpack.common import Memory
from tensorboardX import SummaryWriter
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env_name',  type=str, default="Reacher-v2")
args = parser.parse_args()


class Config(object):
    """Config."""

    def __init__(self):
        """All papameters here."""
        self.rnd = 1
        self.save_path = f"./log/sac/sac2_{args.env_name}"

        # 环境
        self.n_env = 1
        self.dim_observation = None
        self.dim_action = None   # gym中不同环境的action数目不同。

        # 训练长度
        self.trajectory_length = 5000
        self.update_step = 5000   # for each env
        self.warm_start_length = 10000
        self.memory_size = int(1e6)

        # 周期参数
        self.save_model_freq = 50
        self.log_freq = 1

        # 算法参数
        self.batch_size = 100
        self.discount = 0.99
        self.policy_lr_schedule = lambda x: 1e-3
        self.value_lr_schedule = lambda x: 1e-3


class Memory:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
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
        return self.obs1_buf[idxs][np.newaxis, :], self.acts_buf[idxs][np.newaxis, :], self.rews_buf[idxs][np.newaxis, :], self.done_buf[idxs][np.newaxis, :], self.obs2_buf[idxs][np.newaxis, :]


def process_env(env):
    config = Config()
    config.dim_observation = env.dim_observation
    config.dim_action = env.dim_action
    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):
    memory = Memory(config.dim_observation[0], config.dim_action, config.memory_size)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    epr, all_r = deque(maxlen=20), []
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    steps_per_epoch = 5000
    epochs = 100
    total_steps = steps_per_epoch * epochs

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy. 
        """
        if t > 10000:
            a = agent.get_action(o)
        else:
            # a = env.action_space.sample()
            a = env.sample_action()

        # Step the env
        o2, r, d, _ = env.step(a)
        # print(f"o2 max: {np.max(o2)} min: {np.min(o2)}  r: {r}  d: {d}")
        ep_ret += r
        ep_len += 1

        all_r.append(r)

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len == 1000 else d

        # Store experience to replay buffer
        memory.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        if d or (ep_len == 1000):
            """
            Perform all SAC updates at the end of the trajectory.
            This is a slight difference from the SAC specified in the
            original paper.
            """
            print(ep_len, max(all_r), min(all_r))
            for j in range(ep_len):
                batch = memory.sample_batch(100)
                agent.update(batch, update_ratio=0)

            epr.append(ep_ret)
            ep_ret, ep_len = 0, 0
            # o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if t > 0 and t % steps_per_epoch == 0:
            epoch = t // steps_per_epoch

            print(f"epoch: {epoch}  rewmean: {np.mean(epr)}")
            summary_writer.add_scalar("eprew", np.mean(epr), epoch)


if __name__ == "__main__":
    # env = gym.make(f"{args.env_name}")
    env = make_mujoco(f"{args.env_name}")
    env.seed(4)
    config = process_env(env)
    agent = SAC2(config)
    learn(env, agent, config)
