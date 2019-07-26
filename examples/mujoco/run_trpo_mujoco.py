# -*- coding: utf-8 -*-


import argparse
import os
from collections import deque

import gym
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm

from rlpack.algos import ContinuousTRPO
from rlpack.common import AsyncContinuousActionMemory
from rlpack.environment import AsyncMujocoWrapper

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env',  type=str, default="Reacher-v2")
parser.add_argument('--epochs', type=int, default=1000)
args = parser.parse_args()

env = gym.make(args.env)


class Config(object):
    """Config."""

    def __init__(self):
        """All papameters here."""
        self.rnd = 5
        self.save_path = f"./log/trpo/exp_async_{args.env}"

        # 环境
        self.n_env = 1
        self.dim_observation = None
        self.dim_action = None   # gym中不同环境的action数目不同。

        # 训练长度
        self.trajectory_length = 2048
        self.update_step = 5000   # for each env
        self.warm_start_length = 2000
        self.memory_size = 2149

        # 周期参数
        self.log_freq = 1

        # 算法参数
        # self.batch_size = 64
        # self.discount = 0.99
        # self.gae = 0.95
        # self.delta = 0.01
        # self.max_grad_norm = 40


def process_env(env):
    config = Config()
    config.dim_observation = env.dim_observation
    config.dim_action = env.dim_action
    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):
    memory = AsyncContinuousActionMemory(maxsize=config.memory_size, dim_obs=config.dim_observation, dim_act=config.dim_action)
    memory.register(env)
    epinfobuf = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    # 热启动，随机收集数据。
    obs = env.reset()
    memory.store_s(obs)
    print(f"observation: max={np.max(obs)} min={np.min(obs)}")
    for i in tqdm(range(config.warm_start_length)):
        actions = env.sample_action(obs.shape[0])
        memory.store_a(actions)
        next_obs, rewards, dones, infos = env.step(actions)

        memory.store_rds(rewards, dones, next_obs)
        obs = next_obs

    print("Finish warm start.")
    print("Start training.")
    for i in tqdm(range(config.update_step)):
        epinfos = []
        for _ in range(config.trajectory_length):
            actions = agent.get_action(obs)
            memory.store_a(actions)
            next_obs, rewards, dones, infos = env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            memory.store_rds(rewards, dones, next_obs)
            obs = next_obs

        update_ratio = i / config.update_step
        data_batch = memory.get_last_n_samples(config.trajectory_length)
        agent.update(data_batch, update_ratio)

        epinfobuf.extend(epinfos)
        summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
        summary_writer.add_scalar("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]), global_step=i)

        if i > 0 and i % config.log_freq == 0:
            rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
            lenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            tqdm.write(f"iter: {i}  eprewmean: {rewmean}  eplenmean: {lenmean}  rew: {epinfobuf[-1]['r']}  len: {epinfobuf[-1]['l']}")


def run_main():
    agent = ContinuousTRPO(n_env=3,
                           dim_obs=env.dim_observation,
                           dim_act=12)

    s = env.reset()
    for epoch in range(args.epochs):
        states, actions, rewards, dones = [s], [], [], []
        for t in range(local_steps_per_epoch):
            a = agent.get_action(s)[0]
            s, r, d, _ = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)
            dones.append(d)

            if d:
                s = env.reset()


if __name__ == "__main__":
    env = AsyncMujocoWrapper(f"{args.env}", 1, 1, 50049)
    config = process_env(env)
    agent = TRPO(rnd=1,
                 n_env=1,
                 dim_obs=config.dim_observation,
                 dim_act=config.dim_action,
                 discount=0.99,
                 save_path=f"./log/trpo/exp_async_{args.env}",
                 save_model_freq=1000,
                 log_freq=1000,
                 trajectory_length=2048,
                 gae=0.95,
                 delta=0.01,
                 training_epoch=10,
                 max_grad_norm=40,
                 lr=3e-3)

    learn(env, agent, config)
