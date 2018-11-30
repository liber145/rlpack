import os
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from rlpack.algos import ContinuousPPO
from rlpack.common import DistributedMemory
from rlpack.environment import AsyncMujocoWrapper
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.seed = 1
        self.save_path = "./log/ppo_reacher_v25"
        self.save_model_freq = 0.001
        self.log_freq = 1

        # 环境
        self.dim_observation = None
        self.dim_action = None   # gym中不同环境的action数目不同。
        self.n_action = None

        # 训练长度
        self.n_env = 1
        self.trajectory_length = 2048
        self.update_step = 1000   # for each env
        self.batch_size = 64
        self.warm_start_length = 100
        self.memory_size = 2049

        # 训练参数
        self.training_epoch = 10
        self.discount = 0.99
        self.gae = 0.95
        self.policy_lr_schedule = lambda x: 3e-4
        self.value_lr_schedule = lambda x: 3e-4

        self.clip_schedule = lambda x: (1 - x) * 0.1
        self.vf_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 40

        self.initial_epsilon = 0.9
        self.final_epsilon = 0.01
        self.update_target_freq = 100


def process_env(env):
    config = Config()
    config.dim_observation = env.dim_observation
    config.dim_action = env.dim_action[0]
    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):

    memory = DistributedMemory(config.memory_size)
    memory.register(env)
    epinfobuf = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    # 热启动，随机收集数据。
    obs = env.reset()
    memory.store_s(obs)
    print(f"observation: max={np.max(obs)} min={np.min(obs)}")
    for i in tqdm(range(config.warm_start_length)):
        actions = agent.get_action(obs)
        memory.store_a(actions)
        next_obs, rewards, dones, infos = env.step(actions)

        memory.store_rds(rewards, dones, next_obs)
        obs = next_obs

    print("Finish warm start.")
    print("Start training.")
    for i in tqdm(range(config.update_step)):
        epinfos = []
        for _ in range(int(config.trajectory_length * 3 / 3)):
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
        print(f"state shape: {data_batch[0].shape}")
        agent.update(data_batch, update_ratio)

        epinfobuf.extend(epinfos)
        summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
        summary_writer.add_scalar("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]), global_step=i)

        if i > 0 and i % config.log_freq == 0:
            rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
            lenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            print(f"eprewmean: {rewmean}  eplenmean: {lenmean}")


if __name__ == "__main__":
    env = AsyncMujocoWrapper("Reacher-v2", 1, 1, 50001)
    config = process_env(env)
    agent = ContinuousPPO(config)
    learn(env, agent, config)
