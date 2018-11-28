import os
from collections import deque

import numpy as np
from rlpack.algos import PPO
from rlpack.common import DistributedMemory
from rlpack.environment import DistributedAtariWrapper
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.seed = 1
        self.save_path = "./log/ppo_breakout_v16"
        self.save_model_freq = 0.001
        self.log_freq = 10

        # 环境
        self.dim_observation = (84, 84, 4)
        self.dim_action = None   # For continuous action.
        self.n_action = 4   # For discrete action.

        # 训练长度
        self.n_env = 8
        self.trajectory_length = 128
        self.n_trajectory = 10000   # for each env
        self.batch_size = 64
        self.warm_start_length = 1

        # 训练参数
        self.training_epoch = 3
        self.discount = 0.99
        self.gae = 0.95
        self.vf_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.lr_schedule = lambda x: (1 - x) * 2.5e-4
        self.clip_schedule = lambda x: (1 - x) * 0.1
        self.memory_size = 1000


def process_config(env):
    config = Config()
    config.dim_observation = env.observation_space.shape
    config.n_action = env.action_space.n

    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):

    memory = DistributedMemory(maxlen=1000)
    memory.register(env)
    epinfobuf = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    # ------------ Warm start --------------
    obs = env.reset()
    print(obs.shape)
    memory.store_s(obs)
    print(f"observation: max={np.max(obs)} min={np.min(obs)}")
    for i in tqdm(range(config.warm_start_length)):
        actions = agent.get_action(obs)
        memory.store_a(actions)
        obs, rewards, dones, infos = env.step(actions)
        memory.store_rds(rewards, dones, obs)

    print("Finish warm start.")
    print("Start training.")
    # --------------- Interaction, Train and Log ------------------
    for i in tqdm(range(config.n_trajectory)):
        epinfos = []
        for _ in range(config.trajectory_length):
            actions = agent.get_action(obs)
            memory.store_a(actions)
            obs, rewards, dones, infos = env.step(actions)
            memory.store_rds(rewards, dones, obs)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

        # Get the last trajectory from memory and train the algorithm.
        update_ratio = i / config.n_trajectory
        data_batch = memory.get_all_and_clear()
        loginfo = agent.update(data_batch, update_ratio)
        # print(f"critic_loss: {loginfo['critic_loss']}   surrogate: {loginfo['surrogate']}   entropy: {loginfo['entropy']}   sample ratio: {loginfo['sample_ratio']} ")

        epinfobuf.extend(epinfos)
        summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
        summary_writer.add_scalar("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]), global_step=i)

        if i > 0 and i % config.log_freq == 0:
            rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
            lenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            print(f"eprewmean: {rewmean}  eplenmean: {lenmean}")


if __name__ == "__main__":
    config = Config()
    env = DistributedAtariWrapper("BreakoutNoFrameskip-v4", config.n_env)
    pol = PPO(config)

    learn(env, pol, config)
