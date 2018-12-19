import os
from collections import deque

import numpy as np
from rlpack.algos import DQN
from rlpack.common import DiscreteActionMemory
from rlpack.environment import AtariWrapper
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env_name',  type=str, default="BreakoutNoFrameskip-v4")
args = parser.parse_args()


class Config(object):
    def __init__(self):
        # random seed and path.
        self.rnd = 1
        self.save_path = f"./log/dqn_{args.env_name}"

        # Environment parameters.
        self.n_env = 1
        self.dim_observation = None
        self.dim_action = None   # For continuous action.

        # Traning length.
        self.update_step = int(1e6)   # for each env
        self.warm_start_length = 2000

        # Cycle parameters.
        self.save_model_freq = 1000
        self.log_freq = 100
        self.update_target_freq = 100

        # Algorithm parameters.
        self.batch_size = 64
        self.discount = 0.99
        self.max_grad_norm = 0.5
        self.value_lr_schedule = lambda x: 2.5e-4
        self.epsilon_schedule = lambda x: (1-x) * 0.5
        self.memory_size = 10000


def process_config(env):
    config = Config()
    config.dim_observation = env.dim_observation
    config.dim_action = env.dim_action
    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):

    memory = DiscreteActionMemory(capacity=config.memory_size, n_env=config.n_env, dim_obs=config.dim_observation)
    epinfobuf = deque(maxlen=20)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    # ------------ Warm start --------------
    obs = env.reset()
    print(f"observation: max={np.max(obs)} min={np.min(obs)}")
    for i in tqdm(range(config.warm_start_length)):
        actions = env.sample_action()
        next_obs, rewards, dones, infos = env.step(actions)
        memory.store_sards(obs, actions, rewards, dones, obs)
        obs = next_obs


if __name__ == "__main__":
    env = AtariWrapper(f"{args.env_name}", 1)
    config = process_config(env)
    pol = DQN(config)

    learn(env, pol, config)
