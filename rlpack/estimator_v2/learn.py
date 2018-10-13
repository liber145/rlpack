import argparse
import gym
import numpy as np
import time
import os
import tqdm

from .dqn import DQN
from ..common.memory import Memory4 as Memory


parser = argparse.ArgumentParser(description="Parse Arguments.")
parser.add_argument("--epsilon", default=0.01, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--model", default="dqn", type=str)
parser.add_argument("--n_action", default=0, type=int)
parser.add_argument("--dim_observation", default=0, type=int)
parser.add_argument("--n_step", default=10000, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--memory_size", default=50000, type=int)
parser.add_argument("--discount", default=0.99, type=float)
parser.add_argument("--update_target_freq", default=1000, type=int)
parser.add_argument("--update_freq", default=100, type=int)
parser.add_argument("--save_model_freq", default=1000, type=int)
parser.add_argument("--save_path", default=None, type=str)
config = parser.parse_args()


def process_config(env):
    config.save_path = os.path.join("./log", config.model) if config.save_path is None else config.save_path
    config.dim_observation = env.observation_space.shape[0]
    config.n_action = env.action_space.n


def learn():

    env = gym.make("CartPole-v0")
    process_config(env)  # 配置config
    pol = DQN(config)
    mem = Memory(config.memory_size)

    s = env.reset()
    a = pol.get_action(s)

    for i in range(config.n_step):
        next_s, r, d, _ = env.step(a)
        mem.put([s, a, r, next_s, d])

        if i % config.update_freq == 0:
            minibatch = mem.sample(config.batch_size)
            pol.update(minibatch)


if __name__ == "__main__":
    learn()
