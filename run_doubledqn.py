import argparse
import gym
import numpy as np
import time
import datetime
import os
from tqdm import tqdm

from rlpack.algos.double_dqn import DoubleDQN
from rlpack.common.memory import Memory4 as Memory


parser = argparse.ArgumentParser(description="Parse Arguments.")
parser.add_argument("--initial_epsilon", default=0.5, type=float)
parser.add_argument("--final_epsilon", default=0.01, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--model", default="dqn", type=str)
parser.add_argument("--n_step", default=1000000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--memory_size", default=2000, type=int)
parser.add_argument("--discount", default=0.9, type=float)
parser.add_argument("--update_target_freq", default=100, type=int)
parser.add_argument("--update_freq", default=1, type=int)
parser.add_argument("--save_model_freq", default=1000, type=int)
parser.add_argument("--save_path", default=None, type=str)
config = parser.parse_args()


def process_config(env):
    config.model = "doubledqn"
    time_stamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    config.save_path = os.path.join("./log", config.model+time_stamp) if config.save_path is None else config.save_path
    config.dim_observation = env.observation_space.shape[0]
    config.n_action = env.action_space.n


def learn():

    env = gym.make("CartPole-v1")
    process_config(env)  # 配置config
    pol = DoubleDQN(config)
    mem = Memory(config.memory_size)

    s = env.reset()
    for i in tqdm(range(1, config.n_step+1)):
        a = pol.get_action(s)
        next_s, r, d, _ = env.step(a)
        modified_r = -1 if d else 0.1
        pol.put(r, d)
        mem.put([s, a, modified_r, next_s, d])

        # 到了更新周期。
        if i % config.update_freq == 0 and i > config.batch_size:
            minibatch = mem.sample(config.batch_size)
            pol.update(minibatch)

        # 游戏轨迹结束。
        if d is True:
            s = env.reset()

        #　获得新的动作。
        s = next_s


if __name__ == "__main__":
    learn()
