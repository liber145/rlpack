import os
from collections import deque

import numpy as np
from rlpack.algos import DQN
from rlpack.common import DiscreteActionMemory
from rlpack.environment import AtariWrapper
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse

used_gpu = '0'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = used_gpu

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--env_name',  type=str, default="Pong-ramNoFrameskip-v4")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


class Config(object):
    def __init__(self):

        self.n_env = 1
        self.save_path = f"./log/dqn/exp_{args.env_name}"

        # Environment parameters.
        self.dim_observation = None
        self.dim_action = None   # For continuous action.

        # Traning length.
        self.update_step = int(10e6)   # for each env
        self.warm_start_length = 2000

        # Cycle parameters.
        self.log_freq = 1000

        # Algorithm parameters.
        self.batch_size = 64
        self.memory_size = int(1e5)


def process_config(env):
    config = Config()
    config.dim_observation = env.dim_observation
    config.dim_action = env.dim_action
    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):

    memory = DiscreteActionMemory(capacity=config.memory_size, n_env=config.n_env, dim_obs=config.dim_observation, datatype=np.uint8)
    epinfobuf = deque(maxlen=20)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    # ------------ Warm start --------------
    obs = env.reset()
    print("obs shape:", obs.shape, "type:", type(obs), "dtype:", obs.dtype)
    input()

    print(f"observation: max={np.max(obs)} min={np.min(obs)}")
    for i in tqdm(range(config.warm_start_length)):
        actions = agent.get_action(obs)
        next_obs, rewards, dones, infos = env.step(actions)
        memory.store_sards(obs, actions, rewards, dones, next_obs)
        obs = next_obs

        for info in infos:
            if "episode" in info:
                tqdm.write(f"{info}")

    print("Finish warm start.")
    print("Start training.")
    # --------------- Interaction, Train and Log ------------------
    for i in tqdm(range(config.update_step)):
        epinfos = []
        actions = agent.get_action(obs)
        next_obs, rewards, dones, infos = env.step(actions)

        memory.store_sards(obs, actions, rewards, dones, next_obs)

        obs = next_obs

        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)

        # Get the last trajectory from memory and train the algorithm.
        data_batch = memory.sample_transition(config.batch_size)
        loginfo = agent.update(data_batch, i)

        epinfobuf.extend(epinfos)

        if i > 0 and i % config.log_freq == 0:
            rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
            lenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            tqdm.write(f"eprewmean: {rewmean}  eplenmean: {lenmean} ")
            summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
            summary_writer.add_scalar("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]), global_step=i)


if __name__ == "__main__":
    env = AtariWrapper(f"{args.env_name}", 1)
    config = process_config(env)
    pol = DQN(n_env=config.n_env,
              rnd=0,
              dim_obs=config.dim_observation,
              dim_act=config.dim_action,
              discount=0.99,
              save_path=config.save_path,
              save_model_freq=1000,
              log_freq=config.log_freq,
              update_target_freq=10000,
              epsilon_schedule=lambda x: min(1.0, x / 5e6),
              lr=1e-4)

    learn(env, pol, config)
