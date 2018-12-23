import os
from collections import deque

import numpy as np
from rlpack.algos import DoubleDQN
from rlpack.common import AsyncDiscreteActionMemory
from rlpack.environment import AsyncAtariWrapper
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env_name',  type=str, default="BreakoutNoFrameskip-v4")
args = parser.parse_args()


class Config(object):
    def __init__(self):
        # random seed and path.
        self.save_path = f"./log/doubledqn/exp_async_{args.env_name}"

        # Environment parameters.
        self.n_env = 4
        self.dim_observation = None
        self.dim_action = None   # For continuous action.

        # Traning length.
        self.update_step = int(1e6)   # for each env
        self.warm_start_length = 2000

        # Cycle parameters.
        self.log_freq = 1000

        # Algorithm parameters.
        self.batch_size = 64
        self.memory_size = 10000


def process_config(env):
    config = Config()
    config.dim_observation = env.dim_observation
    config.dim_action = env.dim_action
    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):

    memory = AsyncDiscreteActionMemory(maxsize=config.memory_size, dim_obs=config.dim_observation, datatype=np.uint8)
    memory.register(env)
    epinfobuf = deque(maxlen=20)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    # ------------ Warm start --------------
    obs = env.reset()
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
    for i in tqdm(range(config.update_step)):
        epinfos = []

        actions = agent.get_action(obs)
        memory.store_a(actions)
        obs, rewards, dones, infos = env.step(actions)
        memory.store_rds(rewards, dones, obs)

        for info in infos:
            maybeepinfo = info.get('episode')
            if maybeepinfo:
                epinfos.append(maybeepinfo)

        # Get the last trajectory from memory and train the algorithm.
        update_ratio = i / config.update_step
        data_batch = memory.sample_transition(config.batch_size)
        loginfo = agent.update(data_batch, update_ratio)

        epinfobuf.extend(epinfos)
        summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
        summary_writer.add_scalar("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]), global_step=i)

        if i > 0 and i % config.log_freq == 0:
            rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
            lenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            tqdm.write(f"eprewmean: {rewmean}  eplenmean: {lenmean}")


if __name__ == "__main__":
    env = AsyncAtariWrapper(f"{args.env_name}", 4, 3, 50000)
    config = process_config(env)
    pol = DoubleDQN(n_env=config.n_env,
                    rnd=1,
                    dim_obs=config.dim_observation,
                    dim_act=config.dim_action,
                    discount=0.99,
                    save_path=config.save_path,
                    save_model_freq=1000,
                    update_target_freq=10000,
                    epsilon_schedule=lambda x: (1-x),
                    lr=2.5e-4
                    )

    learn(env, pol, config)
