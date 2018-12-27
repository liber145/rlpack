from collections import deque

import numpy as np
from rlpack.algos import DQN
from rlpack.common import DiscreteActionMemory
from rlpack.environment import ClassicControlWrapper
from tqdm import tqdm

import argparse


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--env_name", type=str, default="CartPole-v0")
args = parser.parse_args()


N_ENV = 1
SAVE_PATH = f"./log/dqn/exp_{args.env_name}"
DIM_OBS = None
DIM_ACT = None
UPDATE_STEP = int(1e6)
WARM_START = 500
LOG_FREQ = 100

BATCH_SIZE = 64
MEMORY_SIZE = int(1e5)


def get_dim(env):
    global DIM_OBS, DIM_ACT
    DIM_OBS = env.dim_observation
    DIM_ACT = env.dim_action


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent):

    memory = DiscreteActionMemory(capacity=MEMORY_SIZE, n_env=N_ENV, dim_obs=DIM_OBS)
    epinfobuf = deque(maxlen=20)

    obs = env.reset()
    print(f"observation: max={np.max(obs)} min={np.min(obs)}")
    # for i in tqdm(range(WARM_START)):
    #     actions = env.sample_action()
    #     next_obs, rewards, dones, infos = env.step(actions)
    #     memory.store_sards(obs, actions, rewards, dones, obs)
    #     obs = next_obs

    tt = 0
    for i in tqdm(range(UPDATE_STEP)):
        actions = agent.get_action(obs)
        next_obs, rewards, dones, infos = env.step(actions)
        memory.store_sards(obs, actions, rewards, dones, next_obs)
        obs = next_obs

        epinfobuf.extend([info["episode"] for info in infos if "episode" in info])

        update_ratio = i / UPDATE_STEP
        data_batch = memory.sample_transition(BATCH_SIZE)
        agent.update(data_batch, update_ratio)

        if i % LOG_FREQ == 0:
            rewmean = safemean([x["r"] for x in epinfobuf])
            lenmean = safemean([x["l"] for x in epinfobuf])
            tqdm.write(f"ep: {tt} eprewmean: {rewmean}  eplenmean: {lenmean}")


if __name__ == "__main__":
    env = ClassicControlWrapper(f"{args.env_name}", N_ENV)
    get_dim(env)
    print("DIM_OBS:", DIM_OBS, "DIM_ACT:", DIM_ACT)
    pol = DQN(n_env=N_ENV,
              rnd=1,
              dim_obs=DIM_OBS,
              dim_act=DIM_ACT,
              discount=0.99,
              save_path=SAVE_PATH,
              save_model_freq=1000,
              log_freq=LOG_FREQ,
              update_target_freq=200,
              epsilon_schedule=lambda x: max(0.99 ** int(x/1e-5), 0.01),
              lr=1e-3)

    learn(env, pol)
