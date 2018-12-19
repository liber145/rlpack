import numpy as np
from rlpack.environment.env_wrapper import AtariWrapper, MujocoWrapper
from rlpack.common.memory import DiscreteActionMemory
from tqdm import tqdm

memory = DiscreteActionMemory(capacity=int(1e5), n_env=1, dim_obs=(84, 84, 4))

env = AtariWrapper("AlienNoFrameskip-v4", 1)
# env = MujocoWrapper("HalfCheetah-v2", 1)
obs = env.reset()
print(f"obs: {obs.shape}")


for i in range(1000000):
    act = env.sample_action()
    next_obs, rewards, dones, infos = env.step(env.sample_action())
    memory.store_sards(obs, act, rewards, dones, next_obs)
    obs = next_obs

    if "episode" in infos[0]:
        print(f"iter: {i}  episode: {infos[0]['episode']}")
