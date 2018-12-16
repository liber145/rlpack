import numpy as np
from rlpack.environment.env_wrapper import AtariWrapper, MujocoWrapper
from tqdm import tqdm

# env = AtariWrapper("AlienNoFrameskip-v4", 1)
env = MujocoWrapper("HalfCheetah-v2", 1)
obs = env.reset()
print(f"obs: {obs.shape}")

all_r = []
for i in range(10000):
    obs, rewards, dones, infos = env.step(env.sample_action())
    all_r.append(rewards)
    if dones[0]:
        print(f"iter {i} -- r max: {np.max(all_r)} min: {np.min(all_r)} {obs.shape}")

    if "episode" in infos[0]:
        print(infos[0]["episode"])
