import numpy as np
from rlpack.environment.env_wrapper import DistributedMujocoWrapper
from tqdm import tqdm

env = DistributedMujocoWrapper("Reacher-v2", 2)
obs = env.reset()
print(f"obs: {obs.shape}  {env.dim_observation}  {env.dim_action}")


for _ in tqdm(range(1000)):
    obs, rewards, dones, infos = env.step([np.random.randint(4) for _ in range(2)])
