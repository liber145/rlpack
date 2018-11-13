from rlpack.environment.env_wrapper import DistMujocoWrapper

env = DistMujocoWrapper("Reacher-v2", 4)
obs = env.reset()
print(f"obs: {obs}")
