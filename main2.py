import gym
import numpy as np
from environment.atari_wrapper import get_atari_env_fn
from environment.master_env import EnvMaker

env_fn = get_atari_env_fn('BeamRiderNoFrameskip-v4')
num_agents = 8

env = EnvMaker(num_agents=num_agents, env_fn=env_fn, basename='test')

print('action_space')
print(env.action_space.nvec.dtype, type(env.action_space))

print('observation_space')
assert env.observation_space.shape == (num_agents, 84, 84, 4)
print(env.observation_space.high[0, 0, 0, 0],
      env.observation_space.low[0, 0, 0, 0],
      env.observation_space.shape,
      env.observation_space.dtype)

assert env.reset().shape == (num_agents, 84, 84, 4)
s, r, d, info = env.step(env.action_space.sample())
assert s.shape == (num_agents, 84, 84, 4)
assert r.shape == (num_agents,)
assert d.shape == (num_agents,) and d.dtype == np.bool
print(info)

env.close()

env = EnvMaker(num_agents=num_agents, env_fn=env_fn, basename='test')
env.reset()

reward = np.zeros(num_agents)
for i in range(1000):
    s, r, d, info = env.step(env.action_space.sample())
    reward += r
    if d.any():
        print(reward[d])
        print(info)
        reward[d] = 0.0

env.close()
