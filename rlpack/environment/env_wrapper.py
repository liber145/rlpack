import os 
from .stack_env import StackEnv
import numpy as np

from baselines import bench, logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env import VecEnv
from gym import spaces


class MujocoWrapper(StackEnv):
    def __init__(self, env_id: str, n_env: int):
        super().__init__(env_id, n_env)
        self.trajectory_rewards = [0 for _ in range(self.n_env)]
        self.trajectory_length = [0 for _ in range(self.n_env)]
        self._dim_observation = self.envs[0].observation_space.shape
        self._dim_action = self.envs[0].action_space.shape

    def step(self, actions):
        obs, rewards, dones, _ = super().step(actions)
        epinfos = []
        for i in range(self.n_env):
            if self.last_dones[i]:
                epinfos.append({"episode": {"r": self.trajectory_rewards[i], "l": self.trajectory_length[i]}})
                self.trajectory_length[i] = 0
                self.trajectory_rewards[i] = 0
            else:
                self.trajectory_length[i] += 1
                self.trajectory_rewards[i] += rewards[i]
        return obs, rewards, dones, epinfos

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def dim_action(self):
        return self._dim_action

    @property
    def is_continuous(self):
        return True

class CartpoleWrapper(StackEnv):
    def __init__(self, n_env):
        super().__init__("CartPole-v1", n_env)
        self.trajectory_rewards = [0 for _ in range(self.n_env)]
        self.trajectory_length = [0 for _ in range(self.n_env)]

    def step(self, actions):
        obs, rewards, dones, _ = super().step(actions)
        epinfos = []
        for i in range(self.n_env):
            self.trajectory_length[i] += 1
            self.trajectory_rewards[i] += rewards[i]

            if self.last_dones[i]:
                epinfos.append({"episode": {"r": self.trajectory_rewards[i], "l": self.trajectory_length[i]}})
                self.trajectory_length[i] = 0
                self.trajectory_rewards[i] = 0
                rewards[i] = -1
            else:
                rewards[i] = 0.1
        return obs, rewards, dones, epinfos

    @property 
    def is_continuous(self):
        return False


class AtariWrapper(object):
    def __init__(self, env_id: str, n_env: int):
        self.env = SubprocVecEnv([self._make_env(i, env_id) for i in range(n_env)])
        wos = self.env.observation_space
        low = np.repeat(wos.low, 4, axis=-1)
        high = np.repeat(wos.high, 4, axis=-1)
        self.stackedobs = np.zeros((n_env,)+low.shape, low.dtype)
        self._observation_space = spaces.Box(low=low, high=high)
        self._action_space = self.env.action_space

    def _make_env(self, rank, env_id):
        def env_fn():
            env = make_atari(env_id)
            env.seed(1 + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            env = wrap_deepmind(env)
            return env 
        return env_fn

    def step(self, actions):
        obs, rewards, dones, infos = self.env.step(actions)
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, done) in enumerate(dones):
            if done:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs.astype(np.float32) / 255.0, rewards, dones, infos

    def reset(self):
        obs = self.env.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs 
        return self.stackedobs.astype(np.float32) / 255.0

    def close(self):
        self.env.close()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space 

    @property
    def is_continuous(self):
        return False



if __name__ == "__main__":
    env = AtariWrapper("AlienNoFrameskip-v4", 4)
    obs = env.reset()
    print(f"obs: {obs}")
    obs, rewards, dones, infos = env.step([0, 0, 0, 0])
    print(f"obs: {obs}  rewards: {rewards}  dones: {dones}  infos: {infos}")
