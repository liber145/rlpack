import os.path as osp
import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines import bench, logger
from baselines.common.vec_env import VecEnv
import numpy as np
from gym import spaces


class FrameStack(VecEnv):
    """
    Vectorized environment base class
    """

    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space  # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
        self._observation_space = spaces.Box(low=low, high=high)
        self._action_space = venv.action_space

    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        # out = np.transpose(self.stackedobs, (0, 3, 1, 2))
        return self.stackedobs.astype(np.float32) / 255.0 , rews, news, infos
        # return out, rews, news, infos

    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs 
        # out = np.transpose(self.stackedobs, (0, 3, 1, 2))
        # return self.stackedobs
        return self.stackedobs.astype(np.float32) / 255.0

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def close(self):
        self.venv.close()

    @property
    def num_envs(self):
        return self.venv.num_envs


def make_env(rank, env_id):
    def env_fn():
        env = make_atari(env_id)
        env.seed(1 + rank)
        env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
        env = wrap_deepmind(env)
        return env
    return env_fn


class ScaledFloatFrame(gym.ObservationWrapper):
    def _observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class WrapPyTorch(gym.ObservationWrapper):
    def _observation(self, observation):
        return observation.transpose(2, 0, 1)
