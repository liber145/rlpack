import gym
import numpy as np
from gym.spaces.box import Box
from collections import deque


def mujoco_env(env_id, stack=4):
    env = gym.make(env_id)
    env = FrameStackEnv(env, stack)
    env = ActionNormalizedEnv(env)

    return env


def get_mujoco_env_fn(env_id, stack=4):
    def env_fn():
        env = gym.make(env_id)
        env = FrameStackEnv(env, stack)
        env = ActionNormalizedEnv(env)
        return env

    return env_fn


class ActionNormalizedEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        action_space = self.env.action_space
        h, l, shp = action_space.high, action_space.low, action_space.shape
        self._action_center = (h + l) / 2
        self._action_scale = (h - l) / 2

        self.action_space = Box(
            low=-1.0, high=1.0, shape=shp, dtype=np.float32)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        ac = self._action_scale * action + self._action_center
        # assert self.env.action_space.contains(ac)
        return self.env.step(ac)


class FrameStackEnv(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last obs."""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.obs = deque([], maxlen=k)

        ob_space = self.env.observation_space
        self.observation_space = Box(
            low=np.tile(ob_space.low, k),
            high=np.tile(ob_space.high, k),
            dtype=np.float32)

    def reset(self, **kwargs):
        ob = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.obs.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.obs.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.obs) == self.k
        return np.concatenate(self.obs)
