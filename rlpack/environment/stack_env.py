import numpy as np
from typing import List, Callable


class StackEnv(object):
    """
    Stack several environments.
    """

    def __init__(self, env_func: Callable, n_env: int):
        self.envs = [env_func(i) for i in range(n_env)]

    def reset(self) -> np.ndarray:
        """
        :return: a batch of observations
        """
        obs = []
        for i in range(self.n_env):
            ob = self.envs[i].reset()
            obs.append(ob)
        return np.asarray(obs)

    def step(self, actions: List):
        next_obs, rewards, dones, infos = [], [], [], []
        for i in range(self.n_env):
            obs, rew, done, info = self.envs[i].step(actions[i])

            next_obs.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        return np.asarray(next_obs), np.asarray(rewards), np.asarray(dones), infos

    def close(self):
        for i in range(len(self.n_env)):
            self.envs[i].close()

    @property
    def n_env(self):
        return len(self.envs)

    @property
    def dim_observation(self):
        pass

    @property
    def dim_action(self):
        """
        :return: the dimension of continuous action, or the number of discrete action.
        """
        pass
