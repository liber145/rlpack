import numpy as np
import gym


class FakeSingleEnv(object):
    def __init__(self, dim_observation: list, n_action: int):
        self._dim_observation = dim_observation
        self._n_action = n_action

    def reset(self):
        return np.random.rand(*self.dim_observation)

    def step(self, action):
        ob = np.random.rand(*self.dim_observation)
        reward = np.random.rand(1)
        done = np.random.rand(1) > 0.5
        info_dict = dict()
        return ob, reward, done, info_dict

    def close(self):
        """Close environment."""

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def n_action(self):
        """
        :return: the dimension of continuous action
        """
        return self._n_action


class FakeBatchEnv(object):
    """
    一次reset，周而复始。下面是顺序形式，需要改装成并行化。
    """

    def __init__(self, envs: list):
        """
        :param dim_observation: dimension of observation
        :param n_action: number of action
        :param n_env: number of environments
        """
        self.envs = envs
        self.last_dones = [False for _ in range(self.n_env)]

        self._dim_observation = self.envs[0].dim_observation
        self._n_action = self.envs[0].n_action
        self._n_env = len(self.envs)

    def reset(self) -> np.ndarray:
        """
        :return: a batch of observations
        """
        obs = [env.reset() for env in self.envs]
        return np.asarray(obs)

    def step(self, actions):
        """
        :param actions: Actions for all environments. It is list or np.ndarray.
        :return: observation_batch, reward_batch, done_batch, info_list
        """
        if type(actions) is list:
            assert len(actions) == self.n_env
        if type(actions) is np.ndarray:
            assert actions.shape[0] == self.n_env
        assert actions[0] < self.n_action and actions[0] >= 0

        obs, rewards, dones, infos = [], [], [], []
        for i in range(self.n_env):
            if self.last_dones[i]:
                ob = self.envs[i].reset()
                obs.append(ob)
                rewards.append(0)
                dones.append(True)
                self.last_dones[i] = False
            else:
                ob, reward, done, info = self.envs[i].step(actions[i])
                obs.append(ob)
                rewards.append(reward)
                dones.append(done)
                info.append(info)
                self.last_dones[i] = done
        return np.asarray(obs), np.asarray(rewards), np.asarray(dones), infos

    def close(self):
        """Close all environments."""
        pass

    @property
    def n_env(self):
        return self._n_env

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def n_action(self):
        """
        :return: the dimension of continuous action
        """
        return self._n_action

    @property
    def n_env(self):
        return self._n_env
