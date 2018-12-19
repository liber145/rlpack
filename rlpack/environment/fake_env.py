import numpy as np
from .env_wrapper import AsyncEnvWrapper


class FakeDiscreteEnv(object):
    def __init__(self):
        self._dim_observation = (11,)
        self._dim_action = 2
        self.obs1 = np.array([-1 for i in range(11)])
        self.obs2 = np.array([0 for i in range(11)])
        self.cnt = 0

    def reset(self):
        self.cnt = 0
        return self.obs1

    def step(self, action):
        self.cnt += 1
        terminal = False if self.cnt < 50 else True 
        self.cnt = 0 if terminal else self.cnt

        if action == 0:
            return self.obs1, 1, terminal, dict()
        else:
            return self.obs2, 0, terminal, dict()

    def sample_action(self):
        return np.random.randint(self.dim_action)

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def dim_action(self):
        return self._dim_action


class FakeContinuousEnv(object):
    def __init__(self):
        self._dim_observation = (11,)
        self._dim_action = 3
        self._w = np.array([1, 1, 1])
        self.cnt = 0

    def reset(self):
        self.cnt = 0
        return np.zeros((11,))

    def step(self, action):
        self.cnt += 1
        if self.cnt < 50:
            terminal = False
        else:
            self.cnt = 0
            terminal = True

        if np.dot(action, self._w) > 0:
            obs = np.random.rand(11) + 1
            rew = 1
        else:
            obs = np.random.rand(11) - 1
            rew = 0

        return obs, rew, terminal, dict()

    def sample_action(self):
        """
        Sample #n actions each of which in {0, 1, 2, ..., dim_action-1}
        """
        return np.random.normal(size=(self.dim_action,))

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def dim_action(self):
        return self._dim_action


class AsyncFakeContinuousEnv(AsyncEnvWrapper):
    def __init__(self, n_env: int = 8, n_inference: int = None, port: int = 50000):
        super().__init__(n_env, n_inference, port)

    def _make_env(self):
        env = FakeContinuousEnv()
        return env


from rlpack.environment import AsyncEnvWrapper

class AsyncFakeDiscreteEnv(AsyncEnvWrapper):
    def __init__(self, n_env: int = 8, n_inference: int = None, port: int = 50000):
        super().__init__(n_env, n_inference, port)

    def _make_env(self):
        env = FakeDiscreteEnv()
        return env
