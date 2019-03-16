import os
from abc import ABC, abstractmethod
from multiprocessing import Process
from typing import Callable, List

import numpy as np

from .atari_wrappers import make_atari
from .distributed_env_worker import DistributedEnvClient
from .distributed_env_wrapper import DistributedEnvManager
from .mujoco_wrappers import make_mujoco
from .classical_control_wrapper import make_classic_control


class StackEnv(object):
    """
    Stack several environments.
    """

    def __init__(self, env_func: Callable, n_env: int = 1):
        self.envs = [env_func(i) for i in range(n_env)]
        self._n_env = n_env
        self._dim_obs = self.envs[0].dim_observation
        self._dim_act = self.envs[0].dim_action

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

    def sample_action(self):
        return np.asarray([env.sample_action() for env in self.envs])

    @property
    def n_env(self):
        return self._n_env

    @property
    def dim_observation(self):
        return self._dim_obs

    @property
    def dim_action(self):
        """
        The dimension of continuous action, or the number of discrete action.
        """
        return self._dim_act


class MujocoWrapper(StackEnv):
    def __init__(self, env_id: str, n_env: int = 1):
        self._n_env = n_env
        super().__init__(lambda x: self._make_env(env_id, x), n_env)
        # self._dim_observation = self.envs[0].dim_observation
        # self._dim_action = self.envs[0].dim_action

    def _make_env(self, env_id: str, rank: int = 0):
        env = make_mujoco(env_id)
        env.seed(rank + 1)
        return env

    @property
    def horizon_length(self):
        """
        In some environments, there is an artifical terminal length.
        If it is that case, return the length; else return -1.
        """
        return self.envs[0].horizon_length


class AtariWrapper(StackEnv):
    def __init__(self, env_id: str, n_env: int = 1):
        self._n_env = n_env
        super().__init__(lambda x: self._make_env(env_id, x), n_env)

    def _make_env(self, env_id: str, rank: int = 0):
        if "ramNoFrameskip" in env_id:
            env = make_ram_atari(env_id)
        else:
            env = make_atari(env_id)
        env.seed(rank + 1)
        return env


class ClassicControlWrapper(StackEnv):
    def __init__(self, env_name: str, n_env: int = 1):
        self._n_env = n_env
        super().__init__(lambda x: self._make_env(env_name, x), n_env)

    def _make_env(self, env_name: str, rank: int = 0):
        env = make_classic_control(env_name)
        env.seed(rank + 1)
        return env


class AsyncMujocoWrapper(object):
    def __init__(self, env_name: str, n_env: int = 8, n_inference: int = None, port: int = 50000):
        self.n_env = n_env
        self.n_inference = n_env if n_inference is None else n_inference
        self._env_ids = None
        self.env_manager = DistributedEnvManager(n_env, port=port)
        self.env_manager.configure()
        self.env_manager.start()

        processes = []
        for i in range(n_env):
            p = DistributedEnvClient(self._make_env(env_name, i), port=port)
            p.daemon = True
            p.start()
            processes.append(p)

        self.env_name = env_name
        self._dim_action = p.dim_action
        self._dim_observation = p.dim_observation
        self._horizon_length = p.env.horizon_length

    def _make_env(self, env_name, rank=0):
        env = make_mujoco(env_name)
        env.seed(1 + rank)
        return env

    def sample_action(self, n):
        return np.random.uniform(low=self.action_range[0], high=self.action_range[1], size=(n, self.dim_action))

    def step(self, actions: List):
        """Forward one step according to the given actions.

        Parameters:
            - actions: a list of actions.

        Returns:
            - states (np.ndarray): (n_inference, state_dimension)
            - rewards (np.ndarray): (n_inference)
            - dones (np.ndarray): (n_inference)
            - infos
        """
        act_dict = {env_id: act for env_id, act in zip(self._env_ids, actions)}
        self.env_manager.step(act_dict)
        self._env_ids, obs, rewards, dones, infos = self.env_manager.get_envs_to_inference(n=self.n_inference)
        return np.asarray(obs, dtype=np.float32), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=np.float32), infos

    def reset(self):
        """Reset environment."""
        self._env_ids, states = self.env_manager.get_envs_to_inference(n=self.n_env, state_only=True)
        return np.asarray(states, dtype=np.float32)

    @property
    def dim_observation(self):
        """The dimension of observation."""
        return self._dim_observation

    @property
    def dim_action(self):
        """The dimension of action."""
        return self._dim_action

    @property
    def horizon_length(self):
        """
        In some environments, there is an artifical terminal length.
        If it is that case, return the length; else return -1.
        """
        return self._horizon_length

    @property
    def action_range(self):
        name2act_range = {"Ant-v2": (-1, 1), "HalfCheetah-v2": (-1, 1), "Hopper-v2": (-1, 1), "Humanoid-v2": (-0.4, 0.4),
                          "HumanoidStandup-v2": (-0.4, 0.4), "InvertedDoublePendulum-v2": (-1, 1), "InvertedPendulum-v2": (-3, 3),
                          "Reacher-v2": (-1, 1), "Swimmer-v2": (-1, 1), "Walker2d-v2": (-1, 1)}
        return name2act_range[self.env_name]

    @property
    def env_id(self):
        """The ID of environment."""
        return self._env_ids


class AsyncAtariWrapper(object):
    def __init__(self, env_name: str, n_env: int = 4, n_inference: int = 4, port=50000):
        self.n_env = n_env
        self.n_inference = n_inference
        self.env_ids = None
        self.env_manager = DistributedEnvManager(n_env, port=port)
        self.env_manager.configure()
        self.env_manager.start()

        processes = []
        for i in range(n_env):
            p = DistributedEnvClient(self._make_env(env_name, i), port=port)
            p.daemon = True
            p.start()
            processes.append(p)

        self._dim_observation = p.dim_observation
        self._dim_action = p.dim_action

    def _make_env(self, env_name, rank=0):
        if "ramNoFrameskip" in env_name:
            env = make_ram_atari(env_name)
        else:
            env = make_atari(env_name)
        env.seed(1 + rank)
        return env

    def step(self, actions: List):
        """Forward one step according to the given actions.

        Parameters:
            - actions: a list of actions.

        Returns:
            - states (np.ndarray): (n_inference, state_dimension)
            - rewards (np.ndarray): (n_inference)
            - dones (np.ndarray): (n_inference)
            - infos
        """
        act_dict = {env_id: act for env_id, act in zip(self.env_ids, actions)}
        self.env_manager.step(act_dict)
        self.env_ids, obs, rewards, dones, infos = self.env_manager.get_envs_to_inference(n=self.n_inference)
        return np.asarray(obs, dtype=np.float32), np.asarray(rewards, dtype=np.float32), dones, infos

    def sample_action(self, n):
        return np.random.randint(self.dim_action, size=n)

    def reset(self):
        """Reset the environment."""
        self.env_ids, states = self.env_manager.get_envs_to_inference(n=self.n_env, state_only=True)
        return np.asarray(states, dtype=np.float32)

    @property
    def env_id(self):
        """The id of environment."""
        return self.env_ids

    @property
    def dim_observation(self):
        """The dimension of observation."""
        return self._dim_observation

    @property
    def dim_action(self):
        """The number of action."""
        return self._dim_action


class AsyncEnvWrapper(ABC):
    def __init__(self, n_env: int, n_inference: int, port=50000):
        self.n_env = n_env
        self.n_inference = n_inference
        self.env_ids = None

        self.env_manager = DistributedEnvManager(n_env, port=port)
        self.env_manager.configure()
        self.env_manager.start()

        for i in range(n_env):
            p = DistributedEnvClient(self._make_env(), port=port)
            p.daemon = True
            p.start()

        self._dim_observation = p.env.dim_observation
        self._dim_action = p.env.dim_action
        self._sample_action = p.env.sample_action

    @abstractmethod
    def _make_env(self):
        raise NotImplementedError("To be implemented.")

    def reset(self):
        self.env_ids, states = self.env_manager.get_envs_to_inference(n=self.n_env, state_only=True)
        return np.asarray(states)

    def step(self, actions):
        act_dict = {e_id: act for e_id, act in zip(self.env_ids, actions)}
        self.env_manager.step(act_dict)
        self.env_ids, obs, rews, dones, infos = self.env_manager.get_envs_to_inference(n=self.n_inference)
        return np.asarray(obs), np.asarray(rews), dones, infos

    def sample_action(self, n):
        return np.asarray([self._sample_action() for _ in range(n)])

    @property
    def dim_action(self):
        return self._dim_action

    @property
    def dim_observation(self):
        return self._dim_observation


if __name__ == "__main__":
    env = AtariWrapper("AlienNoFrameskip-v4", 1)
    obs = env.reset()
    print(f"obs: {obs.shape}")

    all_r = []
    for i in range(10000):
        obs, rewards, dones, infos = env.step(env.sample_action())
        all_r.append(rewards)
        if dones[0]:
            print(f"iter {i} -- r max: {np.max(all_r)} min: {np.min(all_r)} {infos[0]['episode']}")
