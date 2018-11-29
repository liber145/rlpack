import os
from multiprocessing import Process
from typing import List

import gym
import numpy as np
from gym import spaces

from .atari_wrappers import make_atari, wrap_deepmind
from .distributed_env_worker import DistributedEnvClient
from .distributed_env_wrapper import DistributedEnvManager
from .stack_env import StackEnv

# class MujocoWrapper(StackEnv):
#     def __init__(self, env_id: str, n_env: int):
#         super().__init__(env_id, n_env)
#         self.trajectory_rewards = [0 for _ in range(self.n_env)]
#         self.trajectory_length = [0 for _ in range(self.n_env)]
#         self._dim_observation = self.envs[0].observation_space.shape
#         self._dim_action = self.envs[0].action_space.shape[0]
#
#     def step(self, actions):
#         obs, rewards, dones, _ = super().step(actions)
#         epinfos = []
#         for i in range(self.n_env):
#             if self.last_dones[i]:
#                 epinfos.append({"episode": {"r": self.trajectory_rewards[i], "l": self.trajectory_length[i]}})
#                 self.trajectory_length[i] = 0
#                 self.trajectory_rewards[i] = 0
#             else:
#                 self.trajectory_length[i] += 1
#                 self.trajectory_rewards[i] += rewards[i]
#         return obs, rewards, dones, epinfos
#
#     @property
#     def dim_observation(self):
#         return self._dim_observation
#
#     @property
#     def dim_action(self):
#         return self._dim_action
#
#     @property
#     def is_continuous(self):
#         return True


class DistributedMujocoWrapper(object):
    def __init__(self, env_name: str, n_env: int):
        self.n_env = n_env
        self.env_ids = None
        self.env_manager = DistributedEnvManager(n_env)
        self.env_manager.configure()
        # p = Process(target=self.env_manager.start)
        # p.start()
        # p.join()
        self.env_manager.start()

        processes = []
        for i in range(n_env):
            p = DistributedEnvClient(self._make_env(i, env_name))
            p.daemon = True
            p.start()
            processes.append(p)

        self._dim_action = p.dim_action
        self._dim_observation = p.dim_observation

    def _make_env(self, rank, env_name):
        env = gym.make(env_name)
        env.seed(1 + rank)
        return env

    def step(self, actions: List):
        act_dict = {env_id: act for env_id, act in zip(self.env_ids, actions)}
        self.env_manager.step(act_dict)
        self.env_ids, obs, rewards, dones, infos = self.env_manager.get_envs_to_inference(n=self.n_env)
        return np.asarray(obs, dtype=np.float32), np.asarray(rewards, dtype=np.float32), np.asarray(dones, dtype=np.float32), infos

    def reset(self):
        self.env_ids, states = self.env_manager.get_envs_to_inference(n=self.n_env, state_only=True)
        return np.asarray(states, dtype=np.float32)

    @property
    def dim_observation(self):
        return self._dim_observation

    @property
    def dim_action(self):
        return self._dim_action

    @property
    def env_id(self):
        return self.env_ids


class DistributedAtariWrapper(object):
    def __init__(self, env_name: str, n_env: int = 8):
        self.n_env = n_env
        self.env_ids = None
        self.env_manager = DistributedEnvManager(n_env)
        self.env_manager.configure()
        self.env_manager.start()

        processes = []
        for i in range(n_env):
            p = DistributedEnvClient(self._make_env(i, env_name))
            p.daemon = True
            p.start()
            processes.append(p)

    def _make_env(self, rank, env_name):
        env = make_atari(env_name)
        env.seed(1 + rank)
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        return env

    def step(self, actions: List):
        act_dict = {env_id: act for env_id, act in zip(self.env_ids, actions)}
        self.env_manager.step(act_dict)
        self.env_ids, obs, rewards, dones, infos = self.env_manager.get_envs_to_inference(n=self.n_env)
        return np.asarray(obs, dtype=np.float32), np.asarray(rewards, dtype=np.float32), dones, infos

    def reset(self):
        self.env_ids, states = self.env_manager.get_envs_to_inference(n=self.n_env, state_only=True)
        return np.asarray(states, dtype=np.float32)

    @property
    def env_id(self):
        return self.env_ids


class CartpoleWrapper(StackEnv):
    def __init__(self, n_env):
        super().__init__("CartPole-v1", n_env)
        self.trajectory_rewards = [0 for _ in range(self.n_env)]
        self.trajectory_length = [0 for _ in range(self.n_env)]

        self._dim_observation = self.envs[0].observation_space.shape
        self._n_action = self.envs[0].action_space.n

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
    def dim_observation(self):
        return self._dim_observation

    @property
    def n_action(self):
        return self._n_action


class AtariWrapper2(StackEnv):
    def __init__(self, env_id: str, n_env: int):
        super().__init__(env_id, n_env)
        self.envs = [self._make_env(1 + i, env_id) for i in range(n_env)]
        self.trajectory_length = [0 for i in range(n_env)]
        self.trajectory_rewards = [0 for i in range(n_env)]

    def _make_env(self, rank, env_id):
        env = make_atari(env_id)
        env.seed(1 + rank)
        env = wrap_deepmind(env, frame_stack=True, scale=True)
        return env

    def step(self, actions):
        obs, rewards, dones, infos = super().step(actions)
        epinfos = []
        for i in range(self.n_env):
            if infos[i]["real_done"]:
                epinfos.append({"episode": {"r": self.trajectory_rewards[i], "l": self.trajectory_length[i]}})
                self.trajectory_length[i] = 0
                self.trajectory_rewards[i] = 0
            else:
                self.trajectory_length[i] += 1
                self.trajectory_rewards[i] += infos[i]["real_reward"]
        return obs, rewards, dones, epinfos


# class AtariWrapper(object):
#     def __init__(self, env_id: str, n_env: int):
#         self.env = SubprocVecEnv([self._make_env(i, env_id) for i in range(n_env)])
#         wos = self.env.observation_space
#         low = np.repeat(wos.low, 4, axis=-1)
#         high = np.repeat(wos.high, 4, axis=-1)
#         self.stackedobs = np.zeros((n_env,) + low.shape, low.dtype)
#         self._observation_space = spaces.Box(low=low, high=high)
#         self._action_space = self.env.action_space
#
#     def _make_env(self, rank, env_id):
#         def env_fn():
#             env = make_atari(env_id)
#             env.seed(1 + rank)
#             env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
#             env = wrap_deepmind(env)
#             return env
#         return env_fn
#
#     def step(self, actions):
#         obs, rewards, dones, infos = self.env.step(actions)
#         self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
#         for (i, done) in enumerate(dones):
#             if done:
#                 self.stackedobs[i] = 0
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs.astype(np.float32) / 255.0, rewards, dones, infos
#
#     def reset(self):
#         obs = self.env.reset()
#         self.stackedobs[...] = 0
#         self.stackedobs[..., -obs.shape[-1]:] = obs
#         return self.stackedobs.astype(np.float32) / 255.0
#
#     def close(self):
#         self.env.close()
#
#     @property
#     def action_space(self):
#         return self._action_space
#
#     @property
#     def observation_space(self):
#         return self._observation_space
#
#     @property
#     def is_continuous(self):
#         return False

if __name__ == "__main__":
    env = DistributedAtariWrapper("BreakoutNoFrameskip-v4", 4)
    obs = env.reset()
    print(f"obs: {obs.shape}")
    for _ in range(1000):
        obs, rewards, dones, infos = env.step(np.random.randint(4, size=4))
        print(f"obs: {obs.shape} max: {np.max(obs)} min: {np.min(obs)}  rewards: {rewards.shape}  dones: {dones}  infos: {infos}")
