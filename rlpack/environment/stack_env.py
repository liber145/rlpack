import numpy as np
from typing import List, Callable


class StackEnv(object):
    """
    一次reset，周而复始。
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

            if done:
                obs = self.envs[i].reset()

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


if __name__ == "__main__":
    pass
    # envs = [gym.make("CartPole-v1") for _ in range(2)]
    # stack_env = StackEnv(envs)

    # obs = stack_env.reset()
    # print(f"obs: {obs}")

    # next_obs, rewards, dones, infos = stack_env.step(np.array([0, 0]))
    # print(f"next obs: {next_obs}")
    # print(f"rewards: {rewards}")
    # print(f"dones: {dones}")
    # print(f"infos: {infos}")

    # next_obs, rewards, dones, infos = stack_env.step(np.asarray([0, 0]))
    # print(f"next obs: {next_obs}")
    # print(f"rewards: {rewards}")
    # print(f"dones: {dones}")
    # print(f"infos: {infos}")

    # env = MujocoWrapper([gym.make("Reacher-v2") for _ in range(2)])
    # obs = env.reset()
    # print(f"obs: {obs}")

    # for _ in range(10000):
    #     obs, rewards, dones, epinfos = env.step([np.array([0, 0]), np.array([1, 1])])
    #     print(f"obs: {obs.shape}")
    #     print(f"rewards: {rewards}")
    #     print(f"dones: {dones}")
    #     print(f"epinfos: {epinfos}")
    #     if dones[0] or dones[1]:
    #         input()
