"""
Environment wrapper for gym.mujoco.
"""
import gym


class LogTrajectory(object):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.trajectory_length = 0
        self.trajectory_reward = 0

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        self.trajectory_length += 1
        self.trajectory_reward += reward
        if done:
            info["episode"] = {"r": self.trajectory_reward, "l": self.trajectory_length}
            self.trajectory_reward = 0
            self.trajectory_length = 0

        return state, reward, done, info

    def reset(self):
        return self.env.reset()

    def seed(self, rnd):
        self.env.seed(rnd)

    @property
    def dim_observation(self):
        return self.env.observation_space.shape

    @property
    def dim_action(self):
        return self.env.action_space.shape[0]

    @property
    def action_range(self):
        return self.env.action_space.low[0], self.env.action_space.high[0]

    @property
    def horizon_length(self):
        pass


def make_mujoco(env_name):
    env = LogTrajectory(env_name)
    return env
