import gym


class LogTrajectory(gym.Wrapper):
    def __init__(self, env):
        self.env = env
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

    @property
    def dim_observation(self):
        return self.env.observation_space.shape

    @property
    def dim_action(self):
        return self.env.action_space.shape

    @property
    def action_range(self):
        return self.env.action_space.low, self.env.action_space.high


def make_mujoco(env_name):
    env = gym.make(env_name)
    env = LogTrajectory(env)
    return env
