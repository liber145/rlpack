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
        """
        Step forever by reset at done.
        """
        state, reward, done, info = self.env.step(action)

        self.trajectory_length += 1
        self.trajectory_reward += reward

        if done or (self.trajectory_length == self.horizon_length):
            info["episode"] = {"r": self.trajectory_reward, "l": self.trajectory_length}
            """
            done is True when it terminates at artificial terminal or it is really done.
            If it terminates at artificial terminal, ignore the terminal by setting done to false.
            """
            done = False if self.trajectory_length == self.horizon_length else done

            state = self.env.reset()

            self.trajectory_reward = 0
            self.trajectory_length = 0

        return state, reward, done, info

    def reset(self):
        return self.env.reset()

    def seed(self, rnd):
        self.env.seed(rnd)

    def sample_action(self):
        return self.env.action_space.sample()

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
        """
        If there is artificial terminal horizon length, return it; 
        else, return -1.
        """
        name2len = {"Ant-v2": 1000, "HalfCheetah-v2": 1000, "Hopper-v2": -1, "Humanoid-v2": -1,
                    "HumanoidStandup-v2": 1000, "InvertedDoublePendulum-v2": -1, "InvertedPendulum-v2": -1,
                    "Reacher-v2": 50, "Swimmer-v2": 1000, "Walker2d-v2": -1}
        assert self.env_name in name2len
        return name2len[self.env_name]


def make_mujoco(env_name):
    env = LogTrajectory(env_name)
    return env


if __name__ == "__main__":
    env = make_mujoco("Ant-v2")
    s = env.reset()

    for i in range(10000):
        s, r, d, info = env.step(env.sample_action())
        if d is True:
            print(i)

        if "episode" in info:
            print("len:", info["episode"]["l"], "rew:", info["episode"]["r"])
