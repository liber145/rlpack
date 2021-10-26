import gym
import numpy as np

class MujocoControl(object):
    def __init__(self, env_name):
        assert env_name in {'Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2', 'Hopper-v1'}
        self.env = gym.make(env_name)
        self.use_cnn = False
        self._dim_obs = self.env.observation_space.shape[0]
        self._dim_act = self.env.action_space.shape[0]
        self._range_act = {'low':-1.0, 'high':1.0}
        self._traj_len = 0
        self._traj_rew = 0

    def reset(self):
        s = self.env.reset()
        return s 
    
    def step(self, action):
        s, r, d, info = self.env.step(action)

        self._traj_len += 1
        self._traj_rew += r 
        
        return s, r, d, info

    def sample_action(self):
        a = np.random.rand(e.action_space.shape[0])
        a = l + a*(h-l)
        return a
    
    def seed(self, rnd):
        self.env.seed(rnd)

    @property
    def dim_obs(self):
        return self._dim_obs

    @property
    def num_act(self):
        return self._num_act

def make_mujoco_control(env_name):
    env = MujocoControl(env_name)
    return env