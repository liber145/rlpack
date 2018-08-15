import logging
import argparse
import gym
import numpy as np
from environment.atari_wrappers import make_deepmind_atari
from environment.scaler import Scaler
from middleware.mq import Client
from middleware.log import logger


class Packet(object):
    def __init__(self, identity, state, reward, done, trajectory, episode_reward, nstep):
        self.state = state
        self.reward = reward
        self.done = done
        self.trajectory = trajectory
        self.episode_reward = episode_reward
        self.id = identity
        self.nstep = nstep


class AtariEnv(Client):
    def __init__(self, identity, n_step, envname):
        super().__init__(identity, n_step)
        self.env = make_deepmind_atari(envname)

        print("In AtariEnv.")

        self.done = True
        self.nstep = 0

    def startsend(self, msg=None):
        if self._is_done() is True:
            self.newgame()
        else:
            self.trajectory = []

    def newgame(self):
        self.state = np.array(self.env.reset())
        self.reward = 0
        self.done = False
        self.trajectory = []
        self.laststate = self.state
        self.episode_reward = 0

    def _perform_action(self, action):
        self.nstep += 1
        self.state, self.reward, self.done, _ = self.env.step(action)
        self.state = np.array(self.state)
        self.trajectory.append(
            [self.laststate, action, self.reward, self.state, self.done])
        self.laststate = self.state

        self.episode_reward += self.reward

    def _get_packet(self):
        if self._check() is True:
            if self._is_done() is True:
                return Packet(self.id, self.state, self.reward, self.done, self.trajectory, self.episode_reward, self.nstep)
            else:
                return Packet(self.id, self.state, self.reward, self.done, self.trajectory, None, self.nstep)
        else:
            return Packet(self.id, self.state, self.reward, self.done, None, None, None)

    def _is_done(self):
        return self.done

    def _check(self):
        if self._is_done() is True:
            return True
        if self.cnt != 0 and self.cnt % self.n_step == 0:
            return True

        return False


class PoleEnv(AtariEnv):
    def __init__(self, identity, n_step, envname):
        super().__init__(identity, n_step, "PongNoFrameskip-v4")
        self.env = gym.make(envname)


class MujocoEnv(AtariEnv):
    def __init__(self, identity, n_step, envname):
        super().__init__(identity, n_step, "PongNoFrameskip-v4")
        self.env = gym.make(envname)
        self.scaler = Scaler(self.env.observation_space.shape[0])

    def _get_packet(self):
        if self._check() is True:

            if self._is_done() is True:
                return Packet(self.id, self.state, self.reward, self.done, self.trajectory, self.episode_reward,  self.nstep)
            else:
                return Packet(self.id, self.state, self.reward, self.done, self.trajectory, None, self.nstep)
        else:
            return Packet(self.id, self.state, self.reward, self.done, None, None, None)


class SimpleEnv(Client):
    def __init__(self, spec):
        super().__init__(spec)

        self.done = True
        self.nstep = 0
        self.episode_reward = 0

    def startsend(self, msg=None):
        if self._is_done() is True:
            self.state = np.array([1, 1, 1])
            self.reward = 1
            self.done = False
            self.trajectory = []
            self.laststate = self.state
        else:
            self.trajectory = []

    def _perform_action(self, action):

        self.nstep += 1
        self.state = np.array([1, 2, 3])
        self.reward = 10

        if self.nstep % 5 == 0:
            self.done = True
        else:
            self.done = False

        self.trajectory.append((self.laststate, action,
                                self.reward, self.state, self.done))
        self.laststate = self.state
        self.episode_reward += self.reward

    def _is_done(self):
        return self.done

    def _get_packet(self):
        if self._check() is True:
            if self._is_done() is True:
                return Packet(self.id, self.state, self.reward, self.done, self.trajectory, self.episode_reward, self.nstep)
            else:
                return Packet(self.id, self.state, self.reward, self.done, self.trajectory, None, self.nstep)
        else:
            return Packet(self.id, self.state, self.reward, self.done, None, None, None)

    def _check(self):
        if self._is_done() is True:
            return True
        if self.cnt != 0 and self.cnt % self.n_step == 0:
            return True

        return False
