import argparse
import queue
import time
from collections import defaultdict, namedtuple
from multiprocessing import Process, Queue
import itertools

import gym
import numpy as np
import tensorflow as tf

from rlpack.algos import PPO
from rlpack.utils import mlp, mlp_gaussian_policy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env',  type=str, default="Reacher-v2")
args = parser.parse_args()


env = gym.make(args.env)

m = 20
n = 20
max_ep_len = env.spec.max_episode_steps

obs_dictqueue = defaultdict(Queue)
act_dictqueue = defaultdict(Queue)
episode_dictqueue = defaultdict(Queue)


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'early_stop', 'next_state'))


class Memory:
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))


class Env(Process):
    def __init__(self, obs_queue, act_queue, episode_queue):

        self._obs_queue = obs_queue
        self._act_queue = act_queue
        self._episode_queue = episode_queue

        self._env = args.env
        super().__init__()

    def run(self):
        env = gym.make(self._env)
        s = env.reset()
        memory, eplen = Memory(), 0
        while True:
            self._obs_queue.put(s)
            a = self._act_queue.get()

            nexts, r, d, _ = env.step(a)
            eplen += 1

            memory.push(s, a, r, int(d), int(eplen == max_ep_len), nexts)

            s = nexts

            terminal = d or (eplen == max_ep_len)
            if terminal:
                s = env.reset()
                if eplen == max_ep_len:
                    self._episode_queue.put(memory.memory)
                    memory, eplen = Memory(), 0


class Master:
    def __init__(self, obs_dictqueue, act_dictqueue, episode_dictqueue):
        self._obs_dictqueue = obs_dictqueue
        self._act_dictqueue = act_dictqueue
        self._episode_dictqueue = episode_dictqueue

        self._obs_list, self._id_list, self._ep_list = [], [], []

    def get_observation(self):
        cnt = 0
        while cnt < n:
            for i in range(m):
                try:
                    obs = self._obs_dictqueue[i].get(block=False)
                    cnt += 1

                    self._id_list.append(i)
                    self._obs_list.append(obs)

                except queue.Empty:
                    continue
            time.sleep(0.001)
        return self._obs_list

    def put_action(self, acts):
        for i in range(len(self._id_list)):
            self._act_dictqueue[self._id_list[i]].put(acts[i])
        self._obs_list, self._id_list = [], []

    def get_episode(self):
        for i in range(m):
            try:
                episode = self._episode_dictqueue[i].get(block=False)
                self._ep_list.append(episode)
            except queue.Empty:
                continue

        if len(self._ep_list) > 19:
            result = self._ep_list
            self._ep_list = []
            return result
        else:
            return None


def policy_fn(x, a):
    return mlp_gaussian_policy(x, a, hidden_sizes=[64, 64], activation=tf.tanh)


def value_fn(x):
    v = mlp(x, [64, 64, 1])
    return tf.squeeze(v, axis=1)


def process_episode(episode_list):
    number = len(episode_list)
    tot_episode_list = list(itertools.chain(*episode_list))
    databatch = [np.array(x) for x in zip(*tot_episode_list)]

    average_reward = np.sum(databatch[2]) / number
    average_length = len(databatch[2]) / number

    return databatch, average_reward, average_length


if __name__ == "__main__":
    for i in range(m):
        Env(obs_dictqueue[i], act_dictqueue[i], episode_dictqueue[i]).start()
    master = Master(obs_dictqueue, act_dictqueue, episode_dictqueue)

    env = gym.make(args.env)
    dim_obs = env.observation_space.shape[0]
    dim_act = env.action_space.shape[0]

    agent = PPO(dim_act=dim_act, dim_obs=dim_obs, policy_fn=policy_fn, value_fn=value_fn, save_path="./log/ppo")

    start_time = time.time()
    for i in range(50):

        for _ in range(env.spec.max_episode_steps):
            obs = master.get_observation()
            obs = np.array(obs)
            act = agent.get_action(obs)
            master.put_action(act)

            episode_list = master.get_episode()

            if episode_list:
                databatch, ave_reward, ave_length = process_episode(episode_list)
                agent.update(databatch)
                print(f"{i} ave_reward: {ave_reward}  ave_length:{ave_length}")

    elapsed_time = time.time() - start_time
    print("elapsed time:", elapsed_time)
