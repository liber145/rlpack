import argparse
from collections import deque, Counter
import gym
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf

from rlpack.algos import AADQN
from rlpack.environment import make_ramatari


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--env", type=str, default="Pong-ramNoFrameskip-v4")
parser.add_argument("--niter", type=int, default=int(10e6))
parser.add_argument("--batchsize", type=int, default=32)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


env = make_ramatari(args.env)


class Memory(object):
    def __init__(self, capacity: int, dim_obs, dim_act, statetype=np.uint8):
        self._state = np.zeros((capacity, *dim_obs), dtype=statetype)
        self._action = np.zeros(capacity, dtype=np.int32)
        self._reward = np.zeros(capacity, dtype=np.float32)
        self._done = np.zeros(capacity, dtype=np.float32)
        self._next_state = np.zeros((capacity, *dim_obs), dtype=statetype)

        self._size = 0
        self._capacity = capacity

    def store_sards(self, state, action, reward, done, next_state):
        ind = self._size % self._capacity
        self._state[ind, ...] = state
        self._action[ind] = action
        self._reward[ind] = reward
        self._done[ind] = done
        self._next_state[ind, ...] = next_state
        self._size += 1

    def sample(self, n: int):
        n_sample = self._size if self._size < self._capacity else self._capacity
        inds = np.random.randint(n_sample, size=n)
        state_batch = self._state[inds, ...]
        action_batch = self._action[inds]
        reward_batch = self._reward[inds]
        done_batch = self._done[inds]
        next_state_batch = self._next_state[inds, ...]
        return state_batch, action_batch, reward_batch, done_batch, next_state_batch


def obs_fn():
    obs = tf.placeholder(shape=[None, 128, 4], dtype=tf.uint8, name="observation")
    obs = tf.to_float(obs) / 255.0
    return obs


def conv1d(obs):
    x = tf.layers.conv1d(obs, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=64, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=env.action_space.n)
    return x


def run_main():
    agent = AADQN(obs_fn=obs_fn,
                  value_fn=conv1d,
                  dim_act=env.action_space.n,
                  update_target_freq=10,
                  log_freq=100,
                  weight_low=-3,
                  weight_high=5,
                  save_path=f"./log/aadqn_ramatari/{args.env}",
                  lr=2.5e-4,
                  epsilon_schedule=lambda x: max(0.1, (1e6-x) / 1e6),
                  )
    mem = Memory(capacity=int(1e6), dim_obs=(128, 4), dim_act=env.action_space.n)
    sw = SummaryWriter(log_dir=f"./log/aadqn_ramatari/{args.env}")
    totrew, totlen, rewcnt = 0, 0, Counter()

    s = env.reset()
    for i in tqdm(range(args.niter)):
        a = agent.get_action(s[np.newaxis, :])[0]
        ns, r, d, _ = env.step(a)
        mem.store_sards(s, a, r, d, ns)
        s = ns

        totrew += r
        totlen += 1
        rewcnt.update([a])

        if i % 4 == 0:
            agent.update(mem.sample(args.batchsize))

        if d is True:
            s = env.reset()
            sw.add_scalars("aadqn", {"totrew": totrew, "totlen": totlen}, i)
            tqdm.write(f"{i}th. totrew={totrew}, totlen={totlen}, rewcnt={rewttt(rewcnt, env.action_space.n)}")
            totrew, totlen, rewcnt = 0, 0, Counter()


def rewttt(rewcnt, dim_act):
    t = ""
    for i in range(dim_act):
        t += f"{i}:{rewcnt[i]} "
    return t


if __name__ == "__main__":
    run_main()
