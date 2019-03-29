import argparse
from collections import deque
import gym
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf

from rlpack.algos import DuelDQN
from rlpack.environment import make_ramatari


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--env", type=str, default="Pong-ramNoFrameskip-v4")
parser.add_argument("--niter", type=int, default=int(2e4))
parser.add_argument("--batchsize", type=int, default=128)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


env = make_ramatari(args.env)


class Memory(object):
    def __init__(self, capacity: int, dim_obs, dim_act, statetype=np.float32):
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


def value_fn(obs):
    x = tf.layers.conv1d(obs, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    return tf.layers.dense(x, 1), tf.layers.dense(x, units=env.action_space.n)


def run_main():
    agent = DuelDQN(obs_fn=obs_fn,
                    value_fn=value_fn,
                    dim_act=env.action_space.n,
                    update_target_freq=10000,
                    log_freq=10,
                    save_path=f"./log/dueldqn_cc/{args.env}",
                    lr=1e-4,
                    train_epoch=1)
    mem = Memory(capacity=int(1e6), dim_obs=(128, 4), dim_act=env.action_space.n)
    sw = SummaryWriter(log_dir=f"./log/dueldqn_cc/{args.env}")
    totrew, totlen = 0, 0

    s = env.reset()
    for i in tqdm(range(args.niter)):
        a = agent.get_action(s[np.newaxis, :])[0]
        ns, r, d, _ = env.step(a)
        mem.store_sards(s, a, r, d, ns)
        s = ns

        totrew += r
        totlen += 1

        agent.update(mem.sample(args.batchsize))

        if d is True:
            s = env.reset()
            sw.add_scalars("dueldqn", {"totrew": totrew, "totlen": totlen}, i)
            tqdm.write(f"{i}th. totrew={totrew}, totlen={totlen}")
            totrew, totlen = 0, 0


def run_game():
    env = gym.make(args.env)
    s = env.reset()
    totrew = 0
    for i in range(100):
        a = np.random.randint(2)
        ns, r, d, _ = env.step(a)
        ns = s
        totrew += r
        if d is True:
            s = env.reset()


if __name__ == "__main__":
    run_main()
