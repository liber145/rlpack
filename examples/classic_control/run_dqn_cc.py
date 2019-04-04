import argparse
from collections import deque, Counter
import gym
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf

from rlpack.algos import DQN


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--env", type=str, default="CartPole-v1")
parser.add_argument("--niter", type=int, default=int(2e4))
parser.add_argument("--batchsize", type=int, default=128)
args = parser.parse_args()

env = gym.make(args.env)


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
        extra = np.zeros(128, dtype=np.float32)
        return (state_batch, extra), action_batch, reward_batch, done_batch, (next_state_batch, extra)


def obs_fn():
    obs = tf.placeholder(shape=[None, *env.observation_space.shape], dtype=tf.float32, name="observation")
    extra = tf.placeholder(tf.float32, [None], "extra")
    return (obs, extra)


def value_fn(inputs):
    obs = inputs[0]
    x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
    x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    x = tf.layers.dense(x, env.action_space.n)
    return x


def run_main():
    agent = DQN(obs_fn=obs_fn,
                value_fn=value_fn,
                dim_act=env.action_space.n,
                update_target_freq=100,
                log_freq=10,
                save_path=f"./log/dqn_cc/{args.env}",
                lr=1e-4,
                epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                train_epoch=1)
    mem = Memory(capacity=int(1e5), dim_obs=env.observation_space.shape, dim_act=env.action_space.n)
    sw = SummaryWriter(log_dir=f"./log/dqn_cc/{args.env}")
    totrew, totlen, rewcnt = 0, 0, Counter()

    s = env.reset()
    for i in tqdm(range(args.niter)):
        a = agent.get_action((s[np.newaxis, :], np.zeros(1)))[0]
        ns, r, d, _ = env.step(a)
        mem.store_sards(s, a, r, d, ns)
        s = ns

        totrew += r
        totlen += 1
        rewcnt.update([a])

        agent.update(mem.sample(args.batchsize))

        if d is True:
            s = env.reset()
            sw.add_scalars("dqn", {"totrew": totrew, "totlen": totlen}, i)

            tqdm.write(f"{i}th. totrew={totrew}, totlen={totlen}, rewcnt={rewttt(rewcnt, env.action_space.n)}")
            totrew, totlen, rewcnt = 0, 0, Counter()


def rewttt(rewcnt, dim_act):
    t = ""
    for i in range(dim_act):
        t += f"{i}:{rewcnt[i]} "
    return t


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
