import argparse
from collections import deque
import gym
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from rlpack.algos import AADQN
from rlpack.environment import make_ramatari


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--env", type=str, default="CartPole-v1")
parser.add_argument("--niter", type=int, default=int(2e4))
parser.add_argument("--batchsize", type=int, default=128)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


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


def run_main():
    env = gym.make(args.env)
    agent = AADQN(dim_obs=env.observation_space.shape,
                  dim_act=env.action_space.n,
                  update_target_freq=100,
                  log_freq=10,
                  weight_low=0,
                  weight_high=1,
                  save_path=f"./log/aadqn_cc/{args.env}",
                  lr=1e-4
                  )
    mem = Memory(capacity=int(1e5), dim_obs=env.observation_space.shape, dim_act=env.action_space.n)
    sw = SummaryWriter(log_dir=f"./log/aadqn_cc/{args.env}")
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
            sw.add_scalars("aadqn", {"totrew": totrew, "totlen": totlen}, i)
            tqdm.write(f"{i}th. totrew={totrew}, totlen={totlen}")
            totrew, totlen = 0, 0


if __name__ == "__main__":
    run_main()
