# -*- coding: utf-8 -*-

import argparse
import time
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf

from rlpack.algos import SparseDQN
from rlpack.utils import mlp, discrete_sparse_policy, discrete_policy
from lineworld import LineWorld 

parser = argparse.ArgumentParser(description="set parameter.")
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--n_action", type=int, default=11)
parser.add_argument('--n_state',  type=int, default=100)
args = parser.parse_args()

print("----------------------------")
for arg in vars(args):
    print(f"{arg}: {getattr(args, arg)}")
print("----------------------------")


env = LineWorld(args.n_state, args.n_action)
epsilon = 0.1
dim_obs = (args.n_state,)
n_act = args.n_action
max_episode_len = env._max_episode_steps
max_epoch_step = max_episode_len * 8



class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, dim_obs, n_act, size):
        self.obs1_buf = np.zeros([size, *dim_obs], dtype=np.float32)
        self.obs2_buf = np.zeros([size, *dim_obs], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.int32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])



def value_fn(x):
    v = mlp(x, [64, 64, n_act], activation=tf.nn.relu)
    return v




def run_main():

    # agent = DQN(n_act=n_act, dim_obs=dim_obs, value_fn=value_fn, save_path=f"./log/classicalcontrol/{args.env}/dqn")
    # agent = DoubleDQN(n_act=n_act, dim_obs=dim_obs, value_fn=value_fn, save_path="./log/classicalcontrol/doubledqn")
    # agent = DuelDQN(n_act=n_act, dim_obs=dim_obs, value_fn=duel_value_fn, save_path="./log/classicalcontrol/dueldqn")
    # agent = DistDQN(n_act=n_act, dim_obs=dim_obs, policy_fn=distdqn_policy_fn, save_path="./log/classicalcontrol/distdqn")
    agent = SparseDQN(n_act=n_act, dim_obs=dim_obs, alpha=args.alpha, value_fn=value_fn, save_path=f"./log/gridworld/lineworld_{args.n_state}_{args.n_action}/sparsedqn_{args.alpha}_l{args.lr}")
    replay_buffer = ReplayBuffer(dim_obs=dim_obs, n_act=n_act, size=int(1000))

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(101):
        ep_ret_list, ep_len_list = [], []
        for t in range(max_epoch_step):
            a = agent.get_action(o[np.newaxis, :])[0]
            if np.random.rand() < epsilon:
                a = np.random.randint(n_act)

            nexto, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            replay_buffer.store(o, a, r, nexto, int(d))

            o = nexto

            terminal = d or (ep_len == max_episode_len)
            if terminal or (t == max_epoch_step-1):

                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                if terminal:
                    # 当到达完结状态或是最长状态时，记录结果
                    ep_ret_list.append(ep_ret)
                    ep_len_list.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        elapsed_time = time.time() - start_time
        print(f"{epoch}th epoch. time={elapsed_time}, average_return={np.mean(ep_ret_list)}, average_len={np.mean(ep_len_list)}")

        agent.add_scalar("average_return", np.mean(ep_ret_list), epoch*max_epoch_step)
        agent.add_scalar("average_length", np.mean(ep_len_list), epoch*max_epoch_step)

        for _ in range(int(max_epoch_step/512)):
            batch = replay_buffer.sample_batch(512)
            agent.update([batch["obs1"], batch["acts"], batch["rews"], batch["done"], batch["obs2"]])



if __name__ == "__main__":
    run_main()
