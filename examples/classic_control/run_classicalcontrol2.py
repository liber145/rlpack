# -*- coding: utf-8 -*-


import argparse
import time
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf

from rlpack.algos import TRPOAlpha, TRPO 
from rlpack.utils import mlp, discrete_sparse_policy, discrete_policy

parser = argparse.ArgumentParser(description='set parameter.')
parser.add_argument('--algo', type=str, default="trpo")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument('--env',  type=str, default="CartPole-v1")
args = parser.parse_args()

env = gym.make(args.env)
dim_obs = env.observation_space.shape
n_act = env.action_space.n
epsilon = 0.1
max_episode_len = env._max_episode_steps
max_epoch_step = max_episode_len * 8


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'early_stop', 'next_state'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))


def sparse_policy_fn(x, a):
    sampled_a, probs, p = discrete_sparse_policy(x, a, n_act, hidden_sizes=[64, 64], activation=tf.nn.relu, top=2)
    return sampled_a, probs, p 

def policy_fn(x, a):
    sampled_a, probs, p = discrete_policy(x, a, n_act, hidden_sizes=[64,64], activation=tf.nn.relu, top=2)
    return sampled_a, probs, p 

def value_fn(x):
    v = mlp(x, [64, 64, 1])
    return tf.squeeze(v)

def run_main():

    if args.algo == "trpoalpha":
        agent = TRPOAlpha(dim_obs=dim_obs, dim_act=n_act, alpha=args.alpha, policy_fn=sparse_policy_fn, value_fn=value_fn, save_path=f"./log/classicalcontrol/{args.env}/trpoalpha")
    elif args.algo == "trpo":    
        agent = TRPO(dim_obs=dim_obs, dim_act=n_act, is_discrete=True, policy_fn=policy_fn, value_fn=value_fn, save_path=f"./log/classicalcontrol/{args.env}/trpo")
    else:
        raise RuntimeError("unknown algorithm.")

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(101):
        memory, ep_ret_list, ep_len_list = Memory(), [], []
        for t in range(max_epoch_step):
            a = agent.get_action(o[np.newaxis, :])[0]
            nexto, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            memory.push(o, a, r, int(d), int(ep_len==max_episode_len or t==max_epoch_step-1), nexto)

            o = nexto

            terminal = d or (ep_len == max_episode_len)
            if terminal or (t == max_epoch_step-1):
                if not terminal:
                    print("Warning: trajectory cut off by epoch at %d steps." % ep_len)
                else:
                    ep_ret_list.append(ep_ret)
                    ep_len_list.append(ep_len)  
                o, ep_ret, ep_len = env.reset(), 0, 0

        print(f"{epoch}th epoch. average_return={np.mean(ep_ret_list)}, average_len={np.mean(ep_len_list)}")

        agent.add_scalar("average_return", np.mean(ep_ret_list), epoch*max_epoch_step)
        agent.add_scalar("average_length", np.mean(ep_len_list), epoch*max_epoch_step)

        # 更新策略。
        batch = memory.sample() 
        agent.update([np.array(x) for x in batch])

    elapsed_time = time.time() - start_time
    print("elapsed time:", elapsed_time)


if __name__ == "__main__":
    run_main()
