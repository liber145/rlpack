# -*- coding: utf-8 -*-


import argparse
import time
from collections import namedtuple

import gym
import numpy as np
import tensorflow as tf

from rlpack.algos import PPO, TRPO, TRPOAlpha
from rlpack.utils import mlp, mlp_gaussian_policy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--algo", type=str, default="trpo")
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument('--env',  type=str, default="Reacher-v2")
args = parser.parse_args()

env = gym.make(args.env)
dim_obs = env.observation_space.shape
dim_act = env.action_space.shape[0]
if args.env == "Reacher-v2":
    max_ep_len = 50 
    max_ep_step = 1000

else:
    max_ep_len = 1000
    max_ep_step = 1000 * 5


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done', 'early_stop', 'next_state'))


class Memory(object):
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self):
        return Transition(*zip(*self.memory))


def trpo_policy_fn(x, a):
    pi, logp, logp_pi, mu, log_std = mlp_gaussian_policy(x, a, hidden_sizes=[64, 64], activation=tf.tanh)
    return pi, logp, logp_pi, mu, log_std


# def ppo_policy_fn(x, a):
#     pi, logp, logp_pi, mu, log_std = mlp_gaussian_policy(x, a, hidden_sizes=[64, 64], activation=tf.tanh)
#     return pi, logp, logp_pi


def value_fn(x):
    v = mlp(x, [64, 64, 1])
    return tf.squeeze(v, axis=1)


def run_main():

    if args.algo == "trpo":
        agent = TRPO(dim_act=dim_act, dim_obs=dim_obs, policy_fn=trpo_policy_fn, value_fn=value_fn, delta=0.1, save_path=f"./log/mujoco/{args.env}/trpo")
    elif args.algo == "trpoalpha":
        agent = TRPOAlpha(dim_act=dim_act, dim_obs=dim_obs, alpha=args.alpha, policy_fn=trpo_policy_fn, value_fn=value_fn, delta=0.1, save_path=f"./log/mujoco/{args.env}/trpoalpha_{args.alpha}")
    # agent = PPO(dim_act=dim_act, dim_obs=dim_obs, policy_fn=ppo_policy_fn, value_fn=value_fn, save_path="./log/ppo")

    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(1000):
        memory, ep_ret_list, ep_len_list = Memory(), [], []
        for t in range(max_ep_step):
            a = agent.get_action(o[np.newaxis, :])[0]
            nexto, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            memory.push(o, a, r, int(d), int(ep_len == max_ep_len or t == max_ep_step-1), nexto)

            o = nexto

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t == max_ep_step-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                if terminal:
                    # 当到达完结状态或是最长状态时，记录结果
                    ep_ret_list.append(ep_ret)
                    ep_len_list.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        print(f"{epoch}th epoch. average_return={np.mean(ep_ret_list)}, average_len={np.mean(ep_len_list)}")

        agent.add_scalar("average_return", np.mean(ep_ret_list), epoch*max_ep_step)
        agent.add_scalar("average_length", np.mean(ep_len_list), epoch*max_ep_step)

        # 更新策略。
        batch = memory.sample()
        agent.update([np.array(x) for x in batch])

    elapsed_time = time.time() - start_time
    print("elapsed time:", elapsed_time)


if __name__ == "__main__":
    run_main()
