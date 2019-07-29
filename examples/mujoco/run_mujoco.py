# -*- coding: utf-8 -*-


import argparse
import os
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from rlpack.algos import PPO, ContinuousPPO

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env',  type=str, default="Reacher-v2")
args = parser.parse_args()


env = gym.make(args.env)
action_dim = env.action_space.shape[0]


MAXITER = 1000


def trajectory(env, agent):
    t = deque()
    s = env.reset()
    tsum = 0
    while True:
        a = agent.get_action(s[np.newaxis, :])[0]
        ns, r, d, _ = env.step(a)
        t.append((s, a, r))
        s = ns
        tsum += r
        if d is True:
            break
    return t, tsum


def policy_fn(obs):
    x = tf.layers.dense(obs, 64, activation=tf.nn.tanh)
    x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
    x = tf.layers.dense(x, units=action_dim, activation=tf.nn.tanh)
    return x


def value_fn(obs):
    x = tf.layers.dense(obs, 64, activation=tf.nn.tanh)
    x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
    x = tf.squeeze(tf.layers.dense(x, 1, activation=None))
    return x


def run_main():

    agent = PPO(dim_obs=env.observation_space.shape,
                dim_act=action_dim,
                is_action_continuous=True,
                policy_fn=policy_fn,
                value_fn=value_fn,
                clip_schedule=lambda x: 0.1,
                lr_schedule=lambda x: 1e-3,
                train_epoch=10,
                batch_size=32,
                log_freq=1,
                save_path="./log/ppo_mujoco",
                save_model_freq=1000)
    totrew = 0
    for i in tqdm(range(MAXITER)):
        traj_list = deque()
        totrew_list = deque()
        for _ in range(10):
            traj, totrew = trajectory(env, agent)
            traj_list.append(traj)
            totrew_list.append(totrew)

        averew = np.mean(totrew_list)
        avelen = np.mean([len(t) for t in traj_list])
        agent.add_scalar("ppo/return", averew, i)
        agent.add_scalar("ppo/length", avelen, i)
        agent.update(traj_list)
        tqdm.write(f"{i}th. len={avelen}, rew={averew}")


if __name__ == "__main__":
    run_main()
