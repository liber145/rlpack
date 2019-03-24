# -*- coding: utf-8 -*-


import argparse
from collections import deque
import gym
import numpy as np
import os
from tensorboardX import SummaryWriter
from tqdm import tqdm

from rlpack.algos import ContinuousPPO

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--env',  type=str, default="MountainCarContinuous-v0")
args = parser.parse_args()

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


def run_main():
    env = gym.make(args.env)
    agent = ContinuousPPO(dim_obs=env.observation_space.shape,
                          dim_act=env.action_space.shape[0],
                          save_path="./log/ppo_mujoco"
                          )
    sw = SummaryWriter(log_dir="./log/ppo_mujoco")
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
        sw.add_scalars("cont_ppo", {"rew": averew, "len": avelen}, i)
        agent.update(traj_list)
        tqdm.write(f"{i}th. len={avelen}, rew={averew}")


if __name__ == "__main__":
    run_main()
