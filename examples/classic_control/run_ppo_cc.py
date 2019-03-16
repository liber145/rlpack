import argparse
from collections import deque
import gym
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

from rlpack.algos import PPO


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--env", type=str, default="CartPole-v1")
parser.add_argument("--niter", type=int, default=1000)
parser.add_argument("--batchsize", type=int, default=128)
args = parser.parse_args()


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
    agent = PPO(dim_obs=env.observation_space.shape,
                dim_act=env.action_space.n,
                clip_schedule=lambda x: 0.1,
                lr_schedule=lambda x: 1e-4,
                train_epoch=10,
                batch_size=args.batchsize,
                log_freq=10,
                save_path="./log/ppo_cc",
                save_model_freq=1000)
    sw = SummaryWriter(log_dir="./log/ppo_cc")
    totrew = 0
    for i in tqdm(range(args.niter)):
        traj_list = deque()
        totrew_list = deque()
        for _ in range(10):
            traj, totrew = trajectory(env, agent)
            # print("traj:", traj, "totrew:", totrew)
            # input()
            traj_list.append(traj)
            totrew_list.append(totrew)
        sw.add_scalars("ppo", {"totrew": np.mean(totrew_list)}, i)
        agent.update(traj_list)
        tqdm.write(f"{i}th. len={np.mean([len(t) for t in traj_list])}")


def run_game():
    env = gym.make(args.env)
    s = env.reset()
    totrew = 0
    for i in range(100):
        a = np.random.randint(2)
        ns, r, d, _ = env.step(a)
        s = ns
        totrew += r
        if d is True:
            s = env.reset()


if __name__ == "__main__":
    run_main()
