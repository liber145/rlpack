import argparse
from collections import deque
import gym
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm


from rlpack.algos import PPO
from rlpack.environment.atari_wrappers import make_atari, wrap_deepmind

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
parser.add_argument("--niter", type=int, default=int(1e4))
args = parser.parse_args()


def trajectory(env, agent):
    t = deque()
    s = env.reset()
    tsum = 0
    while True:
        s = np.array(s)
        a = agent.get_action(s[np.newaxis, :])[0]
        ns, r, d, _ = env.step(a)
        t.append((s, a, r))
        s = ns
        tsum += r
        if d is True:
            break
    return t, tsum


def run_main():
    env = make_atari(args.env)
    env = wrap_deepmind(env, frame_stack=True)
    agent = PPO(dim_obs=env.observation_space.shape, dim_act=env.action_space.n, save_path="./log/ppo_atari")
    sw = SummaryWriter("./log/ppo_atari")

    for i in tqdm(range(args.niter)):
        traj_list = deque()
        totrew_list = deque()
        for _ in range(10):
            traj, totrew = trajectory(env, agent)
            traj_list.append(traj)
            totrew_list.append(totrew)
        sw.add_scalars("ppo", {"total_reward": np.mean(totrew_list)}, global_step=i)
        agent.update(traj_list)
        tqdm.write(f"{i}th iteration: len={np.mean([len(t) for t in traj_list])}")


def run_game():
    env = make_atari(args.env)
    env = wrap_deepmind(env, frame_stack=True)
    s = env.reset()
    tsum = 0
    tt = list()
    for i in range(10000):
        a = np.random.randint(4)
        s, r, d, _ = env.step(a)
        tsum += r
        print("s:", id(s), "type(s):", type(s), "r:", r, "d:", d, "tsum:", tsum, "sum:", np.sum(s))

        if d is True:

            input()
            env.reset()
            tsum = 0


if __name__ == "__main__":
    run_main()
