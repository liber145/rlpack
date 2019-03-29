import argparse
from collections import deque
import gym
import numpy as np
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf

from rlpack.algos import PPO
from rlpack.environment import make_ramatari


parser = argparse.ArgumentParser(description="Parse environment name.")
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--env", type=str, default="Pong-ramNoFrameskip-v4")
parser.add_argument("--niter", type=int, default=1000)
parser.add_argument("--batchsize", type=int, default=128)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


env = make_ramatari(args.env)


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


def obs_fn():
    obs = tf.placeholder(shape=[None, 128, 4], dtype=tf.uint8, name="observation")
    obs = tf.to_float(obs) / 255.0
    return obs


def policy_fn(obs):
    x = tf.layers.conv1d(obs, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=env.action_space.n)
    return x


def value_fn(obs):
    x = tf.layers.conv1d(obs, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
    x = tf.layers.conv1d(x, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
    x = tf.layers.flatten(x)
    x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    x = tf.layers.dense(x, units=1)
    return x


def run_main():
    agent = PPO(obs_fn=obs_fn,
                policy_fn=policy_fn,
                value_fn=value_fn,
                dim_act=env.action_space.n,
                clip_schedule=lambda x: 0.1,
                lr_schedule=lambda x: 1e-4,
                train_epoch=10,
                batch_size=args.batchsize,
                log_freq=10,
                save_path=f"./log/ppo_ramatari/{args.env}",
                save_model_freq=1000)
    sw = SummaryWriter(log_dir=f"./log/ppo_ramatari/{args.env}")
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
