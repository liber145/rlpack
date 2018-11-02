import argparse
import gym
import numpy as np
import time
import datetime
import os
from tqdm import tqdm
from collections import deque
from tensorboardX import SummaryWriter

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines import bench, logger

from rlpack.envs import FrameStack, make_env

from rlpack.environment.atari_wrapper import get_atari_env_fn, get_eval_atari_env_fn
from rlpack.algos.ppo import PPO
from rlpack.common.memory import Memory5 as Memory


# parser = argparse.ArgumentParser(description="Parse Arguments.")
# parser.add_argument("--initial_epsilon", default=0.5, type=float)
# parser.add_argument("--final_epsilon", default=0.01, type=float)
# parser.add_argument("--batch_size", default=32*8, type=int)
# parser.add_argument("--trajectory_length", default=1024, type=int)
# parser.add_argument("--model", default="dqn", type=str)
# parser.add_argument("--n_step", default=int(1e7*1.1), type=int)
# parser.add_argument("--lr", default=0.0001, type=float)
# parser.add_argument("--memory_size", default=int(1e6), type=int)
# parser.add_argument("--discount", default=0.99, type=float)
# parser.add_argument("--update_target_freq", default=100, type=int)
# parser.add_argument("--update_freq", default=1, type=int)
# parser.add_argument("--save_model_freq", default=10000, type=int)
# parser.add_argument("--save_path", default=None, type=str)
# config = parser.parse_args()


class Config(object):
    def __init__(self):
        self.seed = 1
        self.save_path = "./log/alien"
        self.save_model_freq = 0.001

        # 环境
        self.n_stack = 4
        self.dim_observation = (210, 160, 3)
        self.n_action = None   # gym中不同环境的action数目不同。

        # 训练长度
        self.n_env = 8
        self.trajectory_length = 128
        self.n_trajectory = 10000   # for each env
        self.batch_size = 64

        # 训练参数
        self.training_epochs = 3
        self.discount = 0.99
        self.gae = 0.95
        self.lr_schedule = lambda x: (1-x) * 2.5e-4
        self.clip_schedule = lambda x: (1-x) * 0.1
        self.vf_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5


def process_config(env):
    config = Config()
    config.dim_observation = env.observation_space.shape
    config.n_action = env.action_space.n

    return config


class Trainer(object):
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.obs = env.reset()
        self.config = config
        self.memory = Memory(1000)

    def collect_trajectory(self, n_step, n_env):
        mb_obs, mb_actions, mb_rewards, mb_dones = [], [], [], []
        epinfos = []

        for _ in range(n_step):
            actions = self.agent.get_action(self.obs)
            next_obs, rewards, dones, infos = self.env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            mb_obs.append(self.obs.copy())
            mb_actions.append(actions)
            mb_rewards.append(rewards)
            mb_dones.append(dones)

            self.obs = next_obs

        mb_obs.append(self.obs.copy())                                # Add last state.
        mb_obs = np.transpose(np.asarray(mb_obs), (1, 0, 2, 3, 4))    # (8, 128+1, 84, 84, 4)
        mb_actions = np.transpose(np.asarray(mb_actions), (1, 0))     # (8, 128)
        mb_rewards = np.transpose(np.asarray(mb_rewards), (1, 0))     # (8, 128)
        mb_dones = np.transpose(np.asarray(mb_dones), (1, 0))         # (8, 128)

        # print(f"ob shape: {mb_obs.shape}")
        # print(f"act shape: {mb_actions.shape}")
        # print(f"rew shape: {mb_rewards.shape}")
        # print(f"done shape: {mb_dones.shape}")
        # print(f"done: {mb_dones[0, :]}")
        # print(f"rewards: {mb_rewards[0, :]}")
        # print(f"mb_obs: {np.max(mb_obs)}   {np.min(mb_obs)}")
        # print(f"actions: {actions}")
        # print(f"epinfos: {epinfos}")

        return [mb_obs, mb_actions, mb_rewards, mb_dones], epinfos

    def learn(self):

        epinfobuf = deque(maxlen=100)

        summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

        for i in tqdm(range(10000)):

            epinfos = []
            for step in range(config.trajectory_length):
                actions = self.agent.get_action(self.obs)
                next_obs, rewards, dones, infos = self.env.step(actions)

                for info in infos:
                    maybeepinfo = info.get('episode')
                    if maybeepinfo:
                        epinfos.append(maybeepinfo)

                self.memory.store_sard(self.obs.copy(), actions, rewards, dones)
                self.obs = next_obs

            if self.memory.size <= config.trajectory_length:
                print(f"continue......")
                continue

            update_ratio = i/10000.0
            # data_batch, epinfos = self.collect_trajectory(self.config.trajectory_length, self.config.n_env)
            data_batch = self.memory.get_last_n_step(config.trajectory_length)
            pol.update(data_batch, update_ratio)

            epinfobuf.extend(epinfos)
            summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
            summary_writer.add_scalar("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]), global_step=i)

            if i > 0 and i % 10 == 0:
                # print(f"epinfo: {epinfos}")
                rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
                lenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
                tqdm.write(f"eprewmean: {rewmean}  eplenmean: {lenmean}")


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


if __name__ == "__main__":
    n_stack = 4
    nenvs = 8
    env = SubprocVecEnv([make_env(i, 'AlienNoFrameskip-v4') for i in range(nenvs)])
    env = FrameStack(env, n_stack)

    print(f"action space: {env.action_space.n}")
    print(f"observation space: {env.observation_space.shape}")

    # env = get_atari_env_fn("AlienNoFrameskip-v4")()
    # env = gym.make("CartPole-v1")
    config = process_config(env)  # 配置config
    pol = PPO(config)

    trainer = Trainer(env, pol, config)
    trainer.learn()
