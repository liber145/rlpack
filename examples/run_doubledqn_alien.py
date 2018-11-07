import os
from collections import deque

import numpy as np
import tensorflow as tf
from rlpack.algos import DoubleDQN
from rlpack.common import Memory
from rlpack.environment import AtariWrapper
from tensorboardX import SummaryWriter
from tqdm import tqdm


class Config(object):
    def __init__(self):
        self.seed = 1
        self.save_path = "./log/doubledqn_alien"
        self.save_model_freq = 0.001
        self.log_freq = 10

        # 环境
        self.n_stack = 4
        self.dim_observation = None
        self.dim_action = None
        self.n_action = None   # gym中不同环境的action数目不同。

        # 训练长度
        self.n_env = 8
        self.trajectory_length = 128
        self.n_trajectory = 10000   # for each env
        self.batch_size = 64
        self.warm_start_length = 1000

        # 训练参数
        self.initial_epsilon = 0.9
        self.final_epsilon = 0.01
        self.lr = 2.5e-4
        self.update_target_freq = 500

        self.training_epoch = 3
        self.discount = 0.99
        self.gae = 0.95
        self.vf_coef = 1.0
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.lr_schedule = lambda x: (1 - x) * 2.5e-4
        self.clip_schedule = lambda x: (1 - x) * 0.1
        self.memory_size = 10000


class Agent(DoubleDQN):
    def __init__(self, config):
        super().__init__(config)

    def build_network(self):
        self.observation = tf.placeholder(shape=[None, *self.dim_observation], dtype=tf.float32, name="observation")

        with tf.variable_scope("qnet"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu, trainable=True)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu, trainable=True)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu, trainable=True)
            x = tf.contrib.layers.flatten(x)   # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=True)
            self.qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=True)

        with tf.variable_scope("target_qnet"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu, trainable=True)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu, trainable=True)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu, trainable=True)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=True)
            self.target_qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=True)


def process_config(env):
    config = Config()
    config.dim_observation = env.observation_space.shape
    config.n_action = env.action_space.n

    return config


def safemean(x):
    return np.nan if len(x) == 0 else np.mean(x)


def learn(env, agent, config):

    memory = Memory(config.memory_size)
    epinfobuf = deque(maxlen=100)
    summary_writer = SummaryWriter(os.path.join(config.save_path, "summary"))

    # 热启动，随机收集数据。
    obs = env.reset()
    print(f"observation: max={np.max(obs)} min={np.min(obs)}")
    for i in tqdm(range(config.warm_start_length)):
        actions = agent.get_action(obs)
        next_obs, rewards, dones, infos = env.step(actions)

        memory.store_sard(obs, actions, rewards, dones)
        obs = next_obs

    print("Finish warm start.")
    print("Start training.")
    for i in tqdm(range(config.n_trajectory)):
        epinfos = []
        for _ in range(config.trajectory_length):
            actions = agent.get_action(obs)
            next_obs, rewards, dones, infos = env.step(actions)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo:
                    epinfos.append(maybeepinfo)

            memory.store_sard(obs, actions, rewards, dones)
            obs = next_obs

        update_ratio = i / config.n_trajectory
        data_batch = memory.sample_transition(config.batch_size)
        agent.update(data_batch, update_ratio)

        epinfobuf.extend(epinfos)
        summary_writer.add_scalar("eprewmean", safemean([epinfo["r"] for epinfo in epinfobuf]), global_step=i)
        summary_writer.add_scalar("eplenmean", safemean([epinfo['l'] for epinfo in epinfobuf]), global_step=i)

        if i > 0 and i % config.log_freq == 0:
            rewmean = safemean([epinfo["r"] for epinfo in epinfobuf])
            lenmean = safemean([epinfo['l'] for epinfo in epinfobuf])
            print(f"eprewmean: {rewmean}  eplenmean: {lenmean}")


if __name__ == "__main__":
    n_stack = 4
    n_env = 8
    env = AtariWrapper("AlienNoFrameskip-v4", 8)

    print("---------------")
    print(f"action space: {env.action_space.n}")
    print(f"observation space: {env.observation_space.shape}")
    print("---------------")

    config = process_config(env)  # 配置config
    pol = Agent(config)

    learn(env, pol, config)