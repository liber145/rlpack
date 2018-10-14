from abc import ABC, abstractmethod
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from ..common.log import logger


class BaseQ(ABC):
    """Q-learning基类。"""

    def __init__(self, config):
        self.dim_observation = config.dim_observation
        self.n_action = config.n_action
        self.discount = config.discount
        self.batch_size = config.batch_size
        self.initial_epsilon = config.initial_epsilon
        self.final_epsilon = config.final_epsilon
        self.epsilon = self.initial_epsilon
        self.lr = config.lr
        self.update_target_freq = config.update_target_freq
        self.save_path = config.save_path
        self.save_model_freq = config.save_model_freq

        # ------------------------ 申请网络图 ------------------------
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)

        # ------------------------ 搭建网络 ------------------------
        self.build_network()

        # ------------------------ 搭建算法 ------------------------
        self.build_algorithm()

        # ------------------------ 存储模型，存储训练信息，重载上回模型 ------------------------
        self._prepare()

    @abstractmethod
    def build_network(self):
        """搭建网络，为算法服务。"""
        pass

    @abstractmethod
    def build_algorithm(self):
        """搭建算法。"""
        pass

    def _prepare(self):
        # ------------------------ 初始化存储器 ------------------------
        self.saver = tf.train.Saver(max_to_keep=5)

        # ------------------------ 创建会话 ------------------------
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)

        # ------------------------ 初始化参数 ------------------------
        self.sess.run(tf.global_variables_initializer())

        # ------------------------ 初始化记录器 ------------------------
        self.summary_writer = SummaryWriter(os.path.join(self.save_path, "summary"))

        # ------------------------ 从上次存储的模型开始训练 ------------------------
        self.load_model(self.save_path)

        # ------------------------ 初始化其他 ------------------------
        self.total_reward = 0

    @abstractmethod
    def get_action(self, obs):
        """从当前观测值获得动作。"""
        pass

    @abstractmethod
    def update(self, minibatch):
        """更新策略，使用minibatch样本。"""
        pass

    def save_model(self, save_path):
        global_step = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            os.path.join(save_path, "model", "model"),
            global_step,
            write_meta_graph=True
        )

    def load_model(self, save_path):
        latest_checkpoint = tf.train.latest_checkpoint(save_path)
        if latest_checkpoint:
            logger.info("Loading model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            logger.info("New start!")

    def put(self, reward, done):
        """收集过程中的奖励，并将其存储起来。"""
        self.total_reward += reward
        if done is True:
            global_step = self.sess.run(tf.train.get_global_step())
            self.summary_writer.add_scalar("total_reward", self.total_reward, global_step=global_step)
            self.total_reward = 0
