# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
from ..common.log import logger
from .base import Base
from ..common.utils import assert_shape


class DQN(Base):
    """Deep Q Network."""

    def __init__(self, config):
        """
        1. 从config中获得参数。
        2. 初始化tensorflow配置，如申请graph，动态使用gpu空间等。
        3. 搭建神经网络，如值函数网络，目标值函数网络等。
        4. 使用上面的函数近似网络，搭建算法框架，如DQN等。
        5. 创建tf.saver，tf.session。
        """
        super().__init__(config)

    def build_network(self):
        """ ------------- 搭建网络 -------------- """
        self.observation = tf.placeholder(shape=[None, *self.dim_observation], dtype=tf.float32, name="observation")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.target = tf.placeholder(shape=[None], dtype=tf.float32, name="target")  # 目标状态动作值。

        # 值函数网络和目标值函数网络。
        with tf.variable_scope("qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=True)
            self.qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=True)

        with tf.variable_scope("target_qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=False)
            self.target_qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=False)

    def build_algorithm(self):
        """ ------------- 构建算法 -------------- """
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1.5e-8)
        trainable_variables = tf.trainable_variables("qnet")

        # 当前状态动作值。
        batch_size = tf.shape(self.observation)[0]
        # gather_indices = tf.range(batch_size) * self.n_action + self.action
        # action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)

        action_index = tf.stack([tf.range(batch_size), self.action], axis=1)
        action_q = tf.gather_nd(self.qvals, action_index)
        assert_shape(action_q, [None])

        # 计算损失函数，优化参数。
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, action_q))   # 损失值。
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.train.get_global_step(),
                                                var_list=trainable_variables
                                                )

        # 更新目标网络。
        def _update_target(new_net, old_net):
            params1 = tf.trainable_variables(old_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.global_variables(new_net)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param2.assign(param1))
            return update_ops

        self.update_target_op = _update_target("target_qnet", "qnet")

        # ------------------------------------------
        # ------------- 需要记录的中间值 --------------
        # ------------------------------------------
        # 状态值中的最大的q值，用于打印出来，用于反应效果怎么样。
        self.max_qval = tf.reduce_max(self.qvals)

    # def _prepare(self):
    #     # ------------- 创建存储器，并创建session --------------
    #     self.saver = tf.train.Saver(max_to_keep=5)
    #     conf = tf.ConfigProto(allow_soft_placement=True)
    #     conf.gpu_options.allow_growth = True  # pylint: disable=E1101
    #     self.sess = tf.Session(config=conf)
    #     self.sess.run(tf.global_variables_initializer())

    #     # self.cnt = None

    #     # ------------- 初始化记录器 --------------
    #     self.summary_writer = SummaryWriter(os.path.join(self.save_path, "summary"))

    #     # ------------- 从上一次存储的模型开始 --------------
    #     self.load_model(self.save_path)

    #     # ------------- 初始化其他 --------------
    #     self.total_reward = 0

    def get_action(self, obs):
        """Get action according to the given observation and epsilon-greedy method.

        Args:
            obs: observation. The shape needs to be [None, dim_observation].
        """
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        self.epsilon -= (self.initial_epsilon - self.final_epsilon) / 100000
        self.epsilon = max(self.final_epsilon, self.epsilon)

        qvals = self.sess.run(self.qvals, feed_dict={self.observation: newobs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = newobs.shape[0]
        actions = np.random.randint(self.n_action, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon
        actions[idx] = best_action[idx]

        if obs.ndim == 1:
            actions = actions[0]
        return actions

    def update(self, minibatch, update_ratio: float):
        """更新策略，使用minibatch样本。"""

        # 拆分sample样本。
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        target_next_q_vals = self.sess.run(self.target_qvals, feed_dict={self.observation: next_s_batch})
        target_batch = r_batch + (1 - d_batch) * self.discount * target_next_q_vals.max(axis=1)

        _, global_step, loss, max_q_val = self.sess.run(
            [self.train_op,
             tf.train.get_global_step(),
             self.loss,
             self.max_qval],
            feed_dict={
                self.observation: s_batch,
                self.action: a_batch,
                self.target: target_batch
            }
        )

        # 存储结果。
        # self.summary_writer.add_scalar("loss", loss, global_step)
        # self.summary_writer.add_scalar("max_q_value", max_q_val, global_step)

        # 存储模型。
        if global_step % self.save_model_freq == 0:
            self.save_model()

        # 更新目标策略。
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return global_step, {"loss": loss, "max_q_value": max_q_val}

    # def save_model(self, save_path):
    #     global_step = self.sess.run(tf.train.get_global_step())
    #     self.saver.save(
    #         self.sess,
    #         os.path.join(save_path, "model", "model"),
    #         global_step,
    #         write_meta_graph=True
    #     )

    # def load_model(self, save_path):
    #     latest_checkpoint = tf.train.latest_checkpoint(save_path)
    #     if latest_checkpoint:
    #         logger.info("Loading model checkpoint {}...".format(latest_checkpoint))
    #         self.saver.restore(self.sess, latest_checkpoint)
    #     else:
    #         logger.info("New start!!")

    # def put(self, data):
    #     _, _, r, _, d = data
    #     self.total_reward += r
    #     if d is True:
    #         global_step = self.sess.run(tf.train.get_global_step())
    #         self.summary_writer.add_scalar("total_reward", self.total_reward, global_step)
    #         self.total_reward = 0
