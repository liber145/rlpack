"""
Proximal Policy Optimization.
目标loss由三部分组成：1. clipped policy loss；2. value loss；3. entropy。
policy loss需要计算当前policy和old policy在当前state上的ratio。
需要注意的是，state分布依赖于old policy。因此，更新中的old policy是一样的。
"""


import math
from collections import defaultdict, deque

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..common.utils import assert_shape
from .base import Base


class PPO(Base):
    def __init__(self,
                 rnd=0,
                 dim_obs=None, dim_act=None,
                 policy_fn=None, value_fn=None,
                 discount=0.99, gae=0.95, clip_ratio=0.2,
                 train_epoch=40, policy_lr=1e-3, value_lr=1e-3,
                 save_path="./log", log_freq=10, save_model_freq=100):
        # Save.
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._policy_fn = policy_fn
        self._value_fn = value_fn

        self._clip_ratio = clip_ratio
        self._discount = discount
        self._gae = gae
        self._train_epoch = train_epoch
        self._policy_lr = policy_lr
        self._value_lr = value_lr

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build tensorflow operations for algorithms."""
        self._obs = tf.placeholder(tf.float32, [None, self._dim_obs])
        self._act = tf.placeholder(tf.float32, [None, self._dim_act])

        self._adv = tf.placeholder(tf.float32, [None])
        self._ret = tf.placeholder(tf.float32, [None])
        self._logp_old = tf.placeholder(tf.float32, [None])
        self.all_phs = [self._obs, self._act, self._adv, self._ret, self._logp_old]

        self.pi, self.logp, self.logp_pi = self._policy_fn(self._obs, self._act)
        self.v = self._value_fn(self._obs)

    def _build_algorithm(self):
        """Build algorithms using prebuilt networks."""

        ratio = tf.exp(self.logp - self._logp_old)
        surr1 = ratio * self._adv
        surr2 = tf.clip_by_value(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio) * self._adv
        self.policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        self.value_loss = tf.reduce_mean((self.v - self._ret)**2)

        self._train_policy_op = tf.train.AdamOptimizer(self._policy_lr).minimize(self.policy_loss)
        self._train_value_op = tf.train.AdamOptimizer(self._value_lr).minimize(self.value_loss)

    def get_action(self, obs) -> np.ndarray:
        """Return action according to the observations.
        :param obs: the observation that could be image or real-number features
        :return: actions
        """
        a = self.sess.run(self.pi, feed_dict={self._obs: obs})
        return a

    def update(self, databatch):
        """
        参数:
            databatch：一个列表，分别是state, action, reward, done, early_stop, next_state。每个是矩阵或向量。
            state是状态，action是动作，reward是奖励，done是是否完结，early_stop是是否提前结束，next_state是下一个状态。
        """
        preprocess_databatch = self._parse_databatch(*databatch)

        inputs = {k: v for k, v in zip(self.all_phs, preprocess_databatch)}
        pi_l_old, v_l_old = self.sess.run([self.policy_loss, self.value_loss], feed_dict=inputs)

        # Training
        for i in range(self._train_epoch):
            self.sess.run(self._train_policy_op, feed_dict=inputs)
        for i in range(self._train_epoch):
            self.sess.run(self._train_value_op, feed_dict=inputs)

        pi_l_new, v_l_new = self.sess.run([self.policy_loss, self.value_loss], feed_dict=inputs)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        if global_step % self._save_model_freq == 0:
            self.save_model()

    def _parse_databatch(self, states, actions, rewards, dones, earlystops, nextstates):

        batch_size = len(dones)
        oldlogproba, values = self.sess.run([self.logp, self.v], feed_dict={self._obs: states, self._act: actions})
        nextvalues = self.sess.run(self.v, feed_dict={self._obs: nextstates})

        returns = np.zeros(batch_size)
        deltas = np.zeros(batch_size)
        advantages = np.zeros(batch_size)

        for i in reversed(range(batch_size)):

            if dones[i]:
                prev_return = 0
                prev_value = 0
                prev_advantage = 0
            elif earlystops[i]:
                prev_return = nextvalues[i]
                prev_value = prev_return
                prev_advantage = 0

            returns[i] = rewards[i] + self._discount * prev_return * (1 - dones[i])
            deltas[i] = rewards[i] + self._discount * prev_value * (1 - dones[i]) - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + self._discount * self._gae * prev_advantage * (1 - dones[i])

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        advantages = (advantages - advantages.mean()) / advantages.std()

        return [states, actions, advantages, returns, oldlogproba]
