# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .base import Base


class DDPG(Base):
    """Deep Deterministic Policy Gradient."""

    def __init__(self,
                 rnd=0,
                 dim_obs=None, dim_act=None, act_limit=None,
                 policy_fn=None, value_fn=None,
                 discount=0.99,
                 train_epoch=1, policy_lr=1e-3, value_lr=1e-3,
                 epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                 target_update_rate=0.995,
                 save_path="./log", log_freq=10, save_model_freq=100,
                 ):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._act_limit = act_limit
        self._policy_fn = policy_fn
        self._value_fn = value_fn

        self._discount = discount
        self._train_epoch = train_epoch
        self._policy_lr = policy_lr
        self._value_lr = value_lr

        self._update_target_rate = target_update_rate

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        self._obs = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")
        self._act = tf.placeholder(tf.float32, (None, self._dim_act), name="action")
        self._reward = tf.placeholder(tf.float32, [None], name="reward")
        self._done = tf.placeholder(tf.float32, [None], name="done")
        self._obs2 = tf.placeholder(tf.float32, [None, *self._dim_obs], name="next_observation")
        self.all_phs = [self._obs, self._act, self._reward, self._done, self._obs2]

        with tf.variable_scope("main/policy"):
            self.act = self._policy_fn(self._obs)

        with tf.variable_scope("main/value"):
            self.q = self._value_fn(self._obs, self._act)

        with tf.variable_scope("main/value", reuse=True):
            self.q_act = self._value_fn(self._obs, self.act)

        with tf.variable_scope("target/policy"):
            self.act_targ = self._policy_fn(self._obs2)

        with tf.variable_scope("target/value"):
            self.q_targ = self._value_fn(self._obs2, self.act_targ)

    def _build_algorithm(self):
        """Build networks for algorithm."""
        policy_vars = tf.trainable_variables("main/policy")
        value_vars = tf.trainable_variables("main/value")

        policy_loss = -tf.reduce_mean(self.q_act)
        qbackup = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * self.q_targ)
        value_loss = tf.reduce_mean(tf.squared_difference(self.q, qbackup))

        self.train_policy_op = tf.train.AdamOptimizer(self._policy_lr).minimize(policy_loss, var_list=policy_vars)
        self.train_value_op = tf.train.AdamOptimizer(self._value_lr).minimize(value_loss, var_list=value_vars)

        def _update_target(net1, net2, rho=0):
            variables1 = tf.trainable_variables(net1)
            variables1 = sorted(variables1, key=lambda v: v.name)
            variables2 = tf.trainable_variables(net2)
            variables2 = sorted(variables2, key=lambda v: v.name)
            assert len(variables1) == len(variables2)
            return [v1.assign(rho*v1 + (1-rho)*v2) for v1, v2 in zip(variables1, variables2)]

        self._update_target_op = tf.group(_update_target("target/policy", "main/policy", rho=self._update_target_rate)
                                          + _update_target("target/value", "main/value", rho=self._update_target_rate))
        self._init_target_op = tf.group(_update_target("target/policy", "main/policy") + _update_target("target/value", "main/value"))

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch
        inputs = {k: v for k, v in zip(self.all_phs, [s_batch, a_batch, r_batch, d_batch, next_s_batch])}

        for _ in range(self._train_epoch):
            self.sess.run(self.train_value_op, feed_dict=inputs)

        for _ in range(self._train_epoch):
            self.sess.run(self.train_policy_op, feed_dict=inputs)

        self.sess.run(self._update_target_op)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

    def get_action(self, obs):
        """动作添加高斯扰动。
        """
        act = self.sess.run(self.act, feed_dict={self._obs: obs})
        act += 0.1 * np.random.randn(*act.shape)
        return np.clip(act, -self._act_limit, self._act_limit)
