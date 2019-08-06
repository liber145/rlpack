# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from .base import Base


class TD3(Base):
    def __init__(self,
                 rnd=0,
                 dim_obs=None, dim_act=None, act_limit=None,
                 policy_fn=None, value_fn=None,
                 discount=0.99, gae=0.95,
                 train_epoch=20, policy_lr=1e-3, value_lr=1e-3, policy_delay=2,
                 target_update_rate=0.995, noise_std=0.2, noise_clip=0.5,
                 save_path="./log", log_freq=10, save_model_freq=100,
                 ):
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._act_limit = act_limit
        self._policy_fn = policy_fn
        self._value_fn = value_fn

        self._discount, self._gae = discount, gae
        self._train_epoch, self._policy_lr, self._value_lr = train_epoch, policy_lr, value_lr
        self._policy_decay = policy_delay
        self._target_update_ratio = target_update_rate
        self._noise_std, self._noise_clip = noise_std, noise_clip

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

        self.sess.run(self._init_target_policy_op)
        self.sess.run(self._init_target_value_op)

    def _build_network(self):
        """Build networks for algorithm."""
        self._obs = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")
        self._act = tf.placeholder(tf.float32, [None, self._dim_act], name="action")
        self._obs2 = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation2")
        self._reward = tf.placeholder(tf.float32, [None], name="reward")
        self._done = tf.placeholder(tf.float32, [None], name="done")

        self.all_phs = [self._obs, self._act, self._reward, self._done, self._obs2]

        with tf.variable_scope("main/policy"):
            self.act = self._policy_fn(self._obs)

        with tf.variable_scope("main/value/1"):
            self.q1 = self._value_fn(self._obs, self._act)

        with tf.variable_scope("main/value/2"):
            self.q2 = self._value_fn(self._obs, self._act)

        with tf.variable_scope("main/value/1", reuse=True):
            self.q_act = self._value_fn(self._obs, self.act)

        with tf.variable_scope("target/policy"):
            self.act_targ = self._policy_fn(self._obs2)

        epsilon = tf.random_normal(tf.shape(self.act_targ), stddev=self._noise_std)
        epsilon = tf.clip_by_value(epsilon, -self._noise_clip, self._noise_clip)
        a2 = self.act_targ + epsilon
        a2 = tf.clip_by_value(a2, -self._act_limit, self._act_limit)
        with tf.variable_scope("target/value/1"):
            self.q1_targ = self._value_fn(self._obs2, a2)

        with tf.variable_scope("target/value/2"):
            self.q2_targ = self._value_fn(self._obs2, a2)

    def _update_target(self, net1, net2, rho=0):
        params1 = tf.trainable_variables(net1)
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.trainable_variables(net2)
        params2 = sorted(params2, key=lambda v: v.name)
        assert len(params1) == len(params2)
        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param1.assign(rho*param1 + (1-rho)*param2))
        return update_ops

    def _build_algorithm(self):
        policy_vars = tf.trainable_variables("main/policy")
        value_vars = tf.trainable_variables("main/value")

        policy_loss = -tf.reduce_mean(self.q_act)

        min_q_targ = tf.minimum(self.q1_targ, self.q2_targ)
        backup = tf.stop_gradient(self._reward + self._discount*(1-self._done)*min_q_targ)
        q1_loss = tf.reduce_mean((self.q1-backup)**2)
        q2_loss = tf.reduce_mean((self.q2-backup)**2)
        value_loss = q1_loss + q2_loss

        self.train_policy_op = tf.train.AdamOptimizer(self._policy_lr).minimize(policy_loss, var_list=policy_vars)
        self.train_value_op = tf.train.AdamOptimizer(self._value_lr).minimize(value_loss, var_list=value_vars)

        self._update_target_policy_op = self._update_target("target/policy", "main/policy", self._target_update_ratio)
        self._update_target_value_op = self._update_target("target/value", "main/value", self._target_update_ratio)

        self._init_target_policy_op = self._update_target("target/policy", "main/policy")
        self._init_target_value_op = self._update_target("target/value", "main/value")

    def get_action(self, obs):
        """动作添加扰动。
        """
        action = self.sess.run(self.act, feed_dict={self._obs: obs})
        noised_action = action + 0.1 * np.random.randn(*action.shape)
        return noised_action

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch
        inputs = {k: v for k, v in zip(self.all_phs, [s_batch, a_batch, r_batch, d_batch, next_s_batch])}

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        # 更新值函数。
        self.sess.run(self.train_value_op, feed_dict=inputs)

        # 延迟更新策略模型。
        if global_step % self._policy_decay == 0:
            self.sess.run([self.train_policy_op, self._update_target_value_op, self._update_target_policy_op], feed_dict=inputs)

        if global_step % self._save_model_freq == 0:
            self.save_model()
