import sys

import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class SAC(Base):
    def __init__(self,
                 rnd=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=1000,
                 policy_lr=1e-3,
                 value_lr=1e-3,
                 alpha=0.2,
                 action_high=1.0,
                 action_low=-1.0,
                 moving_udpate_target_ratio=0.995
                 ):
        self.action_high = action_high
        self.action_low = action_low
        self.alpha = alpha
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.moving_update_target_ratio = moving_udpate_target_ratio

        self.discount = discount
        self.log_freq = log_freq
        self.save_model_freq = save_model_freq

        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0
        self.EPS = 1e-8

        super().__init__(save_path=save_path, rnd=rnd)
        self.sess.run(self.init_target)

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def clip_but_pass_gradient(self, x, l=-1, u=1):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient(clip_up * (u - x) + clip_low * (l - x))

    def _build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_obs], name="observation")
        self.action = tf.placeholder(tf.float32, [None, self.dim_act], name="action")
        self.reward = tf.placeholder(tf.float32, [None], name="reward")
        self.done = tf.placeholder(tf.float32, [None], name="done")
        self.next_observation = tf.placeholder(tf.float32, [None, *self.dim_obs], name="target_observation")

        with tf.variable_scope("main/policy"):
            x = tf.layers.dense(self.observation, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.mu = tf.layers.dense(x, units=self.dim_act, activation=None)
            self.log_std = tf.layers.dense(x, units=self.dim_act, activation=tf.tanh)
            self.log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (self.log_std + 1)

            std = tf.exp(self.log_std)
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu)) * std
            presum = -0.5 * (((self.pi-self.mu)/(tf.exp(self.log_std)+self.EPS))**2 + np.log(2*np.pi) + 2*self.log_std)
            self.logp_pi = tf.reduce_sum(presum, axis=1)

            # normal_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.log_std))
            # self.pi = normal_dist.sample()
            # self.logp_pi = normal_dist.log_prob(self.pi)

            # Squash into an appropriate scale.
            self.mu = tf.tanh(self.mu)
            self.pi = tf.tanh(self.pi)
            self.logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - self.pi**2, l=0, u=1) + 1e-6), axis=1)

        with tf.variable_scope("main/action_value_1"):
            x = tf.concat([self.observation, self.action], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q1 = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/action_value_1", reuse=True):
            x = tf.concat([self.observation, self.pi], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q1_pi = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/action_value_2"):
            x = tf.concat([self.observation, self.action], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q2 = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/action_value_2", reuse=True):
            x = tf.concat([self.observation, self.pi], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q2_pi = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/state_value"):
            x = tf.layers.dense(self.observation, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.v = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("target/state_value"):
            x = tf.layers.dense(self.next_observation, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.v_targ = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

    def _build_algorithm(self):
        min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

        q_backup = tf.stop_gradient(self.reward + self.discount * (1 - self.done) * self.v_targ)
        v_backup = tf.stop_gradient(min_q_pi - self.alpha * self.logp_pi)

        # Soft actor-critic loss
        self.pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - self.q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        self.value_loss = q1_loss + q2_loss + v_loss

        # Train policy.
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.policy_lr)
        self.update_policy = pi_optimizer.minimize(self.pi_loss, var_list=self.get_vars("main/policy"))

        # Train value.
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.value_lr)
        value_vars = self.get_vars("main/action_value") + self.get_vars("main/state_value")
        with tf.control_dependencies([self.update_policy]):
            self.update_value = value_optimizer.minimize(self.value_loss, var_list=value_vars)

        main_vars = self.get_vars("main/state_value")
        target_vars = self.get_vars("target/state_value")
        with tf.control_dependencies([self.update_value]):
            self.update_target = tf.group([tf.assign(v_targ, self.moving_update_target_ratio * v_targ + (1 - self.moving_update_target_ratio)*v_main) for v_main, v_targ in zip(main_vars, target_vars)])

        self.init_target = tf.group([tf.assign(v_targ, v_main) for v_main, v_targ in zip(main_vars, target_vars)])

    def update(self, minibatch, update_ratio: float):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        n_env, batch_size = s_batch.shape[:2]
        s_batch = s_batch.reshape(n_env * batch_size, *self.dim_obs)
        a_batch = a_batch.reshape(n_env * batch_size, self.dim_act)
        r_batch = r_batch.reshape(n_env * batch_size)
        d_batch = d_batch.reshape(n_env * batch_size)
        next_s_batch = next_s_batch.reshape(n_env * batch_size, *self.dim_obs)

        value_loss, policy_loss, _, _, _ = self.sess.run([self.value_loss, self.pi_loss, self.update_policy, self.update_value, self.update_target], feed_dict={
            self.observation: s_batch,
            self.action: a_batch,
            self.reward: r_batch,
            self.done: d_batch,
            self.next_observation: next_s_batch})

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        if global_step % self.save_model_freq == 0:
            self.save_model()

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        action = self.sess.run(self.pi, feed_dict={self.observation: newobs})
        return action
