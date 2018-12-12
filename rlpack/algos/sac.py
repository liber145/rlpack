import sys

import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class SAC(Base):
    def __init__(self, config):
        self.tau = 0.995
        self.action_high = 1.0
        self.action_low = -1.0
        self.policy_mean_reg_weight = 1e-3
        self.policy_std_reg_weight = 1e-3
        self.alpha = 0.2
        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0
        self.EPS = 1e-8
        super().__init__(config)
        self.sess.run(self.init_target_vf)

    def build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="observation")
        self.action = tf.placeholder(tf.float32, [None, self.dim_action], name="action")
        self.target_observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="target_observation")
        print("observation:", self.dim_observation)

        with tf.variable_scope("policy_net"):
            x = tf.layers.dense(self.observation, 100, activation=tf.nn.relu)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            self.mu = tf.layers.dense(x, self.dim_action)
            self.log_std = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh)
            self.log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (self.log_std + 1)
            std = tf.exp(self.log_std)
            self.sampled_action = self.mu + tf.random_normal(tf.shape(self.mu)) * std

            # self.normal_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.log_std)+self.EPS)
            # self.sampled_action = self.normal_dist.sample()
            # squash into [-1, 1]

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

    def build_algorithm(self):
        self.actor_optimizer = tf.train.AdamOptimizer(1e-3)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)

        self.reward = tf.placeholder(tf.float32, [None], name="reward")
        self.mask = tf.placeholder(tf.float32, [None], name="mask")   # 1 - done

        # Compute log(pi).
        logp = tf.reduce_sum(-0.5 * (((self.sampled_action-self.mu)/(tf.exp(self.log_std)+self.EPS))**2 + 2*self.log_std + np.log(2*np.pi)), axis=1)
        self.mu = tf.tanh(self.mu)
        self.sampled_action = tf.tanh(self.sampled_action)
        logp -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - self.sampled_action**2, l=0.0, u=1.0) + 1e-6), axis=1)
        # logp = self.normal_dist.log_prob(self.sampled_action)
        # logp -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - self.sampled_action**2, l=0.0, u=1.0) + 1e-6), axis=1)
        assert_shape(logp, [None])

        with tf.variable_scope("state_value_net"):
            x = tf.layers.dense(self.observation, 100, activation=tf.nn.relu)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            self.sval = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("target_state_value_net"):
            x = tf.layers.dense(self.target_observation, 100, activation=tf.nn.relu)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            self.target_sval = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("action_value_net"):
            x = tf.concat([self.observation, self.action], axis=1)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            self.qval = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("action_value_net", reuse=True):
            x = tf.concat([self.observation, self.sampled_action], axis=1)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            self.qval_sampled_action = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("action_value_net_2"):
            x = tf.concat([self.observation, self.action], axis=1)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            self.qval_2 = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("action_value_net_2", reuse=True):
            x = tf.concat([self.observation, self.sampled_action], axis=1)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            x = tf.layers.dense(x, 100, activation=tf.nn.relu)
            self.qval_sampled_action_2 = tf.squeeze(tf.layers.dense(x, 1))

        # Compute VF loss.
        min_qval_sampled_action = tf.minimum(self.qval_sampled_action, self.qval_sampled_action_2)
        tmp = 0.5 * (self.sval - tf.stop_gradient(min_qval_sampled_action - self.alpha * logp)) ** 2
        self.VF_loss = tf.reduce_mean(tmp)
        # tf.print("tmp:", tmp.shape, output_stream=sys.stdout)

        # Comput QF loss.
        target = tf.stop_gradient(self.reward + self.discount * self.mask * self.target_sval)
        q1_loss = 0.5 * tf.reduce_mean((self.qval - target) ** 2)
        q2_loss = 0.5 * tf.reduce_mean((self.qval_2 - target) ** 2)
        self.Value_loss = self.VF_loss + q1_loss + q2_loss

        # Compute Policy loss.
        self.Policy_loss = tf.reduce_mean(self.alpha * logp - self.qval_sampled_action)
        mean_reg_loss = self.policy_mean_reg_weight * tf.reduce_mean(self.mu**2)
        std_reg_loss = self.policy_std_reg_weight * tf.reduce_mean(self.log_std**2)
        self.Policy_loss += mean_reg_loss + std_reg_loss

        def _update_target(new_net, old_net, bili=0.0):
            """new is target net. old is updated net."""
            params1 = tf.trainable_variables(new_net)
            params2 = tf.trainable_variables(old_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param1.assign(param1 * bili + param2 * (1 - bili)))
            return update_ops

        # Update network.
        self.update_policy = self.actor_optimizer.minimize(self.Policy_loss, var_list=tf.trainable_variables("policy_net"))

        with tf.control_dependencies([self.update_policy]):
            self.update_value = self.critic_optimizer.minimize(self.Value_loss, var_list=tf.trainable_variables("state_value_net")+tf.trainable_variables("action_value_net"))

        with tf.control_dependencies([self.update_value]):
            self.update_target_vf = _update_target("target_state_value_net", "state_value_net", bili=self.tau)

        self.init_target_vf = _update_target("target_state_value_net", "state_value_net", bili=0.0)

        # Staistics.
        self.all_variables_norm = tf.linalg.global_norm(tf.trainable_variables())

    def update(self, minibatch, update_ratio: float):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch
        n_env, batch_size = s_batch.shape[:2]
        s_batch = s_batch.reshape(n_env * batch_size, *self.dim_observation)
        a_batch = a_batch.reshape(n_env * batch_size, self.dim_action)
        r_batch = r_batch.reshape(n_env * batch_size)
        d_batch = d_batch.reshape(n_env * batch_size)
        next_s_batch = next_s_batch.reshape(n_env * batch_size, *self.dim_observation)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        log_std, value_loss, policy_loss, _, _, _ = self.sess.run([self.log_std, self.Value_loss, self.Policy_loss, self.update_policy, self.update_value, self.update_target_vf], feed_dict={
            self.observation: s_batch,
            self.action: a_batch,
            self.target_observation: next_s_batch,
            self.reward: r_batch,
            self.mask: 1 - d_batch})

        # if global_step % 200 == 0:
        #     print(f"log std: {log_std}")

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        log_std, action = self.sess.run([self.log_std, self.sampled_action], feed_dict={self.observation: newobs})
        return action
