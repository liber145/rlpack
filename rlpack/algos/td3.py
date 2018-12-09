import math
import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class TD3(Base):
    def __init__(self, config):
        """Implementation of PPO.

        Parameters:
            config: a dictionary for training config.

        Returns:
            None
        """

        self.policy_delay = 2
        self.target_update_rate = 0.995
        self.noise_std = 0.2
        self.explore_noise_std = 0.1

        super().__init__(config)

        self.action_low = np.array([-1.0 for i in range(self.dim_action)])
        self.action_high = np.array([1.0 for i in range(self.dim_action)])

        self.sess.run(self.init_target_policy_op)
        self.sess.run(self.init_target_value_op)

    def build_network(self):
        """Build networks for algorithm."""
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="observation")
        self.action = tf.placeholder(tf.float32, [None, self.dim_action], name="action")

        with tf.variable_scope("policy_net"):
            x = tf.layers.dense(self.observation, 400, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
            self.act = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh, trainable=True)

        with tf.variable_scope("target_policy_net"):
            x = tf.layers.dense(self.observation, 400, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=False)
            self.target_act = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh, trainable=False)

        with tf.variable_scope("value_net"):
            x = tf.concat([self.observation, self.action], axis=1)
            x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
            self.qval_1 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=True))

            x = tf.concat([self.observation, self.action], axis=1)
            x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
            self.qval_2 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=True))

        with tf.variable_scope("value_net", reuse=True):
            x = tf.concat([self.observation, self.act], axis=1)
            x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
            self.qval_act = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=True))

        with tf.variable_scope("target_value_net"):
            x = tf.concat([self.observation, self.action], axis=1)
            x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=False)
            self.target_qval_1 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=False))

            x = tf.concat([self.observation, self.action], axis=1)
            x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=False)
            self.target_qval_2 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=False))

    # Update target network.
    def _update_target(self, new_net, old_net, rho=0):
        params1 = tf.trainable_variables(old_net)
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.global_variables(new_net)
        params2 = sorted(params2, key=lambda v: v.name)
        assert len(params1) == len(params2)
        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param2.assign(rho*param2 + (1-rho)*param1))
        return update_ops

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.target_qval = tf.placeholder(tf.float32, [None], name="target_q_value")
        self.actor_lr = tf.placeholder(tf.float32)
        self.critic_lr = tf.placeholder(tf.float32)
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        self.actor_loss = -tf.reduce_mean(self.qval_act)
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.qval_1, self.target_qval)) + tf.reduce_mean(tf.squared_difference(self.qval_2, self.target_qval))

        self.train_actor_op = self.actor_optimizer.minimize(self.actor_loss, var_list=tf.trainable_variables("policy_net"))
        self.train_critic_op = self.critic_optimizer.minimize(self.critic_loss)

        self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)

        self.update_target_policy_op = self._update_target("target_policy_net", "policy_net", self.target_update_rate)
        self.update_target_value_op = self._update_target("target_value_net", "value_net", self.target_update_rate)

        self.init_target_policy_op = self._update_target("target_policy_net", "policy_net")
        self.init_target_value_op = self._update_target("target_value_net", "value_net")

    def get_action(self, obs):
        """Return actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n, action_dimension).
        """
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        action = self.sess.run(self.act, feed_dict={self.observation: newobs})
        action += np.random.normal(scale=self.explore_noise_std, size=action.shape)
        action = np.clip(action, self.action_low, self.action_high)
        return action

    def update(self, minibatch, update_ratio):
        """Update the algorithm by suing a batch of data.

        Parameters:
            - minibatch: A list of ndarray containing a minibatch of state, action, reward, done.

                - state shape: (n_env, batch_size+1, state_dimension)
                - action shape: (n_env, batch_size, state_dimension)
                - reward shape: (n_env, batch_size)
                - done shape: (n_env, batch_size)

            - update_ratio: float scalar in (0, 1).

        Returns:
            - training infomation.
        """
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        mb_s, mb_a, mb_target = [], [], []

        n_env = s_batch.shape[0]
        for i in range(n_env):
            batch_size = d_batch[i].shape[0]
            assert batch_size == 128   # TODO: remove

            # Compute target value.
            next_a_batch = self.sess.run(self.target_act, feed_dict={self.observation: next_s_batch[i, :]}) + np.random.normal(scale=self.noise_std, size=(batch_size, self.dim_action))
            assert next_a_batch.shape == (batch_size, self.dim_action)
            next_a_batch = np.clip(next_a_batch, self.action_low, self.action_high)
            q1, q2 = self.sess.run([self.target_qval_1, self.target_qval_2], feed_dict={self.observation: next_s_batch[i, :], self.action: next_a_batch})
            assert q1.shape == (batch_size,) and q2.shape == (batch_size,)
            target_qval = r_batch[i, :] + self.discount * (1 - d_batch[i, :]) * np.minimum(q1, q2)
            assert target_qval.shape == (batch_size,)

            mb_s.append(s_batch[i, :])
            mb_a.append(a_batch[i, :])
            mb_target.append(target_qval)

        mb_s = np.concatenate(mb_s)
        mb_a = np.concatenate(mb_a)
        mb_target = np.concatenate(mb_target)

        _, global_step = self.sess.run([self.increment_global_step, tf.train.get_global_step()])

        # Update critic net.
        self.sess.run(self.train_critic_op, feed_dict={
                                                self.observation: mb_s,
                                                self.action: mb_a,
                                                self.target_qval: mb_target,
                                                self.critic_lr: self.value_lr_schedule(update_ratio)})

        # Update actor net.
        if global_step % self.policy_delay == 0:
            self.sess.run(self.train_actor_op, feed_dict={self.observation: mb_s, self.actor_lr: self.policy_lr_schedule(update_ratio)})
            self.sess.run([self.update_target_value_op, self.update_target_policy_op])

        if global_step % self.save_model_freq == 0:
            self.save_model()
