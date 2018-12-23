# -*- coding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf

from .base import Base


class DDPG(Base):
    """Deep Deterministic Policy Gradient."""

    def __init__(self,
                 rnd=1,
                 n_env=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_model_freq=1000,
                 save_path="./log",
                 update_target_freq=10000,
                 policy_lr=2.5e-4,
                 value_lr=3e-4,
                 action_low=-1.0,
                 action_high=1.0,
                 ):
        self.n_env = n_env
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.discount = discount

        self.policy_lr = policy_lr
        self.value_lr = value_lr
        self.action_low = action_low
        self.action_high = action_high

        self.update_target_freq = update_target_freq
        self.save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def build_network(self):
        """Build networks for algorithm."""
        # Build placeholders.
        self.observation_ph = tf.placeholder(tf.float32, [None, *self.dim_obs], "observation")
        self.action_ph = tf.placeholder(tf.float32, (None, self.dim_act), "action")

        # Build Q-value net.
        with tf.variable_scope("qval_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=True)
            y = tf.layers.dense(self.action_ph, 64, activation=tf.nn.relu, trainable=True)
            z = tf.concat([x, y], axis=1)
            self.qval = tf.squeeze(tf.layers.dense(z, 1, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)), trainable=True))

        with tf.variable_scope("dummy_qval_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=False)
            y = tf.layers.dense(self.action_ph, 64, activation=tf.nn.relu, trainable=False)
            z = tf.concat([x, y], axis=1)
            self.dummy_qval = tf.squeeze(tf.layers.dense(z, 1, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)), trainable=False))

        # Build action net.
        with tf.variable_scope("act_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=True)
            self.action = tf.layers.dense(x, self.dim_act, activation=tf.nn.tanh, trainable=True)

        with tf.variable_scope("dummy_act_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=False)
            self.dummy_action = tf.layers.dense(x, self.dim_act, activation=tf.nn.tanh, trainable=False)

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdamOptimizer(self.policy_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.value_lr)
        self.target_qval_ph = tf.placeholder(tf.float32, (None,), "next_state_qval")
        self.grad_q_act_ph = tf.placeholder(tf.float32, (None, self.dim_act), "grad_q_act")

        # ---------- Build Policy Algorithm ----------
        # Compute gradient of qval with respect to action.
        self.grad_q_a = tf.gradients(self.qval, self.action_ph)

        # Compute update direction of policy parameter.
        batch_size = tf.to_float(tf.shape(self.observation_ph)[0])
        actor_vars = tf.trainable_variables("act_net")
        grad_surr = tf.gradients(self.action / batch_size, actor_vars, -self.grad_q_act_ph)

        # Update actor parameters.
        self.train_actor_op = self.optimizer.apply_gradients(zip(grad_surr, actor_vars))

        # ---------- Build Value Algorithm ----------
        critic_vars = tf.trainable_variables("qval_net")
        self.value_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.qval - self.target_qval_ph)))

        self.train_critic_op = self.critic_optimizer.minimize(self.value_loss, var_list=critic_vars)

    def update(self, minibatch, update_ratio=None):
        """Update the algorithm by suing a batch of data.

        Parameters:
            - minibatch: A list of ndarray containing a minibatch of state, action, reward, done.

                - state shape: (n_env, batch_size, state_dimension)
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

            # Compute target value.
            next_a_batch = self.sess.run(self.dummy_action, feed_dict={self.observation_ph: next_s_batch[i, :]})
            assert next_a_batch.shape == (batch_size, self.dim_act)
            next_a_batch = np.clip(next_a_batch, self.action_low, self.action_high)
            qval = self.sess.run(self.dummy_qval, feed_dict={self.observation_ph: next_s_batch[i, :], self.action_ph: next_a_batch})
            assert qval.shape == (batch_size,)
            target_qval = r_batch[i, :] + self.discount * (1 - d_batch[i, :]) * qval
            assert target_qval.shape == (batch_size,)

            mb_s.append(s_batch[i, :])
            mb_a.append(a_batch[i, :])
            mb_target.append(target_qval)

        mb_s = np.concatenate(mb_s)
        mb_a = np.concatenate(mb_a)
        mb_target = np.concatenate(mb_target)

        # Update actor.
        grad = self.sess.run(self.grad_q_a, feed_dict={self.observation_ph: mb_s, self.action_ph: mb_a})[0]
        self.sess.run(self.train_actor_op, feed_dict={self.observation_ph: mb_s, self.action_ph: mb_a, self.grad_q_act_ph: grad})

        # Update critic.
        _, loss = self.sess.run([self.train_critic_op, self.value_loss], feed_dict={
            self.observation_ph: mb_s,
            self.action_ph: mb_a,
            self.target_qval_ph: mb_target})

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        if global_step % self.update_target_freq == 0:
            self._copy_parameters("qval_net", "dummy_qval_net")
            self._copy_parameters("act_net", "dummy_act_net")

        if global_step % self.save_model_freq == 0:
            self.save_model()

    def _copy_parameters(self, netnew, netold):
        """Copy parameters from netnew to netold.

        Parameters:
            - netold: string
            - netnew: string
        """

        oldvars = tf.trainable_variables(netold)
        newvars = tf.trainable_variables(netnew)

        assign_op = [x.assign(y) for x, y in zip(oldvars, newvars)]
        self.sess.run(assign_op)

    def _ou_fn(self, x, mu=0.0, theta=0.15, sigma=0.2):
        return theta * (mu - x) + sigma * np.random.randn(1)

    def get_action(self, obs):
        """Return actions according to the given observation.

        Parameters:
            - obs: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n, action_dimension).
        """

        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        best_actions = self.sess.run(self.action, feed_dict={self.observation_ph: newobs})
        actions = best_actions + self._ou_fn(best_actions)
        actions = np.clip(actions, -1, 1)

        if obs.ndim == 1 or obs.ndim == 3:
            actions = actions[0]
        return actions

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]
        # assert n_sample == self.n_env * self.trajectory_length

        index = np.arange(n_sample)
        np.random.shuffle(index)

        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index] if x.ndim == 1 else x[span_index, :] for x in data_batch]
