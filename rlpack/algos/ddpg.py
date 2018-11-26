# -*- coding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf

from .base import Base


class DDPG(Base):
    """Deep Deterministic Policy Gradient."""

    def __init__(self, config):
        self.epsilon = 0.1
        self.lr = config.lr
        self.gae = config.gae
        super().__init__(config)

    def build_network(self):
        # Build placeholders.
        self.observation_ph = tf.placeholder(tf.float32, [None, *self.dim_observation], "observation")
        self.action_ph = tf.placeholder(tf.float32, (None, self.dim_action), "action")

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
            self.action = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh, trainable=True)

        with tf.variable_scope("dummy_act_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=False)
            self.dummy_action = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh, trainable=False)

    def build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr)
        self.target_qval_ph = tf.placeholder(tf.float32, (None,), "next_state_qval")
        self.grad_q_act_ph = tf.placeholder(tf.float32, (None, self.dim_action), "grad_q_act")

        # ---------- Build Policy Algorithm ----------
        # Compute gradient of qval with respect to action.
        self.grad_q_a = tf.gradients(self.qval, self.action_ph)

        # Compute update direction of policy parameter.
        actor_vars = tf.trainable_variables("act_net")
        grad_surr = tf.gradients(self.action / self.batch_size, actor_vars, -self.grad_q_act_ph)

        # Update actor parameters.
        self.train_actor_op = self.optimizer.apply_gradients(zip(grad_surr, actor_vars), global_step=tf.train.get_global_step())

        # ---------- Build Value Algorithm ----------
        critic_vars = tf.trainable_variables("qval_net")
        self.value_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.qval - self.target_qval_ph)))

        self.train_critic_op = self.critic_optimizer.minimize(self.value_loss, var_list=critic_vars)

    def update(self, minibatch, update_ratio=None):
        s_batch, a_batch, r_batch, d_batch = minibatch

        s_batch = s_batch[:, :-1, :]
        r_batch = r_batch[:, :-1]
        d_batch = d_batch[:, :-1]

        # print(s_batch.shape)
        # print(a_batch.shape)
        # print(r_batch.shape)
        # print(d_batch.shape)
        # print(d_batch[1, :-1].shape[0])

        tmp_n = d_batch[0].shape[0]
        target_value_batch = np.empty([self.n_env, tmp_n], dtype=np.float32)
        advantage_value_batch = np.empty([self.n_env, tmp_n], dtype=np.float32)

        for i in range(self.n_env):
            action_value_batch = self.sess.run(self.qval, feed_dict={self.observation_ph: s_batch[i, :, :], self.action_ph: a_batch[i, :]})
            delta_value_batch = r_batch[i, :] + self.discount * (1 - d_batch[i, :]) * action_value_batch[1:] - action_value_batch[:-1]

            # print("--------")
            # print(advantage_value_batch.shape)
            # print(action_value_batch.shape)
            # print(delta_value_batch.shape)
            # print(tmp_n)
            # input()

            last_advantage = 0
            for t in reversed(range(tmp_n)):
                advantage_value_batch[i, t] = delta_value_batch[t] + self.discount * self.gae * (1 - d_batch[i, t]) * last_advantage
                last_advantage = advantage_value_batch[i, t]

            target_value_batch[i, :] = action_value_batch[:-1] + advantage_value_batch[i, :]

        s_batch = s_batch[:, :-1, ...].reshape(self.n_env * tmp_n, *self.dim_observation)
        a_batch = a_batch[:, :-1].reshape(self.n_env * tmp_n, self.dim_action)
        target_value_batch = target_value_batch.reshape(self.n_env * tmp_n)

        # next_action_batch = self.sess.run(self.dummy_action, feed_dict={self.observation_ph: next_s_batch})
        # next_qval_batch = self.sess.run(self.dummy_qval, feed_dict={self.observation_ph: next_s_batch, self.action_ph: next_action_batch})
        # target_qval_batch = r_batch + (1 - d_batch) * self.discount * next_qval_batch

        batch_generator = self._generator([s_batch, a_batch])
        while True:
            try:

                mb_s, mb_a = next(batch_generator)
                grad = self.sess.run(self.grad_q_a, feed_dict={self.observation_ph: mb_s, self.action_ph: mb_a})[0]

                self.sess.run(self.train_actor_op, feed_dict={self.observation_ph: mb_s, self.action_ph: mb_a, self.grad_q_act_ph: grad})

            except StopIteration:
                del batch_generator
                break

        batch_generator = self._generator([s_batch, a_batch, target_value_batch])
        while True:
            try:
                mb_s, mb_a, mb_target = next(batch_generator)

                # Update critic.

                _, loss, global_step = self.sess.run([self.train_critic_op, self.value_loss, tf.train.get_global_step()],
                                                     feed_dict={
                    self.observation_ph: mb_s,
                    self.action_ph: mb_a,
                    self.target_qval_ph: mb_target})

            except StopIteration:
                del batch_generator
                break

        self._copy_parameters("qval_net", "dummy_qval_net")
        self._copy_parameters("act_net", "dummy_act_net")

        return {"critic_loss": loss, "global_step": global_step}

    def _copy_parameters(self, netnew, netold):
        """Copy parameters from netnew to netold.

        Parameters:
            netold: string
            netnew: string
        """

        oldvars = tf.trainable_variables(netold)
        newvars = tf.trainable_variables(netnew)

        assign_op = [x.assign(y) for x, y in zip(oldvars, newvars)]
        self.sess.run(assign_op)

    def _ou_fn(self, x, mu=0.0, theta=0.15, sigma=0.2):
        return theta * (mu - x) + sigma * np.random.randn(1)

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        best_actions = self.sess.run(self.action, feed_dict={self.observation_ph: newobs})
        actions = best_actions + self._ou_fn(best_actions)
        actions = np.clip(actions, [-1, -1], [1, 1])

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
