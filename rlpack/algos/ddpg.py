# -*- coding: utf-8 -*-
import math

import numpy as np
import tensorflow as tf

from .base import Base


class DDPG(Base):
    """Deep Deterministic Policy Gradient."""

    def __init__(self,
                 dim_obs=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                 save_model_freq=1000,
                 save_path="./log",
                 update_target_freq=10000,
                 policy_lr=2.5e-4,
                 value_lr=3e-4,
                 lr=1e-4,
                 log_freq=10,
                 train_epoch=1,
                 ):
        self._dim_obs = dim_obs
        self._dim_act = dim_act

        self._discount = discount
        self._epsilon_schedule = epsilon_schedule

        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._lr = lr

        self._update_target_freq = update_target_freq
        self._train_epoch = train_epoch

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # Build placeholders.
        self._observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")
        self._action = tf.placeholder(tf.int32, (None,), name="action")
        self._reward = tf.placeholder(tf.float32, [None], name="reward")
        self._done = tf.placeholder(tf.float32, [None], name="done")
        self._next_observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="next_observation")

        with tf.variable_scope("main/policy"):
            self._p_act = self._policy_net(self._observation)

        with tf.variable_scope("main/value"):
            self._qvals = self._value_net(self._observation)

        with tf.variable_scope("target/policy"):
            self._target_p_act = self._policy_net(self._next_observation)

        with tf.variable_scope("target/value"):
            self._target_qvals = self._value_net(self._next_observation)

        # with tf.variable_scope("main"):
        #     self._p_act, self._qvals = self._dense(self._observation)

        # with tf.variable_scope("target"):
        #     self._target_p_act, self._target_qvals = self._dense(self._next_observation)

    def _value_net(self, obs):
        x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 64, activation=tf.nn.relu)
        return tf.layers.dense(x, self._dim_act)

    def _dense(self, obs):
        x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 64, activation=tf.nn.relu)
        return tf.layers.dense(x, self._dim_act, activation=tf.nn.softmax), tf.layers.dense(x, self._dim_act)

    def _policy_net(self, obs):
        x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 64, activation=tf.nn.relu)
        return tf.layers.dense(x, self._dim_act, activation=tf.nn.softmax)

    def _build_algorithm(self):
        """Build networks for algorithm."""
        self._policy_optimizer = tf.train.AdamOptimizer(self._policy_lr)
        self._value_optimizer = tf.train.AdamOptimizer(self._value_lr)
        policy_variables = tf.trainable_variables("main/policy")
        value_variables = tf.trainable_variables("main/value")

        # self._optimizer = tf.train.AdamOptimizer(self._lr)
        # trainable_variables = tf.trainable_variables("main")

        nsample = tf.shape(self._observation)[0]
        mean_qvals = tf.reduce_sum(self._qvals * self._p_act, axis=1)
        policy_loss = -tf.reduce_mean(mean_qvals)

        qvals2 = tf.gather_nd(self._qvals, tf.stack([tf.range(nsample), self._action], axis=1))
        target_categorical_dist = tf.distributions.Categorical(probs=self._target_p_act)
        target_act = target_categorical_dist.sample()
        target_qvals = tf.gather_nd(self._target_qvals, tf.stack([tf.range(nsample), target_act], axis=1))
        qbackup = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * target_qvals)
        value_loss = tf.reduce_mean(tf.squared_difference(qvals2, qbackup))

        self._policy_train_op = self._policy_optimizer.minimize(policy_loss, var_list=policy_variables)
        self._value_train_op = self._value_optimizer.minimize(value_loss, var_list=value_variables)

        # total_loss = policy_loss + 1.0 * value_loss
        # self._train_op = self._optimizer.minimize(total_loss, var_list=trainable_variables)

        def _update_target(net1, net2):
            variables1 = tf.trainable_variables(net1)
            variables1 = sorted(variables1, key=lambda v: v.name)
            variables2 = tf.trainable_variables(net2)
            variables2 = sorted(variables2, key=lambda v: v.name)
            assert len(variables1) == len(variables2)
            return [v1.assign(v2) for v1, v2 in zip(variables1, variables2)]

        self._update_target_op = tf.group(_update_target("target/policy", "main/policy") + _update_target("target/value", "main/value"))

        # self._update_target_op = tf.group(_update_target("target", "main"))

        self._log_op = {"policy_loss": policy_loss, "value_loss": value_loss}

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch

        self.sess.run([self._policy_train_op, self._value_train_op],
                      feed_dict={
                          self._observation: s_batch,
                          self._action: a_batch,
                          self._reward: r_batch,
                          self._done: d_batch,
                          self._next_observation: next_s_batch
        })

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._update_target_freq == 0:
            self.sess.run(self._update_target_op)

        if global_step % self._save_model_freq == 0:
            self.save_model()

        if global_step % self._log_freq == 0:
            log = self.sess.run(self._log_op,
                                feed_dict={
                                    self._observation: s_batch,
                                    self._action: a_batch,
                                    self._reward: r_batch,
                                    self._done: d_batch,
                                    self._next_observation: next_s_batch

                                })
            self.sw.add_scalars("ddpg", log, global_step=global_step)

    def get_action(self, obs):
        """Return actions according to the given observation.

        Parameters:
            - obs: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n, action_dimension).
        """
        p_act = self.sess.run(self._p_act, feed_dict={self._observation: obs})
        # act = np.argmax(p_act, axis=1)

        nsample, nact = p_act.shape
        # global_step = self.sess.run(tf.train.get_global_step())
        # idx = np.random.uniform(size=nsample) > self._epsilon_schedule(global_step)
        # actions = np.random.randint(self._dim_act, size=nsample)
        # actions[idx] = act[idx]

        return [np.random.choice(nact, p=p_act[i, :]) for i in range(nsample)]
