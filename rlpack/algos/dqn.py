# -*- coding: utf-8 -*-
from collections import deque
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..common.utils import assert_shape
from .base import Base


class DQN(Base):
    def __init__(self,
                 obs_fn=None,
                 value_fn=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                 update_target_freq=10000,
                 lr=2.5e-4,
                 train_epoch=5,
                 save_path="./log",
                 save_model_freq=10000,
                 log_freq=1000
                 ):
        """Deep Q Networ_value_fnk.
_value_fn
        Keyword Argument_value_fns:
            rnd {int} --_value_fn [description] (default: {1})
            n_env {int} _value_fn-- [description] (default: {1})
            dim_obs {[type]} -- [description] (default: {None})
            dim_act {[type]} -- [description] (default: {None})
            discount {float} -- [description] (default: {0.99})
            save_path {str} -- [description] (default: {"./log"})
            save_model_freq {int} -- [description] (default: {1000})
            update_target_freq {int} -- [description] (default: {10000})
            log_freq {int} -- [description] (default: {1000})
            epsilon_schedule {func} -- epsilon schedule. (default: {lambdax:(1-x)*1})
            lr {[type]} -- [description] (default: {2.5e-4})
        """

        # self._dim_obs = dim_obs

        self._obs_fn = obs_fn
        self._dim_act = dim_act
        self._value_fn = value_fn

        self._discount = discount
        self._update_target_freq = update_target_freq
        self._epsilon_schedule = epsilon_schedule
        self._lr = lr
        self._train_epoch = train_epoch

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.uint8, name="observation")
        # self._observation = tf.to_float(self._observation) / 255.0
        # self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self._observation = self._obs_fn()
        self._action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        self._reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        self._done = tf.placeholder(dtype=tf.float32, shape=[None], name="done")
        # self._next_observation = tf.placeholder(dtype=tf.uint8, shape=[None, *self._dim_obs], name="next_observation")
        # self._next_observation = tf.to_float(self._next_observation) / 255.0
        # self._next_observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="next_observation")
        self._next_observation = self._obs_fn()

        with tf.variable_scope("main/qnet"):
            # self._qvals = self._dense(self._observation)
            self._qvals = self._value_fn(self._observation)

        with tf.variable_scope("target/qnet"):
            # self._target_qvals = self._dense(self._next_observation)
            self._target_qvals = self._value_fn(self._next_observation)

    # def _cnn1d(self, x, cnn1d_hidden_sizes=[(32, 8, 4), (64, 4, 2)], mlp_hidden_sizes=[64, 4]):
    #     for n_filter, stride, pad in cnn1d_hidden_sizes:
    #         x = tf.layers.conv1d(x, n_filter, stride, pad, activation=tf.nn.relu)
    #     x = tf.contrib.layers.flatten(x)
    #     for hsize in mlp_hidden_sizes[:-1]:
    #         x = tf.layers.dense(x, hsize, activation=tf.nn.relu)
    #     return tf.squeeze(tf.layers.dense(x, mlp_hidden_sizes[-1]))

    # def _dense(self, x):
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, self._dim_act)
    #     return x

    def _build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-8)
        trainable_variables = tf.trainable_variables("main/qnet")

        # Compute the state value.
        batch_size = tf.shape(self._done)[0]
        action_index = tf.stack([tf.range(batch_size), self._action], axis=1)
        action_q = tf.gather_nd(self._qvals, action_index)

        # Compute back up.
        q_backup = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * tf.reduce_max(self._target_qvals, axis=1))

        # Compute loss and optimize the object.
        loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))   # 损失值。
        self._train_op = self.optimizer.minimize(loss, var_list=trainable_variables)

        # Update target network.
        def _update_target(old_net, new_net):
            params1 = tf.trainable_variables(old_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.trainable_variables(new_net)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param2.assign(param1))
            return update_ops

        self._update_target_op = _update_target("main/qnet", "target/qnet")
        self._log_op = {"loss": loss}

    def get_action(self, obs):
        """
        Arguments:
            obs -- 两种类型：1. 由np.ndarray构成的tuple类型；2. np.ndarray类型。

        Returns:
            list -- A list of actions.
        """

        if type(obs) is list or type(obs) is tuple:
            batch_size = obs[0].shape[0]
        else:
            batch_size = obs.shape[0]

        q = self.sess.run(self._qvals, feed_dict={self._observation: obs})
        max_a = np.argmax(q, axis=1)
        # Epsilon greedy method.
        global_step = self.sess.run(tf.train.get_global_step())
        actions = np.random.randint(self._dim_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self._epsilon_schedule(global_step)
        actions[idx] = max_a[idx]
        return actions

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch

        for _ in range(self._train_epoch):
            self.sess.run(self._train_op,
                          feed_dict={
                              self._observation: s_batch,
                              self._action: a_batch,
                              self._reward: r_batch,
                              self._done: d_batch,
                              self._next_observation: next_s_batch
                          })

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

        if global_step % self._update_target_freq == 0:
            self.sess.run(self._update_target_op)

        if global_step % self._log_freq == 0:
            log = self.sess.run(self._log_op,
                                feed_dict={
                                    self._observation: s_batch,
                                    self._action: a_batch,
                                    self._reward: r_batch,
                                    self._done: d_batch,
                                    self._next_observation: next_s_batch
                                })
            self.sw.add_scalars("dqn", log, global_step=global_step)
