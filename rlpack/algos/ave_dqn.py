# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class AveDQN(Base):
    """Deep Q Network."""

    def __init__(self,
                 obs_fn=None,
                 value_fn=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=1000,
                 update_target_freq=10000,
                 epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                 lr=2.5e-4,
                 train_epoch=1,
                 n_net=5):

        self._obs_fn = obs_fn
        self._value_fn = value_fn
        self._dim_act = dim_act
        self._discount = discount
        self._save_model_freq = save_model_freq
        self._update_target_freq = update_target_freq
        self._log_freq = log_freq
        self._epsilon_schedule = epsilon_schedule
        self._lr = lr
        self._train_epoch = train_epoch
        self._n_net = n_net

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

        self._target_qvals = []
        for i in range(self._n_net):
            with tf.variable_scope(f"target_{i}/qnet"):
                self._target_qvals.append(self._value_fn(self._next_observation))

    # def _dense(self, t):
    #     x = tf.layers.dense(t, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, self._dim_act)
    #     return x

    def _build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1.5e-8)
        trainable_variables = tf.trainable_variables("main/qnet")

        # Compute the state value.
        batch_size = tf.shape(self._observation)[0]
        action_index = tf.stack([tf.range(batch_size), self._action], axis=1)
        action_q = tf.gather_nd(self._qvals, action_index)
        assert_shape(action_q, [None])

        # Compute back up.
        ave_q = tf.add_n(self._target_qvals) / self._n_net
        assert_shape(tf.reduce_max(ave_q, axis=1), [None])
        q_backup = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * tf.reduce_max(ave_q, axis=1))

        # Compute loss and optimize the object.
        loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))   # 损失值。
        self._train_op = self.optimizer.minimize(loss, var_list=trainable_variables)

        # Update target network.
        update_target_operation = []
        for i in reversed(range(1, self._n_net)):  # i=0表示最近的模型。
            with tf.control_dependencies(update_target_operation):
                update_target_operation.append(self._update_target(f"target_{i}/qnet", f"target_{i-1}/qnet"))

        with tf.control_dependencies(update_target_operation):
            update_target_operation.append(self._update_target("target_0/qnet", "main/qnet"))

        self.update_target_op = update_target_operation
        self._log_op = {"loss": loss}

    def _update_target(self, net1, net2):
        """net1 = net2 

        Arguments:
            net1 {str} -- variable scope of net1.
            net2 {str} -- variable scope of net2
        """
        params1 = tf.trainable_variables(net1)
        params1 = sorted(params1, key=lambda k: k.name)
        params2 = tf.trainable_variables(net2)
        params2 = sorted(params2, key=lambda k: k.name)
        assert len(params1) == len(params2)
        return tf.group([x1.assign(x2) for x1, x2 in zip(params1, params2)])

    def get_action(self, obs):
        """Get actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape(n, state_dimension).

        Returns:
            - An ndarray for action with shape(n).
        """
        q = self.sess.run(self._qvals, feed_dict={self._observation: obs})
        max_a = np.argmax(q, axis=1)

        # Epsilon greedy method.
        global_step = self.sess.run(tf.train.get_global_step())
        batch_size = obs.shape[0]
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

        # Store model.
        if global_step % self._save_model_freq == 0:
            self.save_model()

        # Update target policy.
        if global_step % self._update_target_freq == 0:
            self.sess.run(self.update_target_op)

        if global_step % self._log_freq == 0:
            log = self.sess.run(self._log_op,
                                feed_dict={
                                    self._observation: s_batch,
                                    self._action: a_batch,
                                    self._reward: r_batch,
                                    self._done: d_batch,
                                    self._next_observation: next_s_batch
                                })
            self.sw.add_scalars("avedqn", log, global_step=global_step)
