# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from ..common.utils import assert_shape
from ..common.network import mlp, cnn1d, cnn2d
from .base import Base


class AADQN(Base):
    """Deep Q Network."""

    def __init__(self,
                 obs_fn=None,
                 value_fn=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 ridge_coef=None,
                 n_net=5,
                 weight_low=-3.0,
                 weight_high=5.0,
                 lr=2.5e-4,
                 max_grad_norm=40,
                 epsilon_schedule=lambda x: max(0.1, (1e6-x) / 1e6),
                 update_target_freq=10000,
                 train_epoch=1,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=1000,
                 ):

        self._obs_fn = obs_fn
        self._value_fn = value_fn
        self._dim_act = dim_act

        self._discount = discount
        self._n_net = n_net
        self._ridge_coef = ridge_coef

        self._update_target_freq = update_target_freq
        self._epsilon_schedule = epsilon_schedule
        self._lr = lr
        self._max_grad_norm = max_grad_norm
        self._train_epoch = train_epoch
        self._weight_low = weight_low
        self._weight_high = weight_high

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.uint8, name="observation")
        # self._observation = tf.to_float(self._observation) / 255.0
        # self._observation = tf.placeholder(dtype=tf.float32, shape=[None, *self._dim_obs], name="observation")
        self._observation = self._obs_fn()
        self._action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        self._reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        self._done = tf.placeholder(dtype=tf.float32, shape=[None], name="done")
        # self._next_observation = tf.placeholder(dtype=tf.uint8, shape=[None, *self._dim_obs], name="next_observation")
        # self._next_observation = tf.to_float(self._next_observation) / 255.0
        # self._next_observation = tf.placeholder(dtype=tf.float32, shape=[None, *self._dim_obs], name="next_observation")
        self._next_observation = self._obs_fn()

        with tf.variable_scope("main/qnet"):
            self._qvals = self._value_fn(self._observation)

        self._target_qvals = []
        for i in range(self._n_net):
            with tf.variable_scope(f"target_{i}/qnet"):
                self._target_qvals.append(self._value_fn(self._next_observation))

    # def _dense(self, x):
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, self._dim_act)
    #     return x

    # def _conv1d(self, obs):
    #     x = tf.layers.conv1d(obs, filters=32, kernel_size=8, strides=4, activation=tf.nn.relu)
    #     x = tf.layers.conv1d(x, filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
    #     x = tf.layers.conv1d(x, filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)
    #     x = tf.layers.flatten(x)
    #     x = tf.layers.dense(x, units=256, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, units=self._dim_act)
    #     return x

    def _build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdadeltaOptimizer(1.0)
        trainable_variables = tf.trainable_variables("main/qnet")

        # Compute the state value.
        batch_size = tf.shape(self._observation)[0]
        action_index = tf.stack([tf.range(batch_size), self._action], axis=1)
        assert_shape(action_index, [None, 2])
        action_q = tf.gather_nd(self._qvals, action_index)
        assert_shape(action_q, [None])

        # Compute back up.
        self.weights, self.mat, self.inv_mat = self._compute_weights(action_q)
        q_backup = tf.add_n([self._target_qvals[i] * self.weights[i] for i in range(self._n_net)])
        assert_shape(q_backup, [None, self._dim_act])
        q_backup = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * tf.reduce_max(q_backup, axis=1))
        assert_shape(q_backup, [None])

        # Compute loss and optimize the object.
        loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))   # 损失值。
        grads = tf.gradients(loss, trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        self._train_op = self.optimizer.apply_gradients(zip(clipped_grads, trainable_variables))
        # self._train_op = self.optimizer.minimize(loss, var_list=trainable_variables)

        # Update target network.
        update_target_operation = []
        for i in reversed(range(1, self._n_net)):  # i=0表示最近的模型。
            with tf.control_dependencies(update_target_operation):
                update_target_operation.append(self._update_target(f"target_{i}/qnet", f"target_{i-1}/qnet"))

        with tf.control_dependencies(update_target_operation):
            update_target_operation.append(self._update_target("target_0/qnet", "main/qnet"))

        self._update_target_op = update_target_operation
        self._log_op = {"loss": loss}

    def _compute_weights(self, action_q):
        """
        Compute coefficients for anderson mixing.
        """
        tds = []
        for i in range(self._n_net):
            tds.append(self._reward + self._discount * (1 - self._done) * tf.reduce_max(self._target_qvals[i], axis=1) - action_q)

        td_mat = tf.stack(tds)
        assert_shape(td_mat, [self._n_net, None])

        tmp = tf.matmul(td_mat, td_mat, transpose_b=True)
        tmax = tf.reduce_max(tmp)
        ridge_coef = tmax * 1e-3 if self._ridge_coef is None else self._ridge_coef
        mat = tmp + ridge_coef * tf.eye(self._n_net)
        inv_mat = tf.matrix_inverse(mat)
        ones = tf.ones((self._n_net, 1))
        inv_mat_one = tf.matmul(inv_mat, ones)
        weights = inv_mat_one / tf.matmul(ones, inv_mat_one, transpose_a=True)
        weights = tf.squeeze(weights)
        assert_shape(weights, [self._n_net])

        weights = tf.clip_by_value(weights, self._weight_low, self._weight_high)
        weights = weights / tf.reduce_sum(weights)

        return weights, mat, inv_mat

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
        """
        Arguments:
            obs {np.ndarray} -- A list of observation.

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
        epsilon = self._epsilon_schedule(global_step)
        batch_size = obs.shape[0]
        actions = np.random.randint(self._dim_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) > epsilon
        actions[idx] = max_a[idx]
        return actions

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch

        for _ in range(self._train_epoch):
            weight, mat, inv_mat, _ = self.sess.run([self.weights, self.mat, self.inv_mat, self._train_op],
                                                    feed_dict={
                self._observation: s_batch,
                self._action: a_batch,
                self._reward: r_batch,
                self._done: d_batch,
                self._next_observation: next_s_batch
            })

            # print(">>>> weight:", weight)
            # print(">>>> mat:", mat)
            # print(">>>> inv_mat:", inv_mat)
            # input()

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
            self.sw.add_scalars("aadqn", log, global_step=global_step)
