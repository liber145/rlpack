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
                 rnd=1,
                 n_env=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=1000,
                 update_target_freq=10000,
                 epsilon_schedule=lambda x: min(x/5e6, 1.0),
                 lr=2.5e-4,
                 n_net=5,
                 ridge_coef=0.0,
                 weight_low=-3.0,
                 weight_high=5.0
                 ):

        self.n_env = n_env
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.discount = discount
        self.save_model_freq = save_model_freq
        self.update_target_freq = update_target_freq
        self.log_freq = log_freq
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule(0)
        self.lr = lr
        self.n_net = n_net
        self.ridge_coef = ridge_coef
        self.weight_low = weight_low
        self.weight_high = weight_high

        super().__init__(save_path=save_path, rnd=rnd)

    def _qnet(self, name: str, obs, trainable=False):
        with tf.variable_scope(name):
            # x = tf.layers.conv1d(obs, 32, 8, 4, activation=tf.nn.relu, trainable=trainable)
            # x = tf.layers.conv1d(x, 64, 4, 2, activation=tf.nn.relu, trainable=trainable)
            # x = tf.contrib.layers.flatten(x)
            # x = tf.layers.dense(x, 256, activation=tf.nn.relu, trainable=trainable)
            # qvals = tf.layers.dense(x, self.dim_act, trainable=trainable)
            qvals = cnn1d(obs, cnn1d_hidden_sizes=[(32, 8, 4), (64, 4, 2)], mlp_hidden_sizes=[256, self.dim_act])
        return qvals

    def build_network(self):
        """Build networks for algorithm."""
        self.observation = tf.placeholder(shape=[None, *self.dim_obs], dtype=tf.uint8, name="observation")
        self.observation = tf.to_float(self.observation) / 255.0
        self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        self.done = tf.placeholder(dtype=tf.float32, shape=[None], name="done")
        self.next_observation = tf.placeholder(dtype=tf.uint8, shape=[None, *self.dim_obs], name="next_observation")
        self.next_observation = tf.to_float(self.next_observation) / 255.0

        self.qvals = self._qnet("main/qnet", self.observation, trainable=True)

        self.targ_q = []
        for i in range(self.n_net):
            self.targ_q.append(self._qnet(f"target_{i}/qnet", self.next_observation, trainable=True))

    def _compute_weights(self, action_q):
        """
        Compute coefficients for anderson mixing.
        """
        tds = []
        for i in range(self.n_net):
            tds.append(self.reward + self.discount * (1 - self.done) * tf.reduce_max(self.targ_q[i], axis=1) - action_q)

        td_mat = tf.stack(tds)
        assert_shape(td_mat, [self.n_net, None])

        print("ridge coef:", self.ridge_coef)
        # input()
        mat = tf.matmul(td_mat, td_mat, transpose_b=True) + self.ridge_coef * tf.eye(self.n_net)
        inv_mat = tf.matrix_inverse(mat)
        ones = tf.ones((self.n_net, 1))
        inv_mat_one = tf.matmul(inv_mat, ones)
        weights = inv_mat_one / tf.matmul(ones, inv_mat_one, transpose_a=True)
        weights = tf.squeeze(weights)
        assert_shape(weights, [self.n_net])

        tf.clip_by_value(weights, self.weight_low, self.weight_high)
        weights = weights / tf.norm(weights)

        return weights

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1.5e-8)
        trainable_variables = tf.trainable_variables("main/qnet")

        # Compute the state value.
        batch_size = tf.shape(self.observation)[0]
        action_index = tf.stack([tf.range(batch_size), self.action], axis=1)
        assert_shape(action_index, [None, 2])
        action_q = tf.gather_nd(self.qvals, action_index)
        assert_shape(action_q, [None])

        # Compute back up.
        self.weights = self._compute_weights(action_q)
        q_backup = tf.add_n([self.targ_q[i] * self.weights[i] for i in range(self.n_net)])
        assert_shape(q_backup, [None, self.dim_act])
        q_backup = tf.stop_gradient(self.reward + self.discount * (1 - self.done) * tf.reduce_max(q_backup, axis=1))
        assert_shape(q_backup, [None])

        # Compute loss and optimize the object.
        self.loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))   # 损失值。
        self.train_op = self.optimizer.minimize(self.loss, var_list=trainable_variables)

        # Update target network.
        update_target_operation = []
        for i in reversed(range(1, self.n_net)):
            if len(update_target_operation) == 0:
                print("i", i)
                update_target_operation.append(self._update_target(f"target_{i}/qnet", f"target_{i-1}/qnet"))
            else:
                with tf.control_dependencies(update_target_operation):
                    tmp = self._update_target(f"target_{i}/qnet", f"target_{i-1}/qnet")
                    update_target_operation.append(tmp)

        with tf.control_dependencies(update_target_operation):
            tmp = self._update_target("target_0/qnet", "main/qnet")
            update_target_operation.append(tmp)

        self.update_target_op = update_target_operation

        self.max_qval = tf.reduce_max(self.qvals)

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
        # if obs.ndim == 1 or obs.ndim == 3:
        #     newobs = np.array(obs)[np.newaxis, :]
        # else:
        #     assert obs.ndim == 2 or obs.ndim == 4
        #     newobs = obs

        q = self.sess.run(self.qvals, feed_dict={self.observation: obs})
        max_a = np.argmax(q, axis=1)

        # Epsilon greedy method.
        batch_size = obs.shape[0]
        actions = np.random.randint(self.dim_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) < self.epsilon
        actions[idx] = max_a[idx]

        # if obs.ndim == 1:
        #     actions = actions[0]
        return actions

    def update(self, minibatch, update_step: int):
        """Update the algorithm by suing a batch of data.

        Parameters:
            - minibatch:  a list of ndarray containing a minibatch of state, action, reward, done, next_state.

                - state shape: (n_env, batch_size, state_dimension)
                - action shape: (n_env, batch_size)
                - reward shape: (n_env, batch_size)
                - done shape: (n_env, batch_size)
                - next_state shape: (n_env, batch_size, state_dimension)

            - update_ratio: a float scalar in (0, 1).

        Returns:
            - training infomation.
        """

        self.epsilon = self.epsilon_schedule(update_step)

        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        n_env, batch_size = s_batch.shape[:2]
        s_batch = s_batch.reshape(n_env * batch_size, *self.dim_obs)
        a_batch = a_batch.reshape(n_env * batch_size)
        r_batch = r_batch.reshape(n_env * batch_size)
        d_batch = d_batch.reshape(n_env * batch_size)
        next_s_batch = next_s_batch.reshape(n_env * batch_size, *self.dim_obs)

        # mb_s, mb_a, mb_target = [], [], []

        # n_env = s_batch.shape[0]
        # for i in range(n_env):
        #     target_next_q_vals = self.sess.run(self.target_qvals, feed_dict={self.observation: next_s_batch[i, :]})
        #     target_batch = r_batch[i, :] + (1 - d_batch[i, :]) * self.discount * target_next_q_vals.max(axis=1)
        #     mb_target.append(target_batch)

        #     mb_s.append(s_batch[i, :])
        #     mb_a.append(a_batch[i, :])

        # mb_s = np.concatenate(mb_s)
        # mb_a = np.concatenate(mb_a)
        # mb_target = np.concatenate(mb_target)

        _, loss, max_q_val, weights = self.sess.run(
            [self.train_op,
             self.loss,
             self.max_qval,
             self.weights],
            feed_dict={
                self.observation: s_batch,
                self.action: a_batch,
                self.reward: r_batch,
                self.done: d_batch,
                self.next_observation: next_s_batch
            }
        )

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        # Store model.
        if global_step % self.save_model_freq == 0:
            self.save_model()

        # Update target policy.
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        if global_step % self.log_freq == 0:
            tqdm.write(f"weights: {weights}, loss: {loss}, epsilon: {self.epsilon}")

        return {"loss": loss, "max_q_value": max_q_val, "global_step": global_step}
