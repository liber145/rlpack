import numpy as np
import tensorflow as tf

import scipy
import tensorflow as tf

from .base import Base

from .base import Base


class DoubleDQN(Base):
    def __init__(self, config):
        self.lr = config.value_lr_schedule(0)
        self.epsilon_schedule = config.epsilon_schedule
        self.epsilon = self.epsilon_schedule(0)
        self.update_target_freq = config.update_target_freq
        super().__init__(config)

    def build_network(self):
        """Build networks for algorithm."""
        self.observation = tf.placeholder(shape=[None, *self.dim_observation], dtype=tf.float32, name="observation")

        with tf.variable_scope("qnet"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self.qvals = tf.layers.dense(x, self.dim_action)

        with tf.variable_scope("target_qnet"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu, trainable=False)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=False)
            self.target_qvals = tf.layers.dense(x, self.dim_action, trainable=False)

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.target = tf.placeholder(shape=[None], dtype=tf.float32, name="target")
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1.5e-8)
        trainable_variables = tf.trainable_variables("qnet")

        batch_size = tf.shape(self.observation)[0]
        gather_indices = tf.range(batch_size) * self.dim_action + self.action
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, action_q))
        self.train_op = self.optimizer.minimize(self.loss, var_list=trainable_variables)

        # 更新目标网络。
        def _update_target(new_net, old_net):
            params1 = tf.trainable_variables(old_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.global_variables(new_net)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param2.assign(param1))
            return update_ops

        self.update_target_op = _update_target("target_qnet", "qnet")

        # ------------------------------------------
        # ------------- 需要记录的中间值 --------------
        # ------------------------------------------
        self.max_qval = tf.reduce_max(self.qvals)

    def get_action(self, obs):
        """Return actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n).
        """
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        qvals = self.sess.run(self.qvals, feed_dict={self.observation: newobs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = newobs.shape[0]
        actions = np.random.randint(self.dim_action, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon
        actions[idx] = best_action[idx]

        if obs.ndim == 1 or obs.ndim == 3:
            actions = actions[0]
        return actions

    def get_action_boltzman(self, obs):
        if obs.ndim == 1:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            newobs = obs

        # 0.01 是一个不错的参数。
        alpha = 0.001

        qvals = self.sess.run(self.qvals, feed_dict={self.observation: newobs})
        exp_m = scipy.special.logsumexp(qvals / alpha, axis=1)
        exp_m = np.exp(qvals / alpha - exp_m)

        global_step = self.sess.run(tf.train.get_global_step())

        actions = [np.random.choice(self.dim_action, p=exp_m[i]) for i in range(newobs.shape[0])]

        if obs.ndim == 1:
            actions = actions[0]
        return actions

    def update(self, minibatch, update_ratio):
        """Update the algorithm by suing a batch of data.

        Parameters:
            - minibatch: A list of ndarray containing a minibatch of state, action, reward, done, next_state.

                - state shape: (n_env, batch_size, state_dimension)
                - action shape: (n_env, batch_size)
                - reward shape: (n_env, batch_size)
                - done shape: (n_env, batch_size)
                - next_state shape: (n_env, batch_size, state_dimension)

            - update_ratio: float scalar in (0, 1).

        Returns:
            - training infomation.
        """
        self.epsilon = self.epsilon_schedule(update_ratio)

        # 拆分样本。
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        mb_s, mb_a, mb_target = [], [], []

        n_env = s_batch.shape[0]
        for i in range(n_env):
            batch_size = s_batch[i, :].shape[0]
            current_next_q_vals, target_next_q_vals = self.sess.run(
                [self.qvals, self.target_qvals], feed_dict={self.observation: next_s_batch[i, :]})
            q_next = target_next_q_vals[range(batch_size), current_next_q_vals.argmax(axis=1)]
            target_batch = r_batch[i, :] + (1 - d_batch[i, :]) * self.discount * q_next

            mb_target.append(target_batch)
            mb_s.append(s_batch[i, :])
            mb_a.append(a_batch[i, :])

        mb_s = np.concatenate(mb_s)
        mb_a = np.concatenate(mb_a)
        mb_target = np.concatenate(mb_target)

        _, loss, max_q_val = self.sess.run(
            [self.train_op,
             self.loss,
             self.max_qval],
            feed_dict={
                self.observation: mb_s,
                self.action: mb_a,
                self.target: mb_target
            }
        )

        global_step = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        # Save model.
        if global_step % self.save_model_freq == 0:
            self.save_model()

        # Update policy.
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return {"loss": loss, "max_q_value": max_q_val, "training_step": global_step}
