import numpy as np
import tensorflow as tf

import scipy
import tensorflow as tf

from .base import Base


class DoubleDQN(Base):
    def __init__(self,
                 rnd=1,
                 n_env=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 update_target_freq=10000,
                 epsilon_schedule=lambda x: (1-x),
                 lr=2.5e-4
                 ):

        self.n_env = n_env
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.discount = discount
        self.save_model_freq = save_model_freq
        self.update_target_freq = update_target_freq
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule(0)
        self.lr = lr

        super().__init__(save_path=save_path, rnd=rnd)

    def build_network(self):
        """Build networks for algorithm."""
        self.observation = tf.placeholder(shape=[None, *self.dim_obs], dtype=tf.uint8, name="observation")
        self.observation = tf.to_float(self.observation) / 256.0
        self.action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        self.reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        self.done = tf.placeholder(dtype=tf.float32, shape=[None], name="done")
        self.next_observation = tf.placeholder(dtype=tf.uint8, shape=[None, *self.dim_obs], name="next_observation")
        self.next_observation = tf.to_float(self.next_observation) / 256.0

        with tf.variable_scope("main/qnet"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self.qvals = tf.layers.dense(x, self.dim_act)

        with tf.variable_scope("main/qnet", reuse=True):
            x = tf.layers.conv2d(self.next_observation, 32, 8, 4, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu, trainable=False)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=False)
            self.act_qvals = tf.layers.dense(x, self.dim_act, trainable=False)

        with tf.variable_scope("target/qnet"):
            x = tf.layers.conv2d(self.next_observation, 32, 8, 4, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu, trainable=False)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=False)
            self.target_qvals = tf.layers.dense(x, self.dim_act, trainable=False)

    def build_algorithm(self):
        # self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        # self.target = tf.placeholder(shape=[None], dtype=tf.float32, name="target_qvalue")
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        trainable_variables = tf.trainable_variables("main/qnet")

        # Compute state-action value.
        batch_size = tf.shape(self.observation)[0]
        gather_indices = tf.range(batch_size) * self.dim_act + self.action
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)

        # Compute back up.
        arg_act = tf.argmax(self.act_qvals, axis=1, output_type=tf.int32)
        arg_act_index = tf.stack([tf.range(batch_size), arg_act], axis=1)
        q_backup = self.reward + self.discount * (1 - self.done) * tf.gather_nd(self.target_qvals, arg_act_index)

        self.loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))
        self.train_op = self.optimizer.minimize(self.loss, var_list=trainable_variables)

        # Update target network.
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

        self.update_target_op = _update_target("target/qnet", "main/qnet")

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
        actions = np.random.randint(self.dim_act, size=batch_size)
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

        actions = [np.random.choice(self.dim_act, p=exp_m[i]) for i in range(newobs.shape[0])]

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

        n_env, batch_size = s_batch.shape[:2]
        s_batch = s_batch.reshape(n_env*batch_size, *self.dim_obs)
        a_batch = a_batch.reshape(n_env*batch_size)
        r_batch = r_batch.reshape(n_env * batch_size)
        d_batch = d_batch.reshape(n_env * batch_size)
        next_s_batch = next_s_batch.reshape(n_env * batch_size, *self.dim_obs)

        # mb_s, mb_a, mb_target = [], [], []
        # n_env = s_batch.shape[0]
        # for i in range(n_env):
        #     batch_size = s_batch[i, :].shape[0]
        #     current_next_q_vals, target_next_q_vals = self.sess.run(
        #         [self.qvals, self.target_qvals], feed_dict={self.observation: next_s_batch[i, :]})
        #     q_next = target_next_q_vals[range(batch_size), current_next_q_vals.argmax(axis=1)]
        #     target_batch = r_batch[i, :] + (1 - d_batch[i, :]) * self.discount * q_next

        #     mb_target.append(target_batch)
        #     mb_s.append(s_batch[i, :])
        #     mb_a.append(a_batch[i, :])

        # mb_s = np.concatenate(mb_s)
        # mb_a = np.concatenate(mb_a)
        # mb_target = np.concatenate(mb_target)

        _, loss, max_q_val = self.sess.run(
            [self.train_op,
             self.loss,
             self.max_qval],
            feed_dict={
                self.observation: s_batch,
                self.action: a_batch,
                self.reward: r_batch,
                self.done: d_batch,
                self.next_observation: next_s_batch
            }
        )

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        # Save model.
        if global_step % self.save_model_freq == 0:
            self.save_model()

        # Update policy.
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return {"loss": loss, "max_q_value": max_q_val, "training_step": global_step}
