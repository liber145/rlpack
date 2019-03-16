import numpy as np
import tensorflow as tf

import scipy
import tensorflow as tf

from .base import Base


class DoubleDQN(Base):
    def __init__(self,
                 dim_obs=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 update_target_freq=10000,
                 epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                 lr=2.5e-4,
                 train_epoch=1,
                 log_freq=1000,
                 save_path="./log",
                 save_model_freq=1000,
                 ):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount
        self._update_target_freq = update_target_freq
        self._epsilon_schedule = epsilon_schedule
        self._lr = lr
        self._train_epoch = train_epoch

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.uint8, name="observation")
        # self._observation = tf.to_float(self._observation) / 256.0
        self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self._action = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        self._reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        self._done = tf.placeholder(dtype=tf.float32, shape=[None], name="done")
        # self._next_observation = tf.placeholder(dtype=tf.uint8, shape=[None, *self._dim_obs], name="next_observation")
        # self._next_observation = tf.to_float(self._next_observation) / 256.0
        self._next_observation = tf.placeholder(dtype=tf.float32, shape=[None, *self._dim_obs], name="next_observation")

        with tf.variable_scope("main/qnet"):
            self._qvals = self._dense(self._observation)

        with tf.variable_scope("main/qnet", reuse=True):
            self._act_qvals = tf.stop_gradient(self._dense(self._next_observation))

        with tf.variable_scope("target/qnet"):
            self._target_qvals = tf.stop_gradient(self._dense(self._next_observation))

    def _conv(self, x):
        x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
        return tf.layers.dense(x, self._dim_act)

    def _dense(self, x):
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 64, activation=tf.nn.relu)
        x = tf.layers.dense(x, self._dim_act)
        return x

    def _build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(self._lr)
        trainable_variables = tf.trainable_variables("main/qnet")

        # Compute state-action value.
        batch_size = tf.shape(self._observation)[0]
        gather_indices = tf.range(batch_size) * self._dim_act + self._action
        action_q = tf.gather(tf.reshape(self._qvals, [-1]), gather_indices)

        # Compute back up.
        arg_act = tf.argmax(self._act_qvals, axis=1, output_type=tf.int32)
        arg_act_index = tf.stack([tf.range(batch_size), arg_act], axis=1)
        q_backup = self._reward + self._discount * (1 - self._done) * tf.gather_nd(self._target_qvals, arg_act_index)

        loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))
        self._train_op = self.optimizer.minimize(loss, var_list=trainable_variables)

        # Update target network.

        def _update_target(new_net, old_net):
            params1 = tf.trainable_variables(old_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.trainable_variables(new_net)
            params2 = sorted(params2, key=lambda v: v.name)

            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param2.assign(param1))
            return update_ops

        self._update_target_op = _update_target("target/qnet", "main/qnet")

        self._log_op = {"loss": loss}

    def get_action(self, obs):
        qvals = self.sess.run(self._qvals, feed_dict={self._observation: obs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        actions = np.random.randint(self._dim_act, size=batch_size)
        global_step = self.sess.run(tf.train.get_global_step())
        idx = np.random.uniform(size=batch_size) > self._epsilon_schedule(global_step)
        actions[idx] = best_action[idx]
        return actions

    def get_action_boltzman(self, obs):
        if obs.ndim == 1:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            newobs = obs

        # 0.01 是一个不错的参数。
        alpha = 0.001

        qvals = self.sess.run(self._qvals, feed_dict={self._observation: newobs})
        exp_m = scipy.special.logsumexp(qvals / alpha, axis=1)
        exp_m = np.exp(qvals / alpha - exp_m)

        actions = [np.random.choice(self._dim_act, p=exp_m[i]) for i in range(newobs.shape[0])]

        if obs.ndim == 1:
            actions = actions[0]
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
            self.sw.add_scalars("aadqn", log, global_step=global_step)
