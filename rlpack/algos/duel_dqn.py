import numpy as np
import tensorflow as tf

from .base import Base


class DuelDQN(Base):
    """Dueling Archtecture, Double DQN"""

    def __init__(self,
                 dim_obs=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 update_target_freq=10000,
                 log_freq=1000,
                 epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                 lr=2.5e-4,
                 train_epoch=1,
                 ):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount
        self._epsilon_schedule = epsilon_schedule
        self._lr = lr
        self._train_epoch = train_epoch
        self._update_target_freq = update_target_freq

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.uint8, name="observation")
        # self._observation = tf.to_float(self._observation) / 256.0
        self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self._action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self._reward = tf.placeholder(shape=[None], dtype=tf.float32, name="reward")
        self._done = tf.placeholder(shape=[None], dtype=tf.float32, name="done")
        self._next_observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="next_observation")

        with tf.variable_scope("main"):
            x = self._dense(self._observation)
            self._v = tf.layers.dense(x, 1)
            self._adv = tf.layers.dense(x, self._dim_act)

        with tf.variable_scope("main", reuse=True):
            x = self._dense(self._next_observation)
            self._act_v = tf.stop_gradient(tf.layers.dense(x, 1))
            self._act_adv = tf.stop_gradient(tf.layers.dense(x, self._dim_act))

        with tf.variable_scope("target"):
            x = self._dense(self._next_observation)
            self._target_v = tf.stop_gradient(tf.layers.dense(x, 1))
            self._target_adv = tf.stop_gradient(tf.layers.dense(x, self._dim_act))

    def _conv(self, x):
        x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
        return x

    def _dense(self, x):
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 64, activation=tf.nn.relu)
        return x

    def _build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1.5e-8)
        trainable_variables = tf.trainable_variables("main")

        batch_size = tf.shape(self._observation)[0]
        self._qvals = self._v + (self._adv - tf.reduce_mean(self._adv, axis=1, keepdims=True))
        self._act_qvals = self._act_v + (self._act_adv - tf.reduce_mean(self._act_adv, axis=1, keepdims=True))
        self._target_qvals = self._target_v + (self._target_adv - tf.reduce_mean(self._target_adv, axis=1, keepdims=True))

        max_act = tf.argmax(self._act_qvals, axis=1, output_type=tf.int32)
        act_index = tf.stack([tf.range(batch_size), max_act], axis=1)
        q_backup = self._reward + (1 - self._done) * self._discount * tf.gather_nd(self._target_qvals, act_index)

        act_index = tf.stack([tf.range(batch_size), self._action], axis=1)
        action_q = tf.gather_nd(self._qvals, act_index)

        loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))

        self._train_op = self.optimizer.minimize(loss, var_list=trainable_variables)

        # 更新目标网络。
        def _update_target(net1, net2):

            params1 = tf.trainable_variables(net1)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.trainable_variables(net2)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param1.assign(param2))
            return update_ops

        self._update_target_op = _update_target("target", "main")

        self._log_op = {"loss": loss}

    def get_action(self, obs):
        """Get actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n).
        """
        qvals = self.sess.run(self._qvals, feed_dict={self._observation: obs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        global_step = self.sess.run(tf.train.get_global_step())
        actions = np.random.randint(self._dim_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self._epsilon_schedule(global_step)
        actions[idx] = best_action[idx]
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
            log = self.sess.run(self._log_op, feed_dict={
                self._observation: s_batch,
                self._action: a_batch,
                self._reward: r_batch,
                self._done: d_batch,
                self._next_observation: next_s_batch
            })
            self.sw.add_scalars("dueldqn", log, global_step=global_step)
