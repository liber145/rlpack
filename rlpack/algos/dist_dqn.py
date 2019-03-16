import numpy as np
import tensorflow as tf

from tensorboardX import SummaryWriter

from .base import Base


class DistDQN(Base):
    def __init__(self,
                 dim_obs=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 update_target_freq=10000,
                 epsilon_schedule=lambda x: max(0.1, (1e4-x) / 1e4),
                 lr=2.5e-4,
                 n_histogram=51,
                 vmax=10,
                 vmin=-10,
                 log_freq=10,
                 train_epoch=1,
                 ):

        self._n_histogram = n_histogram
        self._vmax = vmax
        self._vmin = vmin
        self._delta = (self._vmax - self._vmin) / (self._n_histogram - 1)
        self._split_points = np.linspace(self._vmin, self._vmax, self._n_histogram)

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount

        self._lr = lr
        self._epsilon_schedule = epsilon_schedule
        self._update_target_freq = update_target_freq
        self._train_epoch = train_epoch

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.uint8, name="observation")
        # self._observation = tf.to_float(self._observation) / 256.0
        self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.target = tf.placeholder(tf.float32, [None], name="target")
        self._new_p_act = tf.placeholder(tf.float32, [None, self._n_histogram], name="next_input")
        self._next_observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="next_observation")

        with tf.variable_scope("main"):
            self._logits = self._dense(self._observation)

        with tf.variable_scope("target"):
            self._target_logits = tf.stop_gradient(self._dense(self._next_observation))

    def _conv(self, t):
        x = tf.layers.conv2d(t, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
        return tf.layers.dense(x, self._dim_act * self._n_histogram)

    def _dense(self, t):
        x = tf.layers.dense(t, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 64, activation=tf.nn.relu)
        return tf.layers.dense(x, self._dim_act * self._n_histogram)

    def _build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdamOptimizer(self._lr)
        trainable_variables = tf.trainable_variables('main')

        batch_size = tf.shape(self._observation)[0]
        self._p_act = tf.nn.softmax(tf.reshape(self._logits, [-1, self._dim_act, self._n_histogram]))
        self._target_p_act = tf.nn.softmax(tf.reshape(self._target_logits, [-1, self._dim_act, self._n_histogram]))

        gather_indices = tf.range(batch_size) * self._dim_act + self.action
        self.action_probs = tf.gather(tf.reshape(self._p_act, [-1, self._n_histogram]), gather_indices)
        self.action_probs_clip = tf.clip_by_value(self.action_probs, 0.00001, 0.99999)

        loss = tf.reduce_mean(-tf.reduce_sum(self._new_p_act * tf.log(self.action_probs_clip), axis=-1))
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
        """Return actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n).
        """
        probs = self.sess.run(self._p_act, feed_dict={self._observation: obs})
        qvals = np.sum(probs * self._split_points, axis=-1)
        best_action = np.argmax(qvals, axis=1)

        batch_size = obs.shape[0]
        global_step = self.sess.run(tf.train.get_global_step())
        actions = np.random.randint(self._dim_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self._epsilon_schedule(global_step)
        actions[idx] = best_action[idx]
        return actions

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch
        next_q_probs = self.sess.run(self._target_p_act, feed_dict={self._next_observation: next_s_batch})
        next_q_vals = np.sum(next_q_probs * self._split_points, axis=-1)
        best_action = np.argmax(next_q_vals, axis=1)

        def compute_histogram(reward, probability, done):
            m = np.zeros(self._n_histogram, dtype=np.float32)
            projection = (np.clip(reward + self._discount * (1 - done) * self._split_points,
                                  self._vmin, self._vmax) - self._vmin) / self._delta
            for p, b in zip(probability, projection):
                l = np.floor(b).astype(np.int32)
                u = np.ceil(b).astype(np.int32)
                m[l] += p * (u - b)
                m[u] += p * (b - l)
            return m

        new_p_act = []
        for rew, prob, d in zip(r_batch, next_q_probs[np.arange(best_action.shape[0]), best_action], d_batch):
            new_p_act.append(compute_histogram(rew, prob, d))
        new_p_act = np.array(new_p_act)

        for _ in range(self._train_epoch):
            self.sess.run(self._train_op,
                          feed_dict={
                              self._observation: s_batch,
                              self.action: a_batch,
                              self._new_p_act: new_p_act
                          })

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

        # 更新目标策略。
        if global_step % self._update_target_freq == 0:
            self.sess.run(self._update_target_op)

        if global_step % self._log_freq == 0:
            log = self.sess.run(self._log_op,
                                feed_dict={
                                    self._observation: s_batch,
                                    self.action: a_batch,
                                    self._new_p_act: new_p_act
                                })
            self.sw.add_scalars("distdqn", log, global_step=global_step)
