import math

import numpy as np
import tensorflow as tf

from .base import Base


class A2C(Base):
    """Advantage Actor Critic."""

    def __init__(self,
                 rnd=1,
                 n_env=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=1000,
                 trajectory_length=2048,
                 gae=0.95,
                 training_epoch=10,
                 lr=3e-4,
                 batch_size=64
                 ):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self.discount = discount
        self.gae = gae
        self.lr = lr

        self.training_epoch = training_epoch
        self.log_freq = log_freq
        self.save_model_freq = save_model_freq
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size

        super().__init__(save_path=save_path, rnd=rnd)

    def build_network(self):
        """Build networks for algorithm."""
        # Build inputs.
        self.observation = tf.placeholder(tf.float32, [None, *self._dim_obs], "observation")

        # Build Nets.
        with tf.variable_scope("policy_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.mu = tf.layers.dense(x, self._dim_act, activation=tf.nn.tanh)
            self.log_var = tf.get_variable("logvars", [self.mu.shape.as_list()[1]], tf.float32, tf.constant_initializer(0.0)) - 1

        with tf.variable_scope("value_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.state_value = tf.squeeze(tf.layers.dense(x, 1, activation=None))

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.actor_optimizer = tf.train.AdamOptimizer(self.lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr)

        self.action = tf.placeholder(tf.float32, [None, self._dim_act], "action")
        self.span_reward = tf.placeholder(tf.float32, [None], "span_reward")
        self.advantage = tf.placeholder(tf.float32, [None], "advantage")

        self.old_mu = tf.placeholder(tf.float32, (None, self._dim_act), "old_mu")
        self.old_log_var = tf.placeholder(tf.float32, [self._dim_act], "old_log_var")

        logp = -0.5 * tf.reduce_sum(self.log_var)
        logp += -0.5 * tf.reduce_sum(tf.square(self.action - self.mu) / tf.exp(self.log_var), axis=1, keepdims=True)

        logp_old = -0.5 * tf.reduce_sum(self.old_log_var)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_mu) / tf.exp(self.old_log_var), axis=1, keepdims=True)

        self.actor_loss = -tf.reduce_mean(self.advantage * tf.exp(logp - logp_old))

        # Update by adam.
        self.train_actor_op = self.actor_optimizer.minimize(self.actor_loss)

        # ---------- Build critic algorithm. ----------
        self.critic_loss = tf.reduce_mean(tf.square(self.state_value - self.span_reward))

        # Update by adam.
        self.train_critic_op = self.critic_optimizer.minimize(self.critic_loss, global_step=tf.train.get_global_step())

        # ---------- Build action. ----------
        self.sampled_act = (self.mu + tf.exp(self.log_var / 2.0) * tf.random_normal(shape=[self._dim_act], dtype=tf.float32))

    def update(self, minibatch, update_ratio):
        """Update the algorithm by suing a batch of data.

        Parameters:
            - minibatch: A list of ndarray containing a minibatch of state, action, reward, done.

                - state shape: (n_env, batch_size+1, state_dimension)
                - action shape: (n_env, batch_size, state_dimension)
                - reward shape: (n_env, batch_size)
                - done shape: (n_env, batch_size)

            - update_ratio: float scalar in (0, 1).

        Returns:
            - training infomation.
        """
        s_batch, a_batch, r_batch, d_batch = minibatch
        assert s_batch.shape == (self.n_env, self.trajectory_length + 1, *self._dim_obs)

        # Compute advantage batch.
        advantage_batch = np.empty([self.n_env, self.trajectory_length], dtype=np.float32)
        target_value_batch = np.empty([self.n_env, self.trajectory_length], dtype=np.float32)

        for i in range(self.n_env):
            state_value_batch = self.sess.run(self.state_value, feed_dict={self.observation: s_batch[i, ...]})

            delta_value_batch = r_batch[i, :] + self.discount * (1 - d_batch[i, :]) * state_value_batch[1:] - state_value_batch[:-1]
            assert state_value_batch.shape == (self.trajectory_length + 1,)
            assert delta_value_batch.shape == (self.trajectory_length,)

            last_advantage = 0
            for t in reversed(range(self.trajectory_length)):
                advantage_batch[i, t] = delta_value_batch[t] + self.discount * self.gae * (1 - d_batch[i, t]) * last_advantage
                last_advantage = advantage_batch[i, t]

            # Compute target value.
            target_value_batch[i, :] = state_value_batch[:-1] + advantage_batch[i, :]

        # Flat the batch values.
        s_batch = s_batch[:, :-1, ...].reshape(self.n_env * self.trajectory_length, *self._dim_obs)
        a_batch = a_batch.reshape(self.n_env * self.trajectory_length, self._dim_act)
        advantage_batch = advantage_batch.reshape(self.n_env * self.trajectory_length)
        target_value_batch = target_value_batch.reshape(self.n_env * self.trajectory_length)

        # Normalize advantage.
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-5)

        # Compute old terms for placeholder.
        old_mu_batch, old_log_var = self.sess.run([self.mu, self.log_var], feed_dict={self.observation: s_batch})

        # ---------- Update actor ----------
        for _ in range(self.training_epoch):
            batch_generator = self._generator([s_batch, a_batch, advantage_batch, old_mu_batch, target_value_batch], batch_size=self.batch_size)

            while True:
                try:
                    mb_s, mb_a, mb_advantage, mb_old_mu, mb_target_value = next(batch_generator)

                    self.sess.run(self.train_actor_op, feed_dict={
                        self.observation: mb_s,
                        self.action: mb_a,
                        self.advantage: mb_advantage,
                        self.old_mu: mb_old_mu,
                        self.old_log_var: old_log_var})

                except StopIteration:
                    del batch_generator
                    break

        # ---------- Update critic ----------
        for _ in range(self.training_epoch):
            batch_generator = self._generator([s_batch, target_value_batch], batch_size=self.batch_size)

            while True:
                try:
                    mb_s, mb_target_value = next(batch_generator)

                    _, global_step, critic_loss = self.sess.run([self.train_critic_op, tf.train.get_global_step(), self.critic_loss],
                                                                feed_dict={
                                                                self.observation: mb_s,
                                                                self.span_reward: mb_target_value})
                except StopIteration:
                    del batch_generator
                    break

        return {"critic_loss": critic_loss, "global_step": global_step}

    def get_action(self, ob):
        """Return actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n, action_dimension).
        """

        action = self.sess.run(self.sampled_act, feed_dict={self.observation: ob})
        return action

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]
        assert n_sample == self.n_env * self.trajectory_length

        index = np.arange(n_sample)
        np.random.shuffle(index)

        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index] if x.ndim == 1 else x[span_index, :] for x in data_batch]
