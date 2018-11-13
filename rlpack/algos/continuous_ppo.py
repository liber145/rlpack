import math

import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape, exponential_decay, linear_decay
from .base import Base


class ContinuousPPO(Base):
    def __init__(self, config):
        self.tau = config.gae

        self.lr = config.lr
        self.trajectory_length = config.trajectory_length

        self.entropy_coef = config.entropy_coef
        self.critic_coef = config.vf_coef

        self.max_grad_norm = config.max_grad_norm

        self.training_epoch = config.training_epoch

        self.n_trajectory = config.n_trajectory

        self.n_env = config.n_env

        self.lr_schedule = config.lr_schedule
        self.clip_schedule = config.clip_schedule

        super().__init__(config)

    def build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="observation")

        with tf.variable_scope("policy_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.mu = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh)
            self.log_var = tf.get_variable("logvars", [self.mu.shape.as_list()[1]], tf.float32, tf.constant_initializer(0.0)) - 1

        with tf.variable_scope("value_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.state_value = tf.squeeze(tf.layers.dense(x, 1, activation=None))

    def build_algorithm(self):
        self.clip_epsilon = tf.placeholder(tf.float32)
        self.actor_optimizer = tf.train.AdamOptimizer(self.lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr)

        self.action = tf.placeholder(tf.float32, [None, self.dim_action], "action")
        self.span_reward = tf.placeholder(tf.float32, [None], "span_reward")
        self.advantage = tf.placeholder(tf.float32, [None], "advantage")
        self.old_mu = tf.placeholder(tf.float32, [None, self.dim_action], "old_mu")
        self.old_log_var = tf.placeholder(tf.float32, [self.dim_action], "old_var")

        logp = -0.5 * tf.reduce_sum(self.log_var)
        logp += -0.5 * tf.reduce_sum(tf.square(self.action - self.mu) / tf.exp(self.log_var), axis=1, keepdims=True)

        logp_old = -0.5 * tf.reduce_sum(self.old_log_var)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_mu) /
                                         tf.exp(self.old_log_var), axis=1, keepdims=True)

        # Compute KL divergence.
        log_det_cov_old = tf.reduce_sum(self.old_log_var)
        log_det_cov_new = tf.reduce_sum(self.log_var)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_var - self.log_var))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new + tf.reduce_sum(
            tf.square(self.mu - self.old_mu) / tf.exp(self.log_var), axis=1) - self.dim_action)

        self.entropy = 0.5 * (self.dim_action + self.dim_action * tf.log(2 * np.pi) + tf.exp(tf.reduce_sum(self.log_var)))

        # Build surrogate loss.
        ratio = tf.exp(logp - logp_old)
        surr1 = ratio * self.advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * self.advantage
        self.surrogate = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Build value loss.
        self.critic_loss = tf.reduce_mean(tf.square(self.state_value - self.span_reward))

        # You can also build total loss and clip the gradients.
        # # Build total_loss.
        # self.total_loss = self.surrogate + self.critic_coef * self.critic_loss + self.entropy_coef * self.entropy   # TODO

        # # Build training operation.
        # grads = tf.gradients(self.total_loss, tf.trainable_variables())
        # clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # self.total_train_op = self.optimizer.apply_gradients(zip(clipped_grads, tf.trainable_variables()), global_step=tf.train.get_global_step())

        # Build actor operation.
        self.train_actor_op = self.actor_optimizer.minimize(self.surrogate)

        # Build critic operation.
        self.train_critic_op = self.critic_optimizer.minimize(self.critic_loss)

        # Build action sample.
        self.sample_action = self.mu + tf.exp(self.log_var / 2.0) * tf.random_normal(shape=[self.dim_action], dtype=tf.float32)

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        actions = self.sess.run(self.sample_action, feed_dict={self.observation: newobs})
        return actions

    def update(self, minibatch, update_ratio):
        """minibatch is a trajectory.

        Arguments:
            minibatch: n_env * trajectory_length * self.dim_observation
        """
        s_batch, a_batch, r_batch, d_batch = minibatch
        assert s_batch.shape == (self.n_env, self.trajectory_length + 1, *self.dim_observation)

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
                advantage_batch[i, t] = delta_value_batch[t] + self.discount * self.tau * (1 - d_batch[i, t]) * last_advantage
                last_advantage = advantage_batch[i, t]

            # Compute target value.
            target_value_batch[i, :] = state_value_batch[:-1] + advantage_batch[i, :]

        # Flat the batch values.
        s_batch = s_batch[:, :-1, ...].reshape(self.n_env * self.trajectory_length, *self.dim_observation)
        a_batch = a_batch.reshape(self.n_env * self.trajectory_length, self.dim_action)
        advantage_batch = advantage_batch.reshape(self.n_env * self.trajectory_length)
        target_value_batch = target_value_batch.reshape(self.n_env * self.trajectory_length)

        # Normalize advantage.
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-5)

        # Compute old terms for placeholder.
        old_mu_batch, old_log_var = self.sess.run([self.mu, self.log_var], feed_dict={self.observation: s_batch})

        for _ in range(self.training_epoch):
            batch_generator = self._generator([s_batch, a_batch, advantage_batch, old_mu_batch, target_value_batch], batch_size=self.batch_size)
            while True:
                try:
                    mb_s, mb_a, mb_advantage, mb_old_mu, mb_target_value = next(batch_generator)

                    self.sess.run(self.train_actor_op, feed_dict={
                        self.observation: mb_s,
                        self.action: mb_a,
                        self.span_reward: mb_target_value,
                        self.advantage: mb_advantage,
                        self.old_mu: mb_old_mu,
                        self.old_log_var: old_log_var,
                        self.clip_epsilon: 0.2})

                    self.sess.run(self.train_critic_op, feed_dict={
                        self.observation: mb_s,
                        self.span_reward: mb_target_value,
                        self.clip_epsilon: 0.2})

                except StopIteration:
                    del batch_generator
                    break
        if (update_ratio / self.save_model_freq) % 1 == 0:
            self.save_model()

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]
        assert n_sample == self.n_env * self.trajectory_length

        index = np.arange(n_sample)
        np.random.shuffle(index)

        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index] if x.ndim == 1 else x[span_index, :] for x in data_batch]
