import math

import numpy as np
import tensorflow as tf

from .base import Base


class A2C(Base):
    """Advantage Actor Critic."""

    def __init__(self, config):
        self.tau = config.gae
        self.n_env = config.n_env
        self.training_epoch = config.training_epoch

        super().__init__(config)

    def build_network(self):
        # Build inputs.
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], "observation")

        # # Build Nets.
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
        # ------------ Compute g of object. -------------
        self.actor_optimizer = tf.train.AdamOptimizer(self.lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr)

        self.action = tf.placeholder(tf.float32, [None, self.dim_action], "action")
        self.span_reward = tf.placeholder(tf.float32, [None], "span_reward")
        self.advantage = tf.placeholder(tf.float32, [None], "advantage")

        self.old_mu = tf.placeholder(tf.float32, (None, self.dim_action), "old_mu")
        self.old_log_var = tf.placeholder(tf.float32, [self.dim_action], "old_log_var")

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
        self.sampled_act = (self.mu + tf.exp(self.log_var / 2.0) * tf.random_normal(shape=[self.dim_action], dtype=tf.float32))

    def update(self, minibatch, update_ratio):

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

        # data_batch = utils.trajectories_to_batch(trajectories, self.discount)
        #
        # old_mu_val, old_log_var_val = self.sess.run(
        #     [self.mu, self.log_var], feed_dict={self.observation: data_batch["state"]})
        #
        # # log_var is a common parameter independent of states.
        # # So there is no shuffle for it.
        # data_batch["oldmu"] = old_mu_val

        # ---------- Update actor ----------

        # Update actor.
        for _ in range(self.training_epoch):
            batch_generator = self._generator([s_batch, a_batch, advantage_batch, old_mu_batch, target_value_batch], batch_size=self.batch_size)

            # batch_generator = utils.generator(data_batch, self.batch_size)

            while True:
                try:
                    mb_s, mb_a, mb_advantage, mb_old_mu, mb_target_value = next(batch_generator)

                    # sample_batch = next(batch_generator)

                    # Compute advantage.
                    # nextstate_val = self.sess.run(
                    #     self.state_value, feed_dict={self.observation: sample_batch["nextstate"]})
                    # state_val = self.sess.run(
                    #     self.state_value, feed_dict={self.observation: sample_batch["state"]})
                    #
                    # advantage = (sample_batch["reward"] + self.discount *
                    #              (1 - sample_batch["done"]) * nextstate_val) - state_val
                    #
                    # self.feeddict = {self.observation: sample_batch["state"],
                    #                  self.action: sample_batch["action"],
                    #                  self.span_reward: sample_batch["spanreward"],
                    #                  self.old_mu: sample_batch["oldmu"],
                    #                  self.old_log_var: old_log_var_val,
                    #                  self.advantage: advantage
                    #                  }

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
        # critic_loss = self.sess.run(self.critic_loss, feed_dict=self.feeddict)
        # print("old critic loss:", critic_loss)

        for _ in range(10):
            batch_generator = self._generator([s_batch, target_value_batch], batch_size=self.batch_size)

            # batch_generator = utils.generator(data_batch)

            while True:
                try:
                    mb_s, mb_target_value = next(batch_generator)

                    # sample_batch = next(batch_generator)

                    # self.feeddict[self.observation] = sample_batch["state"]
                    # self.feeddict[self.span_reward] = sample_batch["spanreward"]

                    _, global_step, critic_loss = self.sess.run([self.train_critic_op, tf.train.get_global_step(), self.critic_loss],
                                                                feed_dict={
                                                                self.observation: mb_s,
                                                                self.span_reward: mb_target_value})
                except StopIteration:
                    del batch_generator
                    break

        return {"loss": critic_loss, "global_step": global_step}

    def get_action(self, ob, epsilon=None):
        action = self.sess.run(self.sampled_act, feed_dict={
            self.observation: ob})
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
