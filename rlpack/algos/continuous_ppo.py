from collections import deque
import math
import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class ContinuousPPO(Base):
    def __init__(self,
                 rnd=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 gae=0.95,
                 save_path="./log",
                 save_model_freq=50,
                 log_freq=100,
                 vf_coef=1.0,
                 entropy_coef=0.01,
                 max_grad_norm=40,
                 lr_schedule=lambda x: 2.5e-4,
                 clip_schedule=lambda x: max(0.01, 0.1 * (10000-x) / 10000),
                 batch_size=64,
                 train_epoch=10
                 ):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount
        self._gae = gae

        self._batch_size = batch_size
        self._save_model_freq = save_model_freq

        self._lr_schedule = lr_schedule
        self._clip_schedule = clip_schedule

        self._entropy_coef = entropy_coef
        self._vf_coef = vf_coef
        self._max_grad_norm = max_grad_norm
        self._train_epoch = train_epoch
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def build_network(self):
        """Build networks for algorithm."""
        self._observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")

        assert_shape(self._observation, [None, *self._dim_obs])

        with tf.variable_scope("policy_net"):
            x = tf.layers.dense(self._observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.mu = tf.layers.dense(x, self._dim_act, activation=tf.nn.tanh)
            self.log_var = tf.get_variable("logvars", [self.mu.shape.as_list()[1]], tf.float32, tf.constant_initializer(0.0))

        with tf.variable_scope("value_net"):
            x = tf.layers.dense(self._observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self._state_value = tf.squeeze(tf.layers.dense(x, 1, activation=None))

    def build_algorithm(self):
        """Build networks for algorithm."""
        self._clip_epsilon = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(self._lr)

        self.action = tf.placeholder(tf.float32, [None, self._dim_act], "action")
        self.span_reward = tf.placeholder(tf.float32, [None], "span_reward")
        self.advantage = tf.placeholder(tf.float32, [None], "advantage")
        self.old_mu = tf.placeholder(tf.float32, [None, self._dim_act], "old_mu")
        self.old_log_var = tf.placeholder(tf.float32, [self._dim_act], "old_var")

        logp = -0.5 * tf.reduce_sum(self.log_var * 2)
        logp += -0.5 * tf.reduce_sum(tf.square(self.action - self.mu) / tf.exp(self.log_var * 2), axis=1)  # 　- 0.5 * math.log(2 * math.pi)

        # var = tf.exp(self.log_var * 2)
        # logp = - (self.action - self.mu) ** 2 / (2 * var) - self.log_var  # - 0.5 * math.log(2 * math.pi)
        # logp = tf.reduce_sum(logp, axis=1)
        # print(f"logp shape: {logp.shape}")

        logp_old = -0.5 * tf.reduce_sum(self.old_log_var * 2)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_mu) / tf.exp(self.old_log_var * 2), axis=1)  # - 0.5 * math.log(2 * math.pi)

        # var_old = tf.exp(self.old_log_var * 2)
        # logp_old = - (self.action - self.old_mu) ** 2 / (2 * var_old) - self.old_log_var
        # logp_old = tf.reduce_sum(logp_old, axis=1)

        # Compute KL divergence.
        log_det_cov_old = tf.reduce_sum(self.old_log_var)
        log_det_cov_new = tf.reduce_sum(self.log_var)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_var - self.log_var))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new + tf.reduce_sum(
            tf.square(self.mu - self.old_mu) / tf.exp(self.log_var), axis=1) - self._dim_act)

        entropy = 0.5 * (self._dim_act + self._dim_act * tf.log(2 * np.pi) + tf.exp(tf.reduce_sum(self.log_var)))

        # Build surrogate loss.
        ratio = tf.exp(logp - logp_old)
        surr1 = ratio * self.advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - self._clip_epsilon, 1.0 + self._clip_epsilon) * self.advantage
        policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Build value loss.
        value_loss = tf.reduce_mean(tf.square(self._state_value - self.span_reward))

        # Build total_loss.
        total_loss = policy_loss + self._vf_coef * value_loss - self._entropy_coef * entropy

        grads = tf.gradients(total_loss, tf.trainable_variables())
        clipped_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        self._train_op = optimizer.apply_gradients(zip(clipped_grads, tf.trainable_variables()))

        self._log_op = {"policy_loss": policy_loss, "value_loss": value_loss, "total_loss": total_loss}

        # Build action sample.
        self.sample_action = self.mu + tf.exp(self.log_var) * tf.random_normal(shape=[self._dim_act], dtype=tf.float32)

    def get_action(self, obs):
        actions = self.sess.run(self.sample_action, feed_dict={self._observation: obs})
        return actions

    def update(self, databatch):
        s_batch, a_batch, tsv_batch, adv_batch = self._parse_databatch(databatch)

        # Normalize advantage.
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        # Compute old terms for placeholder.
        old_mu_batch, old_log_var = self.sess.run([self.mu, self.log_var], feed_dict={self._observation: s_batch})

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        for _ in range(self._train_epoch):

            n_sample = s_batch.shape[0]
            index = np.arange(n_sample)
            np.random.shuffle(index)

            for i in range(math.ceil(n_sample / self._batch_size)):
                span_index = slice(i*self._batch_size, min((i+1)*self._batch_size, n_sample))
                span_index = index[span_index]

                self.sess.run(self._train_op,
                              feed_dict={
                                  self._observation: s_batch[span_index, ...],
                                  self.action: a_batch[span_index, ...],
                                  self.span_reward: tsv_batch[span_index],
                                  self.advantage: adv_batch[span_index],
                                  self.old_mu: old_mu_batch[span_index, ...],
                                  self.old_log_var: old_log_var,
                                  self._lr: self._lr_schedule(global_step),
                                  self._clip_epsilon: self._clip_epsilon(global_step)
                              })

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

        if global_step % self._log_freq == 0:
            log = self.sess.run(self._log_op,
                                feed_dict={
                                    self._observation: s_batch[span_index, ...],
                                    self.action: a_batch[span_index, ...],
                                    self.span_reward: tsv_batch[span_index],
                                    self.advantage: adv_batch[span_index],
                                    self.old_mu: old_mu_batch[span_index, ...],
                                    self.old_log_var: old_log_var,
                                })
            self.sw.add_scalars("cont_ppo", log, global_step=global_step)

    def _parse_databatch(self, databatch):
        s_list = deque()
        a_list = deque()
        tsv_list = deque()
        adv_list = deque()
        for trajectory in databatch:
            s_batch, a_batch, tsv_batch, adv_batch = self._parse_trajectory(trajectory)
            s_list.append(s_batch)
            a_list.append(a_batch)
            tsv_list.append(tsv_batch)
            adv_list.append(adv_batch)
        return np.concatenate(s_list), np.concatenate(a_list), np.concatenate(tsv_list), np.concatenate(adv_list)

    def _parse_trajectory(self, trajectory):
        """trajectory由一系列(s,a,r)构成。最后一组操作之后游戏结束。
        """
        n = len(trajectory)
        target_sv_batch = np.zeros(n, dtype=np.float32)
        adv_batch = np.zeros(n, dtype=np.float32)

        a_batch = np.array([t[1] for t in trajectory], dtype=np.int32)
        s_batch = np.array([t[0] for t in trajectory], dtype=np.float32)
        sv_batch = self.sess.run(self._state_value, feed_dict={self._observation: s_batch})

        for i, (_, _, r) in enumerate(reversed(trajectory)):   # 注意这里是倒序操作。
            i = n-1-i
            state_value = sv_batch[i]
            if i == n-1:
                adv_batch[i] = r - state_value
                target_sv_batch[i] = r
                last_state_value = state_value
                continue
            delta_value = r + self._discount * last_state_value - state_value
            adv_batch[i] = delta_value + self._discount * self._gae * adv_batch[i+1]
            target_sv_batch[i] = state_value + adv_batch[i]
            last_state_value = state_value

        return s_batch, a_batch, target_sv_batch, adv_batch
