import math

import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class PPO(Base):
    def __init__(self, config):
        """An implementation of PPO.

        Parameters:
            config: a dictionary for training config.

        Returns:
            None
        """
        self.entropy_coefficient = config.entropy_coef
        self.critic_coefficient = config.vf_coef
        self.trajectory_length = config.trajectory_length
        self.clip_schedule = config.clip_schedule

        super().__init__(config)

    def build_network(self):
        """Build networks for algorithm."""
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="observation")

        x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
        self.logit_action_probability = tf.layers.dense(x, self.dim_action, activation=None, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01))
        self.state_value = tf.squeeze(tf.layers.dense(x, 1, activation=None, kernel_initializer=tf.truncated_normal_initializer()))

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.init_clip_epsilon = 0.1
        self.init_lr = 2.5e-4
        self.clip_epsilon = tf.placeholder(tf.float32)
        self.moved_lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)

        self.old_logit_action_probability = tf.placeholder(tf.float32, [None, self.dim_action])
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.target_state_value = tf.placeholder(tf.float32, [None], "target_state_value")

        # Get selected action index.
        batch_size = tf.shape(self.observation)[0]
        selected_action_index = tf.stack([tf.range(batch_size), self.action], axis=1)

        # Compute entropy of the action probability.
        log_prob_1 = tf.nn.log_softmax(self.logit_action_probability)
        log_prob_2 = tf.stop_gradient(tf.nn.log_softmax(self.old_logit_action_probability))
        assert_shape(log_prob_1, [None, self.dim_action])
        assert_shape(log_prob_2, [None, self.dim_action])

        prob_1 = tf.nn.softmax(log_prob_1)
        prob_2 = tf.stop_gradient(tf.nn.softmax(log_prob_2))
        assert_shape(prob_1, [None, self.dim_action])
        # assert_shape(prob_2, [None, self.dim_action])

        self.entropy = - tf.reduce_sum(log_prob_1 * prob_1, axis=1)
        assert_shape(self.entropy, [None])

        # Compute ratio of the action probability.
        logit_act1 = tf.gather_nd(log_prob_1, selected_action_index)
        logit_act2 = tf.gather_nd(log_prob_2, selected_action_index)
        assert_shape(logit_act1, [None])
        assert_shape(logit_act2, [None])

        self.ratio = tf.exp(logit_act1 - logit_act2)

        # Get surrogate object.
        surrogate_1 = self.ratio * self.advantage
        surrogate_2 = tf.clip_by_value(self.ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * self.advantage
        assert_shape(self.ratio, [None])
        assert_shape(surrogate_1, [None])
        self.surrogate = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))

        # Compute critic loss.
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.state_value, self.target_state_value))

        # Compute gradients.
        self.total_loss = self.surrogate + self.critic_coefficient * self.critic_loss - self.entropy_coefficient * self.entropy
        grads = tf.gradients(self.total_loss, tf.trainable_variables())

        # Clip gradients.
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.total_train_op = self.optimizer.apply_gradients(
            zip(clipped_grads, tf.trainable_variables()), global_step=tf.train.get_global_step())

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs


        logit = self.sess.run(self.logit_action_probability, feed_dict={self.observation: newobs})
        logit = logit - np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit) / np.sum(np.exp(logit), axis=1, keepdims=True)
        action = [np.random.choice(self.dim_action, p=prob[i, :]) for i in range(newobs.shape[0])]
        assert len(action) == newobs.shape[0]
        return np.array(action)

    def update(self, minibatch, update_ratio):
        """Update the algorithm by suing a batch of data.

        Parameters:
            - minibatch: A list of ndarray containing a minibatch of state, action, reward, done, next_state.
                - state shape: (n_env, batch_size+1, state_dimension)
                - action shape: (n_env, batch_size)
                - reward shape: (n_env, batch_size)
                - done shape: (n_env, batch_size)
                - next_state shape: (n_env, batch_size, state_dimension)

            - update_ratio: float scalar in (0, 1).

        Returns:
            - training infomation.
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
                advantage_batch[i, t] = delta_value_batch[t] + self.discount * self.gae * (1 - d_batch[i, t]) * last_advantage
                last_advantage = advantage_batch[i, t]

            # Compute target value.
            target_value_batch[i, :] = state_value_batch[:-1] + advantage_batch[i, :]

        # Flat the batch values.
        s_batch = s_batch[:, :-1, :, :, :].reshape(self.n_env * self.trajectory_length, *self.dim_observation)
        a_batch = a_batch.reshape(self.n_env * self.trajectory_length)
        advantage_batch = advantage_batch.reshape(self.n_env * self.trajectory_length)
        target_value_batch = target_value_batch.reshape(self.n_env * self.trajectory_length)

        # Normalize Advantage.
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-5)

        old_logit_action_probability_batch = self.sess.run(self.logit_action_probability, feed_dict={self.observation: s_batch})

        # Train network.
        for _ in range(self.training_epoch):
            # Get training sample generator.
            batch_generator = self._generator([s_batch, a_batch, advantage_batch, old_logit_action_probability_batch, target_value_batch], batch_size=self.batch_size)

            while True:
                try:
                    mini_s_batch, mini_a_batch, mini_advantage_batch, mini_old_logit_action_probability_batch, mini_target_state_value_batch = next(
                        batch_generator)

                    global_step = self.sess.run(tf.train.get_global_step())

                    # Train actor.
                    c_loss, surr, entro, p_ratio, _ = self.sess.run([self.critic_loss,
                                                                     self.surrogate,
                                                                     self.entropy,
                                                                     self.ratio,
                                                                     self.total_train_op],
                                                                    feed_dict={
                        self.observation: mini_s_batch,
                        self.old_logit_action_probability: mini_old_logit_action_probability_batch,
                        self.action: mini_a_batch,
                        self.advantage: mini_advantage_batch,
                        self.target_state_value: mini_target_state_value_batch,
                        self.moved_lr: self.policy_lr_schedule(update_ratio),
                        self.clip_epsilon: self.clip_schedule(update_ratio)})

                except StopIteration:
                    del batch_generator
                    break

        if (update_ratio / self.save_model_freq) % 1 == 0:
            self.save_model()

        return {"critic_loss": c_loss, "surrogate": surr, "entropy": entro, "training_step": global_step, "sample_ratio": p_ratio[0]}

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]
        assert n_sample == self.n_env * self.trajectory_length

        index = np.arange(n_sample)
        np.random.shuffle(index)

        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index] if x.ndim == 1 else x[span_index, :] for x in data_batch]
