import math

import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class PPO(Base):
    def __init__(self, config):
        self.tau = config.gae
        self.entropy_coefficient = config.entropy_coef
        self.critic_coefficient = config.vf_coef
        self.max_grad_norm = config.max_grad_norm

        self.n_trajectory = config.n_trajectory
        self.trajectory_length = config.trajectory_length
        self.n_env = config.n_env
        self.training_epoch = config.training_epoch

        self.dim_observation = config.dim_observation
        self.n_action = config.n_action
        self.discount = config.discount
        self.batch_size = config.batch_size

        self.lr_schedule = config.lr_schedule
        self.clip_schedule = config.clip_schedule

        self.save_path = config.save_path
        self.save_model_freq = config.save_model_freq

        # ------------------------ 申请网络图 ------------------------
        tf.reset_default_graph()
        tf.Variable(0, name="global_step", trainable=False)

        # ------------------------ 搭建网络 ------------------------
        self.build_network()

        # ------------------------ 搭建算法 ------------------------
        self.build_algorithm()

        # ------------------------ 存储模型，存储训练信息，重载上回模型 ------------------------
        self._prepare()

    # def build_network(self):
    #     self.observation = tf.placeholder(tf.float32, [None, self.dim_observation], "observation")
    #     with tf.variable_scope("policy_net"):
    #         x = tf.layers.dense(self.observation, 64, activation=tf.nn.relu)
    #         x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #         self.logit_action_probability = tf.layers.dense(x, self.n_action, activation=None)

    #     with tf.variable_scope("value_net"):
    #         x = tf.layers.dense(self.observation, 64, activation=tf.nn.relu)
    #         x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #         self.state_value = tf.squeeze(tf.layers.dense(x, 1))

    def build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="observation")
        x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
        x = tf.layers.dense(x, 512, activation=tf.nn.relu)
        self.logit_action_probability = tf.layers.dense(
            x, self.n_action, activation=None, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01))
        self.state_value = tf.squeeze(tf.layers.dense(
            x, 1, activation=None, kernel_initializer=tf.truncated_normal_initializer()))

    # def build_network(self):
    #     self.observation = tf.placeholder(tf.float32, [None, 224, 320, 4], name="observation")
    #     with tf.variable_scope("policy_net"):
    #         x = tf.layers.conv2d(self.observation, 32, 5, 2, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x, training=True)
    #         x = tf.layers.conv2d(x, 32, 5, 2, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x, training=True)
    #         x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
    #         x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #         self.logit_action_probability = tf.layers.dense(x, self.n_action, activation=None)

    #     with tf.variable_scope("value_net"):
    #         x = tf.layers.conv2d(self.observation, 32, 5, 2, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x, training=True)
    #         x = tf.layers.conv2d(x, 32, 5, 2, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x, training=True)
    #         x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
    #         x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #         self.state_value = tf.squeeze(tf.layers.dense(x, 1))

    def build_algorithm(self):
        self.init_clip_epsilon = 0.1
        self.init_lr = 2.5e-4
        self.clip_epsilon = tf.placeholder(tf.float32)
        self.moved_lr = tf.placeholder(tf.float32)
        self.optimizer = tf.train.AdamOptimizer(self.moved_lr, epsilon=1e-5)

        self.old_logit_action_probability = tf.placeholder(tf.float32, [None, self.n_action])
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.target_state_value = tf.placeholder(tf.float32, [None], "target_state_value")

        # Get selected action index.
        batch_size = tf.shape(self.observation)[0]
        selected_action_index = tf.stack([tf.range(batch_size), self.action], axis=1)

        # Compute entropy of the action probability.
        log_prob_1 = tf.nn.log_softmax(self.logit_action_probability)
        log_prob_2 = tf.stop_gradient(tf.nn.log_softmax(self.old_logit_action_probability))
        assert_shape(log_prob_1, [None, self.n_action])
        assert_shape(log_prob_2, [None, self.n_action])

        prob_1 = tf.nn.softmax(log_prob_1)
        prob_2 = tf.stop_gradient(tf.nn.softmax(log_prob_2))
        assert_shape(prob_1, [None, self.n_action])
        assert_shape(prob_2, [None, self.n_action])

        self.entropy = - tf.reduce_sum(log_prob_1 * prob_1, axis=1)   # entropy = - \sum_i p_i \log(p_i)
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
        assert_shape(surrogate_2, [None])
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
        if isinstance(obs, list):
            n_inference = len(obs)
        elif isinstance(obs, np.ndarray):
            n_inference = obs.shape[0]
        else:
            raise NotImplementedError("not supported observation type.")
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        logit = self.sess.run(self.logit_action_probability, feed_dict={self.observation: newobs})
        logit = logit - np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit) / np.sum(np.exp(logit), axis=1, keepdims=True)
        action = [np.random.choice(self.n_action, p=prob[i, :]) for i in range(n_inference)]
        return np.array(action)

    def update(self, minibatch, update_ratio):

        s_batch, a_batch, r_batch, d_batch = minibatch
        # assert s_batch.shape == (self.n_env, self.trajectory_length+1, *self.dim_observation)

        # Compute advantage batch.
        # advantage_batch = np.empty([self.n_env, self.trajectory_length], dtype=np.float32)
        # target_value_batch = np.empty([self.n_env, self.trajectory_length], dtype=np.float32)

        advantage_batch, target_value_batch = [], []
        for i in range(len(d_batch)):
            assert d_batch[i].ndim == 1
            traj_size = len(d_batch[i])
            adv = np.empty(traj_size, dtype=np.float32)

            state_value = self.sess.run(self.state_value, feed_dict={self.observation: s_batch[i]})

            delta_value = r_batch[i] + self.discount * (1 - d_batch[i]) * state_value[1:] - state_value[:-1]

            last_advantage = 0

            for t in reversed(range(traj_size)):
                adv[t] = delta_value[t] + self.discount * self.tau * (1 - d_batch[i][t]) * last_advantage
                last_advantage = adv[t]

            # Compute target value.
            target_value_batch.append(state_value[:-1] + adv)
            # Collect advantage.
            advantage_batch.append(adv)

        # Flat the batch values.
        advantage_batch = np.concatenate(advantage_batch, axis=0)
        target_value_batch = np.concatenate(target_value_batch, axis=0)
        all_step = sum(len(dones) for dones in d_batch)

        s_batch = np.concatenate([s[:-1, ...] for s in s_batch], axis=0)
        a_batch = np.concatenate(a_batch, axis=0)
        advantage_batch = advantage_batch.reshape(all_step)
        target_value_batch = target_value_batch.reshape(all_step)

        # s_batch = s_batch[:, :-1, :, :, :].reshape(self.n_env * self.trajectory_length, *self.dim_observation)
        # a_batch = a_batch.reshape(self.n_env * self.trajectory_length)
        # advantage_batch = advantage_batch.reshape(self.n_env * self.trajectory_length)
        # target_value_batch = target_value_batch.reshape(self.n_env * self.trajectory_length)

        # Normalize Advantage.
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-5)

        # batch_size = s_batch.shape[0]
        # next_state_value_batch = self.sess.run(self.state_value, feed_dict={self.observation: next_s_batch})
        # state_value_batch = self.sess.run(self.state_value, feed_dict={self.observation: s_batch})

        # assert next_state_value_batch.ndim == 1
        # assert state_value_batch.ndim == 1

        # # Compute generalized advantage.
        # delta_batch = r_batch + self.discount * (1 - d_batch) * next_state_value_batch - state_value_batch
        # assert delta_batch.shape == (batch_size,)
        # advantage_batch = np.empty(batch_size, dtype=np.float32)
        # last_advantage = 0
        # for t in reversed(range(batch_size)):
        #     advantage_batch[t] = delta_batch[t] + self.discount * self.tau * (1 - d_batch[t]) * last_advantage
        #     last_advantage = advantage_batch[t].copy()

        # # Compute target state value.
        # target_state_value_batch = advantage_batch + state_value_batch

        # # Normalization advantage. This must be done after target state value.
        # advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-5)

        old_logit_action_probability_batch = self.sess.run(
            self.logit_action_probability, feed_dict={self.observation: s_batch})

        # Train network.
        for _ in range(self.training_epoch):
            # Get training sample generator.
            batch_generator = self._generator(
                [s_batch, a_batch, advantage_batch, old_logit_action_probability_batch, target_value_batch], batch_size=self.batch_size)
            # # Train actor.
            # while True:
            #     try:
            #         mini_s_batch, mini_a_batch, mini_advantage_batch, mini_old_logit_action_probability_batch, mini_target_state_value_batch = next(
            #             batch_generator)

            #         # print(f"mini target state value shape: {mini_target_state_value_batch.shape}")

            #         global_step = self.sess.run(tf.train.get_global_step())

            #         # Train actor.
            #         p_ratio, _ = self.sess.run([self.ratio, self.actor_train_op], feed_dict={
            #             self.observation: mini_s_batch,
            #             self.old_logit_action_probability: mini_old_logit_action_probability_batch,
            #             self.action: mini_a_batch,
            #             self.advantage: mini_advantage_batch,
            #             self.actor_lr: exponential_decay(self.actor_init_lr, 0.9999, 100000, global_step)})

            #         # print(f"p_ratio: {p_ratio}")

            #         # Train Critic.
            #         self.sess.run(self.critic_train_op, feed_dict={
            #             self.observation: mini_s_batch,
            #             self.target_state_value: mini_target_state_value_batch,
            #             self.critic_lr: exponential_decay(self.critic_init_lr, 0.9999, 100000, global_step)})
            #     except StopIteration:
            #         del batch_generator
            #         break

            while True:
                try:
                    mini_s_batch, mini_a_batch, mini_advantage_batch, mini_old_logit_action_probability_batch, mini_target_state_value_batch = next(batch_generator)

                    # print(f"mini target state value shape: {mini_target_state_value_batch.shape}")

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
                        self.moved_lr: self.lr_schedule(update_ratio),
                        self.clip_epsilon: self.clip_schedule(update_ratio)})

                    if global_step % 100 == 0:
                        print(f"c_loss: {c_loss}  surr: {surr}  entro: {entro[0]}  ratio: {p_ratio[0]} at step {global_step}")

                except StopIteration:
                    del batch_generator
                    break

        if (update_ratio / self.save_model_freq) % 1 == 0:
            self.save_model()

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]

        index = np.arange(n_sample)
        np.random.shuffle(index)

        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index] if x.ndim == 1 else x[span_index, :] for x in data_batch]
