import numpy as np
import tensorflow as tf
from .baseq import BaseQ
from ..common.utils import assert_shape
import math


class PPO(BaseQ):
    def __init__(self, config):
        self.tau = 0.98
        super().__init__(config)

    def build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, self.dim_observation], "observation")
        with tf.variable_scope("policy_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            self.action_probability = tf.layers.dense(x, self.n_action, activation=tf.nn.softmax)

        with tf.variable_scope("value_net"):
            x = tf.layers.dense(self.observation, 128, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            self.state_value = tf.squeeze(tf.layers.dense(x, 1))

    def build_algorithm(self):
        self.actor_optimizer = tf.train.AdamOptimizer(0.001)
        self.critic_optimizer = tf.train.AdamOptimizer(0.01)
        self.old_action_probability = tf.placeholder(tf.float32, [None, self.n_action])
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self.target_state_value = tf.placeholder(tf.float32, [None], "target_state_value")

        # Get selected action index.
        batch_size = tf.shape(self.observation)[0]
        selected_action_index = tf.stack([tf.range(batch_size), self.action], axis=1)

        # Get action probability.
        old_act_p = tf.gather_nd(self.old_action_probability, selected_action_index)
        assert_shape(old_act_p, [None])
        act_p = tf.gather_nd(self.action_probability, selected_action_index)
        assert_shape(act_p, [None])

        # Compute ratio and surrogate object.
        ratio = act_p / old_act_p
        surrogate_1 = ratio * self.advantage
        surrogate_2 = tf.clip_by_value(act_p / old_act_p, 1.0 - 0.2, 1.0 + 0.2) * self.advantage
        assert_shape(ratio, [None])
        assert_shape(surrogate_1, [None])
        assert_shape(surrogate_2, [None])
        self.surrogate = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))

        # Train actor.
        self.actor_train_op = self.actor_optimizer.minimize(self.surrogate,
                                                            var_list=tf.trainable_variables("policy_net"))

        # Train critic.
        self.critic_loss = tf.reduce_mean(tf.squared_difference(self.state_value, self.target_state_value))
        self.critic_train_op = self.critic_optimizer.minimize(self.critic_loss, global_step=tf.train.get_global_step(),
                                                              var_list=tf.trainable_variables("value_net"))

    def get_action(self, obs):
        assert obs.ndim == 1
        newobs = np.array(obs)[np.newaxis, :]

        act_p = self.sess.run(self.action_probability, feed_dict={self.observation: newobs})
        action = np.random.choice(self.n_action, p=act_p[0])
        return action

    def update(self, minibatch):
        """minibatch is a trajectory.
        """
        s_batch, a_batch, r_batch, next_s_batch, d_batch = minibatch

        batch_size = s_batch.shape[0]
        next_state_value_batch = self.sess.run(self.state_value, feed_dict={self.observation: next_s_batch})
        state_value_batch = self.sess.run(self.state_value, feed_dict={self.observation: s_batch})

        assert next_state_value_batch.ndim == 1
        assert state_value_batch.ndim == 1

        # Compute generalized advantage.
        delta_batch = r_batch + self.discount * (1 - d_batch) * next_state_value_batch - state_value_batch
        advantage_batch = np.empty(batch_size, dtype=np.float32)
        last_advantage = 0
        for t in reversed(range(batch_size)):
            advantage_batch[t] = delta_batch[t] + self.discount * self.tau * (1 - d_batch[t]) * last_advantage
            last_advantage = advantage_batch[t]

        target_state_value_batch = advantage_batch + state_value_batch
        old_action_probability_batch = self.sess.run(self.action_probability, feed_dict={self.observation: s_batch})

        # Train network.
        for _ in range(10):
            # Get training sample generator.
            batch_generator = self._generator([s_batch, a_batch, advantage_batch, old_action_probability_batch,
                                               target_state_value_batch], batch_size=self.batch_size)
            # Train actor.
            while True:
                try:
                    mini_s_batch, mini_a_batch, mini_advantage_batch, mini_old_action_probability_batch, mini_target_state_value_batch = next(
                        batch_generator)

                    # print(f"mini target state value shape: {mini_target_state_value_batch.shape}")

                    # Train actor.
                    self.sess.run(self.actor_train_op, feed_dict={
                        self.observation: mini_s_batch,
                        self.old_action_probability: mini_old_action_probability_batch,
                        self.action: mini_a_batch,
                        self.advantage: mini_advantage_batch})

                    # Train Critic.
                    self.sess.run(self.critic_train_op, feed_dict={
                        self.observation: mini_s_batch,
                        self.target_state_value: mini_target_state_value_batch})
                except StopIteration:
                    del batch_generator
                    break

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]
        index = np.arange(n_sample)
        np.random.shuffle(index)

        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i*batch_size, min((i+1)*batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index] if x.ndim == 1 else x[span_index, :] for x in data_batch]
