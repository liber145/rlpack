import math

import numpy as np
import tensorflow as tf

from .base import Base


class DDPG(Base):
    """Deep Deterministic Policy Gradient."""

    def __init__(self, config):
        self.epsilon = 0.1
        super().__init__(config)

    def build_network(self):
        # Build placeholders.
        self.observation_ph = tf.placeholder(tf.float32, [None, *self.dim_observation], "observation")
        self.action_ph = tf.placeholder(tf.float32, (None, self.dim_action), "action")

        # Build Q-value net.
        with tf.variable_scope("qval_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=True)
            y = tf.layers.dense(self.action_ph, 64, activation=tf.nn.relu, trainable=True)
            z = tf.concat([x, y], axis=1)
            self.qval = tf.layers.dense(z, 1, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)), trainable=True)

        with tf.variable_scope("dummy_qval_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=False)
            y = tf.layers.dense(self.action_ph, 64, activation=tf.nn.relu, trainable=False)
            z = tf.concat([x, y], axis=1)
            self.dummy_qval = tf.layers.dense(z, 1, activation=None, kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1 / 64)), trainable=False)

        # Build action net.
        with tf.variable_scope("act_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=True)
            self.action = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh, trainable=True)

        with tf.variable_scope("dummy_act_net"):
            x = tf.layers.dense(self.observation_ph, 64, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, trainable=False)
            self.dummy_action = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh, trainable=False)

    def build_algorithm(self):
        self.target_qval_ph = tf.placeholder(tf.float32, (None,), "next_state_qval")
        self.grad_q_act_ph = tf.placeholder(tf.float32, (None, self.dim_act), "grad_q_act")

        # ---------- Build Policy Algorithm ----------
        # Compute gradient of qval with respect to action.
        self.grad_q_a = tf.gradients(self.qval, self.action_ph)

        # Compute update direction of policy parameter.
        actor_vars = tf.trainable_variables("act_net")
        grad_surr = tf.gradients(self.action / self.batch_size, actor_vars, -self.grad_q_act_ph)

        # Update actor parameters.
        self.train_actor_op = self.optimizer.apply_gradients(zip(grad_surr, actor_vars), global_step=tf.train.get_global_step())

        # ---------- Build Value Algorithm ----------
        critic_vars = tf.trainable_variables("qval_net")
        self.value_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.qval - self.target_qval_ph), axis=1))

        self.train_critic_op = self.critic_optimizer.minimize(self.value_loss, var_list=critic_vars)

    def update(self, minibatch, update_ratio=None):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        next_action_batch = self.sess.run(self.dummy_action, feed_dict={self.observation_ph: next_s_batch})
        next_qval_batch = self.sess.run(self.dummy_qval, feed_dict={self.observation_ph: next_s_batch, self.action_ph: next_action_batch})
        target_qval_batch = r_batch + (1 - d_batch) * self.discount * next_qval_batch

        batch_generator = self._generator([s_batch, a_batch])
        while True:
            try:

                mb_s, mb_a = next(batch_generator)
                grad = self.sess.run(self.grad_q_a, feed_dict={self.observation_ph: mb_s, self.action_ph: mb_a})
                self.sess.run(self.train_actor_op, feed_dict={self.observation_ph: mb_s, self.action_ph: mb_a, self.grad_q_act_ph: grad})

            except StopIteration:
                del batch_generator
                break

        batch_generator = self._generator([s_batch, a_batch, target_qval_batch])
        while True:
            try:
                mb_s, mb_a, mb_target = next(batch_generator)

                # Update critic.
                _, loss, global_step = self.sess.run([self.train_critic_op, self.value_loss, tf.train.get_global_step()],
                                                     feed_dict={
                    self.observation_ph: mb_s,
                    self.action_ph: mb_a,
                    self.target_qval_ph: mb_target})
            except StopIteration:
                del batch_generator
                break

        self._copy_parameters("qval_net", "dummy_qval_net")
        self._copy_parameters("act_net", "dummy_act_net")

        return {"critic_loss": loss, "global_step": global_step}

    def _copy_parameters(self, netnew, netold):
        """Copy parameters from netnew to netold.

        Parameters:
            netold: string
            netnew: string
        """

        oldvars = tf.trainable_variables(netold)
        newvars = tf.trainable_variables(netnew)

        assign_op = [x.assign(y) for x, y in zip(oldvars, newvars)]
        self.sess.run(assign_op)

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        best_actions = self.sess.run(self.action, feed_dict={self.observation_ph: newobs})
        actions = np.random.randint(self.n_action, size=best_actions.shape[0])
        idx = np.random.uniform(size=best_actions.shape[0]) > self.epsilon
        actions[idx] = best_actions[idx]

        if obs.ndim == 1 or obs.ndim == 3:
            actions = actions[0]
        return actions

    def _generator(self, data_batch, batch_size=32):
        n_sample = data_batch[0].shape[0]
        assert n_sample == self.n_env * self.trajectory_length

        index = np.arange(n_sample)
        np.random.shuffle(index)

        for i in range(math.ceil(n_sample / batch_size)):
            span_index = slice(i * batch_size, min((i + 1) * batch_size, n_sample))
            span_index = index[span_index]
            yield [x[span_index] if x.ndim == 1 else x[span_index, :] for x in data_batch]
