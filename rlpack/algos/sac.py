import tensorflow as tf
from .base import Base
import numpy as np
from ..common.utils import assert_shape


class SAC(Base):
    def __init__(self, config):
        self.tau = 0.995
        self.action_high = 1.0
        self.action_low = -1.0
        super().__init__(config)
        self.sess.run(self.init_target_vf)

    def build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="observation")
        self.action = tf.placeholder(tf.float32, [None, self.dim_action], name="action")
        self.target_observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="target_observation")

        with tf.variable_scope("policy_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            self.mu = tf.layers.dense(x, self.dim_action, activation=tf.nn.tanh)
            self.log_std = self.log_var = tf.get_variable("log_var", [self.dim_action], tf.float32, tf.constant_initializer(0.0))

        with tf.variable_scope("state_value_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            self.sval = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("target_state_value_net"):
            x = tf.layers.dense(self.target_observation, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu, )
            self.target_sval = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("action_value_net"):
            x = tf.concat([self.observation, self.action], axis=1)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            self.qval = tf.squeeze(tf.layers.dense(x, 1))

        with tf.variable_scope("action_value_net", reuse=True):
            x = tf.concat([self.observation, self.mu], axis=1)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            x = tf.layers.dense(x, 64, activation=tf.nn.relu)
            self.qval_mu = tf.squeeze(tf.layers.dense(x, 1))

    def build_algorithm(self):
        self.actor_optimizer = tf.train.AdamOptimizer(3e-4)
        self.sval_optimizer = tf.train.AdamOptimizer(3e-4)
        self.qval_optimizer = tf.train.AdamOptimizer(3e-4)

        self.reward = tf.placeholder(tf.float32, [None], name="reward")
        self.mask = tf.placeholder(tf.float32, [None], name="mask")   # 1 - done

        # Compute log(pi).
        var = tf.exp(self.log_std * 2)
        logp = -0.5 * ((self.action - self.mu) ** 2 / var + 2 * self.log_std + tf.log(2 * np.pi))
        logp = tf.reduce_sum(logp, axis=1)
        assert_shape(logp, [None])

        # Compute VF loss.
        tmp = 0.5 * (self.sval - tf.stop_gradient(self.qval - logp)) ** 2
        VF_loss = tf.reduce_mean(tmp)

        # Comput QF loss.
        target = tf.stop_gradient(self.reward + self.discount * self.mask * self.target_sval)
        tmp = 0.5 * (self.qval - target) ** 2
        QF_loss = tf.reduce_mean(tmp)

        # Compute Policy loss.
        self.n_output_action = tf.placeholder(tf.int32)
        tmp_action = self.mu + tf.random_normal(shape=[self.n_output_action, self.dim_action], mean=0.0, stddev=0.1)
        tmp_logp = logp = -0.5 * ((tmp_action - self.mu) ** 2 / var + 2 * self.log_std + tf.log(2 * np.pi))
        tmp_logp = tf.reduce_sum(tmp_logp, axis=1)
        Policy_loss = tmp_logp - self.qval_mu
        assert_shape(tmp_logp, [None])

        def _update_target(new_net, old_net, bili=0.0):
            """new is target net. old is updated net."""
            params1 = tf.trainable_variables(new_net)
            params2 = tf.trainable_variables(old_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param1.assign(param1 * bili + param2 * (1 - bili)))
            return update_ops

        self.update_vf = self.sval_optimizer.minimize(VF_loss)
        self.update_qf = self.qval_optimizer.minimize(QF_loss)
        actor_vars = tf.trainable_variables("policy_net")
        g = tf.gradients(Policy_loss, actor_vars)
        self.update_policy = self.actor_optimizer.apply_gradients(zip(g, actor_vars))
        self.update_target_vf = _update_target("target_state_value_net", "state_value_net", bili=self.tau)
        self.init_target_vf = _update_target("target_state_value_net", "state_value_net", bili=0.0)

    def update(self, minibatch, update_ratio: float):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch
        n_env, batch_size = s_batch.shape[:2]
        s_batch = s_batch.reshape(n_env * batch_size, *self.dim_observation)
        a_batch = a_batch.reshape(n_env * batch_size, self.dim_action)
        r_batch = r_batch.reshape(n_env * batch_size)
        d_batch = d_batch.reshape(n_env * batch_size)
        next_s_batch = next_s_batch.reshape(n_env * batch_size, *self.dim_observation)

        self.sess.run(self.update_vf, feed_dict={self.observation: s_batch, self.action: a_batch})
        self.sess.run(self.update_qf, feed_dict={
                                        self.observation: s_batch,
                                        self.action: a_batch,
                                        self.target_observation: next_s_batch,
                                        self.reward: r_batch,
                                        self.mask: 1 - d_batch})
        self.sess.run(self.update_policy, feed_dict={self.observation: s_batch, self.n_output_action: n_env*batch_size})
        self.sess.run(self.update_target_vf)

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        mu, log_std = self.sess.run([self.mu, self.log_std], feed_dict={self.observation: newobs})
        action = mu + np.random.normal(scale=np.exp(log_std), size=mu.shape)
        action = np.clip(action, self.action_low, self.action_high)
        return action
