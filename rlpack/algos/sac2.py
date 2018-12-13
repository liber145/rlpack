import sys

import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class SAC2(Base):
    def __init__(self, config):
        self.tau = 0.995
        self.action_high = 1.0
        self.action_low = -1.0
        self.policy_mean_reg_weight = 1e-3
        self.policy_std_reg_weight = 1e-3
        self.alpha = 0.2
        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0
        self.EPS = 1e-8
        super().__init__(config)
        self.sess.run(self.init_target)

    # def mlp(self, x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    #     for h in hidden_sizes[:-1]:
    #         x = tf.layers.dense(x, units=h, activation=activation)
    #     return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    # def get_vars(self, scope):
    #     return [x for x in tf.global_variables() if scope in x.name]

    # def gaussian_likelihood(self, x, mu, log_std):
    #     pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+self.EPS))**2 + 2*log_std + np.log(2*np.pi))
    #     return tf.reduce_sum(pre_sum, axis=1)

    # def clip_but_pass_gradient(self, x, l=-1., u=1.):
    #     clip_up = tf.cast(x > u, tf.float32)
    #     clip_low = tf.cast(x < l, tf.float32)
    #     return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)

    # """
    # Policies
    # """
    # def mlp_gaussian_policy(self, x, a, hidden_sizes, activation, output_activation):
    #     act_dim = a.shape.as_list()[-1]
    #     net = self.mlp(x, list(hidden_sizes), activation, activation)
    #     mu = tf.layers.dense(net, act_dim, activation=output_activation)

    #     """
    #     Because algorithm maximizes trade-off of reward and entropy,
    #     entropy must be unique to state---and therefore log_stds need
    #     to be a neural network output instead of a shared-across-states
    #     learnable parameter vector. But for deep Relu and other nets,
    #     simply sticking an activationless dense layer at the end would
    #     be quite bad---at the beginning of training, a randomly initialized
    #     net could produce extremely large values for the log_stds, which
    #     would result in some actions being either entirely deterministic
    #     or too random to come back to earth. Either of these introduces
    #     numerical instability which could break the algorithm. To
    #     protect against that, we'll constrain the output range of the
    #     log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is
    #     slightly different from the trick used by the original authors of
    #     SAC---they used tf.clip_by_value instead of squashing and rescaling.
    #     I prefer this approach because it allows gradient propagation
    #     through log_std where clipping wouldn't, but I don't know if
    #     it makes much of a difference.
    #     """
    #     log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    #     log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

    #     std = tf.exp(log_std)
    #     pi = mu + tf.random_normal(tf.shape(mu)) * std
    #     logp_pi = self.gaussian_likelihood(pi, mu, log_std)
    #     return mu, pi, logp_pi

    # def apply_squashing_func(self, mu, pi, logp_pi):
    #     mu = tf.tanh(mu)
    #     pi = tf.tanh(pi)
    #     # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    #     logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    #     return mu, pi, logp_pi

    # """
    # Actor-Critics
    # """
    # def mlp_actor_critic(self, x, a, policy, hidden_sizes=(400,300), activation=tf.nn.relu,
    #                      output_activation=None, action_space=None):
    #     # policy
    #     with tf.variable_scope('pi'):
    #         mu, pi, logp_pi = policy(x, a, hidden_sizes, activation, output_activation)
    #         mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)

    #     # make sure actions are in correct range
    #     # action_scale = action_space.high[0]
    #     # mu *= action_scale
    #     # pi *= action_scale

    #     # vfs
    #     vf_mlp = lambda x : tf.squeeze(self.mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    #     with tf.variable_scope('q1'):
    #         q1 = vf_mlp(tf.concat([x,a], axis=-1))
    #     with tf.variable_scope('q1', reuse=True):
    #         q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    #     with tf.variable_scope('q2'):
    #         q2 = vf_mlp(tf.concat([x,a], axis=-1))
    #     with tf.variable_scope('q2', reuse=True):
    #         q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
    #     with tf.variable_scope('v'):
    #         v = vf_mlp(x)
    #     return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v

    # ---------------------

    # def mlp(self, x, hidden_sizes=(100, 100), activation=None, output_activation=None):
    #     for h in hidden_sizes[:-1]:
    #         x = tf.layers.dense(x, units=h, activation=activation)
    #     return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    # def gaussian_likelihood(self, x, mu, log_std):
    #     presum = -0.5 * (((x-mu)/(tf.exp(log_std)+self.EPS))**2 + np.log(2*np.pi) + 2*log_std)
    #     return tf.reduce_sum(presum, axis=1)

    def clip_but_pass_gradient(self, x, l=-1, u=1):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient(clip_up * (u - x) + clip_low * (l - x))

    # def mlp_gaussian_policy(self, x, a, hidden_sizes, activation, output_activation):
    #     act_dim = a.shape.as_list()[-1]
    #     net = self.mlp(x, list(hidden_sizes), activation, activation)
    #     mu = tf.layers.dense(net, act_dim, activation=output_activation)
    #     # mu = tf.layers.dense(net, act_dim, activation=tf.tanh)   # Change output_activation to tf.tanh

    #     log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    #     log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

    #     # normal_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale_diag=tf.exp(log_std))
    #     # pi = normal_dist.sample()
    #     # logp_pi = normal_dist.log_prob(pi)

    #     std = tf.exp(log_std)
    #     pi = mu + tf.random_normal(tf.shape(mu)) * std
    #     logp_pi = self.gaussian_likelihood(pi, mu, log_std)
    #     return mu, pi, logp_pi

    # def apply_squashing_func(self, mu, pi, logp_pi):
    #     mu = tf.tanh(mu)
    #     pi = tf.tanh(pi)
    #     logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    #     return mu, pi, logp_pi

    # def mlp_actor_critic(self, x, a, policy, hidden_sizes=(100, 100), activation=tf.nn.relu, output_activation=None):
    #     with tf.variable_scope("pi"):
    #         mu, pi, logp_pi = self.mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation)
    #         mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)

    #     def vf_mlp(x): return tf.squeeze(self.mlp(x, list(hidden_sizes)+[1], activation=activation, output_activation=None), axis=1)
    #     with tf.variable_scope("q1"):
    #         q1 = vf_mlp(tf.concat([x, a], axis=-1))
    #     with tf.variable_scope("q1", reuse=True):
    #         q1_pi = vf_mlp(tf.concat([x, pi], axis=-1))
    #     with tf.variable_scope("q2"):
    #         q2 = vf_mlp(tf.concat([x, a], axis=-1))
    #     with tf.variable_scope("q2", reuse=True):
    #         q2_pi = vf_mlp(tf.concat([x, pi], axis=-1))
    #     with tf.variable_scope("v"):
    #         v = vf_mlp(x)
    #     return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v

    def build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="observation")
        self.action = tf.placeholder(tf.float32, [None, self.dim_action], name="action")
        self.reward = tf.placeholder(tf.float32, [None], name="reward")
        self.done = tf.placeholder(tf.float32, [None], name="done")
        self.next_observation = tf.placeholder(tf.float32, [None, *self.dim_observation], name="target_observation")

        with tf.variable_scope("main/pi"):
            x = tf.layers.dense(self.observation, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.mu = tf.layers.dense(x, units=self.dim_action, activation=None)
            self.log_std = tf.layers.dense(x, units=self.dim_action, activation=tf.tanh)
            self.log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (self.log_std + 1)

            std = tf.exp(self.log_std)
            self.pi = self.mu + tf.random_normal(tf.shape(self.mu)) * std
            presum = -0.5 * (((self.pi-self.mu)/(tf.exp(self.log_std)+self.EPS))**2 + np.log(2*np.pi) + 2*self.log_std)
            self.logp_pi = tf.reduce_sum(presum, axis=1)

            # normal_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.mu, scale_diag=tf.exp(self.log_std))
            # self.pi = normal_dist.sample()
            # self.logp_pi = normal_dist.log_prob(self.pi)

            # Squash into an appropriate scale.
            self.mu = tf.tanh(self.mu)
            self.pi = tf.tanh(self.pi)
            print(f"pi shape: {self.pi.shape}")
            self.logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - self.pi**2, l=0, u=1) + 1e-6), axis=1)

        with tf.variable_scope("main/q1"):
            x = tf.concat([self.observation, self.action], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q1 = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/q1", reuse=True):
            x = tf.concat([self.observation, self.pi], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q1_pi = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/q2"):
            x = tf.concat([self.observation, self.action], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q2 = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/q2", reuse=True):
            x = tf.concat([self.observation, self.pi], axis=-1)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.q2_pi = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("main/v"):
            x = tf.layers.dense(self.observation, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.v = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        with tf.variable_scope("target/v"):
            x = tf.layers.dense(self.next_observation, units=100, activation=tf.nn.relu)
            x = tf.layers.dense(x, units=100, activation=tf.nn.relu)
            self.v_targ = tf.squeeze(tf.layers.dense(x, units=1, activation=None), axis=1)

        # with tf.variable_scope("main"):
        #     self.mu, self.pi, self.logp_pi, self.q1, self.q2, self.q1_pi, self.q2_pi, self.v = self.mlp_actor_critic(self.observation, self.action, policy=self.mlp_gaussian_policy, hidden_sizes=[100, 100])
        # with tf.variable_scope("target"):
        #     _, _, _, _, _, _, _, self.v_targ = self.mlp_actor_critic(self.next_observation, self.action, policy=self.mlp_gaussian_policy, hidden_sizes=[100, 100])

    def build_algorithm(self):
        min_q_pi = tf.minimum(self.q1_pi, self.q2_pi)

        q_backup = tf.stop_gradient(self.reward + self.discount * (1 - self.done) * self.v_targ)
        v_backup = tf.stop_gradient(min_q_pi - self.alpha * self.logp_pi)

        # Soft actor-critic loss
        self.pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - self.q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        self.value_loss = q1_loss + q2_loss + v_loss

        # Train policy.
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.update_policy = pi_optimizer.minimize(self.pi_loss, var_list=self.get_vars("main/pi"))

        # Train value.
        value_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        value_vars = self.get_vars("main/q") + self.get_vars("main/v")
        with tf.control_dependencies([self.update_policy]):
            self.update_value = value_optimizer.minimize(self.value_loss, var_list=value_vars)

        # main_vars = self.get_vars("main")
        # target_vars = self.get_vars("target")
        main_vars = self.get_vars("main/v")
        target_vars = self.get_vars("target/v")
        print(f"main vars: {main_vars}")
        print(f"targ vars: {target_vars}")
        with tf.control_dependencies([self.update_value]):
            self.update_target = tf.group([tf.assign(v_targ, 0.995*v_targ + (1-0.995)*v_main) for v_main, v_targ in zip(main_vars, target_vars)])

        self.first_var = main_vars[0]
        print("main vars:", len(main_vars))
        print("target vars:", len(target_vars))
        self.init_target = tf.group([tf.assign(v_targ, v_main) for v_main, v_targ in zip(main_vars, target_vars)])

    def update(self, minibatch, update_ratio: float):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        n_env, batch_size = s_batch.shape[:2]
        s_batch = s_batch.reshape(n_env * batch_size, *self.dim_observation)
        a_batch = a_batch.reshape(n_env * batch_size, self.dim_action)
        r_batch = r_batch.reshape(n_env * batch_size)
        d_batch = d_batch.reshape(n_env * batch_size)
        next_s_batch = next_s_batch.reshape(n_env * batch_size, *self.dim_observation)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        value_loss, policy_loss, _, _, _ = self.sess.run([self.value_loss, self.pi_loss, self.update_policy, self.update_value, self.update_target], feed_dict={
            self.observation: s_batch,
            self.action: a_batch,
            self.reward: r_batch,
            self.done: d_batch,
            self.next_observation: next_s_batch})

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        action = self.sess.run(self.pi, feed_dict={self.observation: newobs})
        return action
