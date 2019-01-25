import math

import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape, exponential_decay, linear_decay
from .base import Base


class ESPPO(Base):
    def __init__(self,
                 rnd=1,
                 n_env=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 gae=0.95,
                 save_path="./log",
                 save_model_freq=50,
                 vf_coef=1.0,
                 entropy_coef=0.01,
                 max_grad_norm=40,
                 policy_lr_schedule=lambda x: 3e-4,
                 value_lr_schedule=lambda x: 3e-4,
                 trajectory_length=2048,
                 batch_size=64,
                 training_epoch=10,
                 sigma=1.0
                 ):

        self.n_env = n_env
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.discount = discount
        self.gae = gae

        self.batch_size = batch_size
        self.save_model_freq = save_model_freq
        self.trajectory_length = trajectory_length

        self.policy_lr_schedule = policy_lr_schedule
        self.value_lr_schedule = value_lr_schedule

        self.entropy_coef = entropy_coef
        self.critic_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.training_epoch = training_epoch
        self.sigma = sigma

        self.nu = 0.95
        self.D = list()

        super().__init__(save_path=save_path, rnd=rnd)

    def build_network(self):
        """Build networks for algorithm."""
        self.observation = tf.placeholder(tf.float32, [None, *self.dim_obs], name="observation")

        assert_shape(self.observation, [None, *self.dim_obs])

        with tf.variable_scope("main/policy_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.mu = tf.layers.dense(x, self.dim_act, activation=tf.nn.tanh)
            self.log_var = tf.get_variable("logvars", [self.mu.shape.as_list()[1]], tf.float32, tf.constant_initializer(0.0))

        with tf.variable_scope("target/policy_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.target_mu = tf.layers.dense(x, self.dim_act, activation=tf.nn.tanh)
            self.target_log_var = tf.get_variable("target_logvars", [self.target_mu.shape.as_list()[1]], tf.float32, tf.constant_initializer(0.0))

        with tf.variable_scope("value_net"):
            x = tf.layers.dense(self.observation, 64, activation=tf.nn.tanh)
            x = tf.layers.dense(x, 64, activation=tf.nn.tanh)
            self.state_value = tf.squeeze(tf.layers.dense(x, 1, activation=None))

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.clip_epsilon = tf.placeholder(tf.float32)

        self.policy_lr = tf.placeholder(tf.float32)
        self.value_lr = tf.placeholder(tf.float32)
        self.actor_optimizer = tf.train.AdamOptimizer(self.policy_lr)
        self.critic_optimizer = tf.train.AdamOptimizer(self.value_lr)

        self.action = tf.placeholder(tf.float32, [None, self.dim_act], "action")
        self.span_reward = tf.placeholder(tf.float32, [None], "span_reward")
        self.advantage = tf.placeholder(tf.float32, [None], "advantage")
        self.old_mu = tf.placeholder(tf.float32, [None, self.dim_act], "old_mu")
        self.old_log_var = tf.placeholder(tf.float32, [self.dim_act], "old_var")

        def compute_policy_loss(mu, log_var):
            logp = -0.5 * tf.reduce_sum(log_var * 2)
            logp += -0.5 * tf.reduce_sum(tf.square(self.action - mu) / tf.exp(log_var * 2), axis=1)  # 　- 0.5 * math.log(2 * math.pi)

            logp_old = -0.5 * tf.reduce_sum(self.old_log_var * 2)
            logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_mu) / tf.exp(self.old_log_var * 2), axis=1)  # - 0.5 * math.log(2 * math.pi)

            # Build surrogate loss.
            ratio = tf.exp(logp - logp_old)
            surr1 = ratio * self.advantage
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * self.advantage
            surrogate = -tf.reduce_mean(tf.minimum(surr1, surr2))

            return surrogate

        # Update target network.
        def assign_target(a_net, b_net, t=1):
            """a = b + epsilon"""
            params1 = tf.trainable_variables(a_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.trainable_variables(b_net)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            epsilons = []

            if t == 1:
                self.D = list()

            for i in range(len(params1)):
                param1, param2 = params1[i], params2[i]

                if t == 1:
                    self.D.append(tf.Variable(tf.ones_like(param2, dtype=tf.float32)))

                print("param2:", param2.shape)
                print("D[i]:", self.D[i].shape)
                H = self.D[i] / (1 - self.nu**t)
                print("H:", H.shape)
                H_inv = 1 / H
                epsilon = tf.random_normal(param2.shape)

                print("epsilon:", epsilon.shape)
                print("H_inv:", H_inv.shape)

                epsilons.append(epsilon * tf.sqrt(H_inv))
                # print("para2:", param2, param2.shape)
                # input()
                update_ops.append(param1.assign(param2 + epsilon * self.sigma))
            return update_ops, params2, epsilons

        self.surrogate = compute_policy_loss(self.mu, self.log_var)

        # # Initialize D.
        # train_params = tf.trainable_variables("main/policy_net")
        # train_params = sorted(train_params, key=lambda v: v.name)
        # for t_param in train_params:
        #     self.D.append(tf.ones_like(t_param.shape))

        # target theta = original theta + epsilon * gaussian
        self.assign_target_op, policy_params, epsilons_val = assign_target("target/policy_net", "main/policy_net")

        # use dependency to compute target surrogate
        with tf.control_dependencies(self.assign_target_op):
            self.target_surrogate = compute_policy_loss(self.target_mu, self.target_log_var)

            tmp = (self.target_surrogate - self.surrogate) / self.sigma
            gs = [tmp * eps for eps in epsilons_val]

            print("tar:", self.target_surrogate.shape)
            input()

            self.train_actor_op = [tf.assign_sub(param, self.policy_lr * g) for param, g in zip(policy_params, gs)]

            self.update_D_op = [d.assign(self.nu*d + (1-self.nu)*g**2) for d, g in zip(self.D, gs)]

        # logp = -0.5 * tf.reduce_sum(self.log_var * 2)
        # logp += -0.5 * tf.reduce_sum(tf.square(self.action - self.mu) / tf.exp(self.log_var * 2), axis=1)  # 　- 0.5 * math.log(2 * math.pi)

        # # var = tf.exp(self.log_var * 2)
        # # logp = - (self.action - self.mu) ** 2 / (2 * var) - self.log_var  # - 0.5 * math.log(2 * math.pi)
        # # logp = tf.reduce_sum(logp, axis=1)
        # # print(f"logp shape: {logp.shape}")

        # logp_old = -0.5 * tf.reduce_sum(self.old_log_var * 2)
        # logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_mu) / tf.exp(self.old_log_var * 2), axis=1)  # - 0.5 * math.log(2 * math.pi)

        # # var_old = tf.exp(self.old_log_var * 2)
        # # logp_old = - (self.action - self.old_mu) ** 2 / (2 * var_old) - self.old_log_var
        # # logp_old = tf.reduce_sum(logp_old, axis=1)

        # # Compute KL divergence.
        # log_det_cov_old = tf.reduce_sum(self.old_log_var)
        # log_det_cov_new = tf.reduce_sum(self.log_var)
        # tr_old_new = tf.reduce_sum(tf.exp(self.old_log_var - self.log_var))

        # self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new + tf.reduce_sum(
        #     tf.square(self.mu - self.old_mu) / tf.exp(self.log_var), axis=1) - self.dim_act)

        # self.entropy = 0.5 * (self.dim_act + self.dim_act * tf.log(2 * np.pi) + tf.exp(tf.reduce_sum(self.log_var)))

        # # Build surrogate loss.
        # ratio = tf.exp(logp - logp_old)
        # surr1 = ratio * self.advantage
        # surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * self.advantage
        # self.surrogate = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # Build value loss.
        self.critic_loss = tf.reduce_mean(tf.square(self.state_value - self.span_reward))

        # You can also build total loss and clip the gradients.
        # # Build total_loss.
        # self.total_loss = self.surrogate + self.critic_coef * self.critic_loss + self.entropy_coef * self.entropy

        # # Build training operation.
        # grads = tf.gradients(self.total_loss, tf.trainable_variables())
        # clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # self.total_train_op = self.optimizer.apply_gradients(zip(clipped_grads, tf.trainable_variables()))

        # # Build actor operation.
        # actor_vars = tf.trainable_variables("main/policy_net")

        # grads = tf.gradients(self.surrogate, actor_vars)
        # clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # self.train_actor_op = self.actor_optimizer.apply_gradients(zip(clipped_grads, actor_vars))

        # Build critic operation.
        critic_vars = tf.trainable_variables("value_net")
        regularization = 0.001 * tf.reduce_sum([tf.nn.l2_loss(c_var) for c_var in critic_vars])
        self.critic_loss += regularization

        grads = tf.gradients(self.critic_loss, critic_vars)
        clipped_grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.train_critic_op = self.critic_optimizer.apply_gradients(zip(clipped_grads, critic_vars))
        # self.train_critic_op = self.critic_optimizer.minimize(self.critic_loss)

        # Build action sample.
        self.sample_action = self.mu + tf.exp(self.log_var) * tf.random_normal(shape=[self.dim_act], dtype=tf.float32)

    def get_action(self, obs):
        """Return actions according to the given observation.

        Parameters:
            - obs: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n, action_dimension).
        """
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        actions = self.sess.run(self.sample_action, feed_dict={self.observation: newobs})
        return actions

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
        assert s_batch.shape == (self.n_env, self.trajectory_length + 1, *self.dim_obs)

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
        s_batch = s_batch[:, :-1, ...].reshape(self.n_env * self.trajectory_length, *self.dim_obs)
        a_batch = a_batch.reshape(self.n_env * self.trajectory_length, self.dim_act)
        advantage_batch = advantage_batch.reshape(self.n_env * self.trajectory_length)
        target_value_batch = target_value_batch.reshape(self.n_env * self.trajectory_length)

        # Normalize advantage.
        advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-10)

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
                        self.clip_epsilon: 0.2,
                        self.policy_lr: self.policy_lr_schedule(update_ratio)})

                    self.sess.run(self.train_critic_op, feed_dict={
                        self.observation: mb_s,
                        self.span_reward: mb_target_value,
                        self.clip_epsilon: 0.2,
                        self.value_lr: self.value_lr_schedule(update_ratio)})

                except StopIteration:
                    del batch_generator
                    break

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        if global_step % self.save_model_freq == 0:
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
