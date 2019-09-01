# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from ..utils import diagonal_gaussian_kl, discrete_kl
from .base import Base


class TRPO(Base):
    def __init__(self,
                 rnd=0, is_discrete=False, 
                 dim_obs=None, dim_act=None,
                 policy_fn=None, value_fn=None,
                 discount=0.99, gae=0.95, delta=0.01,
                 train_epoch=40, policy_lr=1e-3, value_lr=1e-3, max_grad_norm=40,
                 save_path="./log", log_freq=10, save_model_freq=100,
                 ):
        self._is_discrete = is_discrete
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._policy_fn = policy_fn
        self._value_fn = value_fn

        self._discount = discount
        self._gae = gae
        self._delta = delta
        self._train_epoch = train_epoch
        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._max_grad_norm = max_grad_norm

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        self._obs = tf.placeholder(tf.float32, [None, *self._dim_obs], "observation")
        
        self._adv = tf.placeholder(tf.float32, [None])
        self._ret = tf.placeholder(tf.float32, [None])

        if self._is_discrete:
            self._act = tf.placeholder(tf.int32, [None], "action")    
            self._old_probs = tf.placeholder(tf.float32, [None, self._dim_act])
            self._old_p = tf.placeholder(tf.float32, [None])
        else:
            self._act = tf.placeholder(tf.float32, [None, self._dim_act], "action")
            self._logp_old = tf.placeholder(tf.float32, [None])
            self._old_mu = tf.placeholder(tf.float32, [None, self._dim_act])
            self._old_log_std = tf.placeholder(tf.float32, [self._dim_act])
            

        with tf.variable_scope("pi"):
            if self._is_discrete:
                self.sampled_a, self.probs, self.p = self._policy_fn(self._obs, self._act)
            else:
                self.sampled_a, self.logp, self.logp_pi, self.mu, self.log_std = self._policy_fn(self._obs, self._act)
        with tf.variable_scope("value"):
            self.v = self._value_fn(self._obs)

        self.policy_vars = tf.trainable_variables("pi")
        size_vec = np.sum([np.prod(x.shape.as_list()) for x in self.policy_vars])
        self.vec = tf.placeholder(tf.float32, [size_vec], "vector")

        if self._is_discrete:
            self.all_phs = [self._obs, self._act, self._adv, self._ret, self._old_probs, self._old_p]
        else:
            self.all_phs = [self._obs, self._act, self._adv, self._ret, self._logp_old, self._old_mu, self._old_log_std]

        for ph in self.all_phs:
            print("??????? ph shape :", ph.shape.as_list())

    def _build_algorithm(self):
        if self._is_discrete:
            self.d_kl = discrete_kl(self._old_probs, self.probs)
        else:
            self.d_kl = diagonal_gaussian_kl(self.mu, self.log_std, self._old_mu, self._old_log_std)
        g_kl = self._flat_param_list(tf.gradients(self.d_kl, self.policy_vars))
        # Add damping vector.
        self.Hv = self._flat_param_list(tf.gradients(tf.reduce_sum(g_kl * self.vec), self.policy_vars))  # + 0.01 * self.vec

        if self._is_discrete:
            ratio = self.p / self._old_p
        else:
            ratio = tf.exp(self.logp - self._logp_old)
        self.policy_loss = -tf.reduce_mean(ratio * self._adv)
        value_loss = tf.reduce_mean((self.v - self._ret)**2)

        self._policy_grad_op = self._flat_param_list(tf.gradients(self.policy_loss, self.policy_vars))
        self._train_value_op = tf.train.AdamOptimizer(self._value_lr).minimize(value_loss)

    def update(self, databatch):
        """
        参数:
            databatch：一个列表，分别是state, action, reward, done, early_stop, next_state。每个是矩阵或向量。
            state是状态，action是动作，reward是奖励，done是是否完结，early_stop是是否提前结束，next_state是下一个状态。
        """
        preprocess_databatch = self._parse_databatch(*databatch)

        inputs = {k: v for k, v in zip(self.all_phs, preprocess_databatch)}

        for i in range(self._train_epoch):
            self.sess.run(self._train_value_op, feed_dict=inputs)

        old_policy_vars = self.sess.run(self._flat_param_list(self.policy_vars))
        policy_grad = self.sess.run(self._policy_grad_op, feed_dict=inputs)
        step_direction = self._conjugate_gradient(policy_grad, inputs)
        max_step_length = np.sqrt(2*self._delta/np.dot(policy_grad, step_direction))

        def func(theta):
            self._recover_param_list(theta)
            return self.sess.run([self.policy_loss, self.d_kl], feed_dict=inputs)

        self._line_search(old_policy_vars, step_direction, max_step_length, func)

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        if global_step % self._save_model_freq == 0:
            self.save_model()

    def _line_search(self, old_theta, step_dir, step_len, target_func, max_backtrack=10):
        fval, _ = target_func(old_theta)
        for i in range(max_backtrack):
            step_frac = 0.5 ** i
            theta = old_theta - step_frac * step_len * step_dir
            new_fval, new_kl = target_func(theta)
            if new_kl < self._delta and new_fval < fval:
                print(f"line search finished in the {i}th backtrack")
                break
            if i == max_backtrack - 1:
                print("line search failed.")
                self._recover_param_list(old_theta)

    def _flat_param_list(self, ts):
        return tf.concat([tf.reshape(t, [-1]) for t in ts], axis=0)

    def _recover_param_list(self, ts):
        res = []
        start = 0
        for param in self.policy_vars:
            shape = param.shape.as_list()
            param_np = np.reshape(ts[start: start + np.prod(shape)], shape)
            res.append(param_np)
            start += np.prod(shape)

        assign_weight_op = [x.assign(y) for x, y in zip(self.policy_vars, res)]
        self.sess.run(assign_weight_op)

    def _conjugate_gradient(self, g, inputs, residual_tol=1e-8, cg_damping=0.1, epoch=20):
        """计算H^{-1} g
        """
        x = np.zeros_like(g)
        p = g.copy()
        r = -g.copy()

        for _ in range(epoch):
            Ap = self.sess.run(self.Hv, feed_dict={**inputs, self.vec: p})

            alpha = np.dot(r, r) / np.dot(p, Ap)
            x = x + alpha * p
            r_new = r + alpha * Ap
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = -r_new + beta * p
            r = r_new

            if np.dot(r, r) < residual_tol:
                break
        return x

    def _parse_databatch(self, states, actions, rewards, dones, earlystops, nextstates):

        batch_size = len(dones)
        if self._is_discrete:
            values, old_probs, old_p = self.sess.run([self.v, self.probs, self.p],
                                                                     feed_dict={self._obs: states, self._act: actions})
        else:
            oldlogproba, values, old_mu, old_log_std = self.sess.run([self.logp, self.v, self.mu, self.log_std],
                                                                     feed_dict={self._obs: states, self._act: actions})
        nextvalues = self.sess.run(self.v, feed_dict={self._obs: nextstates})

        returns = np.zeros(batch_size)
        deltas = np.zeros(batch_size)
        advantages = np.zeros(batch_size)

        for i in reversed(range(batch_size)):

            if dones[i]:
                prev_return = 0
                prev_value = 0
                prev_advantage = 0
            elif earlystops[i]:
                prev_return = nextvalues[i]
                prev_value = prev_return
                prev_advantage = 0

            returns[i] = rewards[i] + self._discount * prev_return * (1 - dones[i])
            deltas[i] = rewards[i] + self._discount * prev_value * (1 - dones[i]) - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + self._discount * self._gae * prev_advantage * (1 - dones[i])

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        advantages = (advantages - advantages.mean()) / advantages.std()

        if self._is_discrete:
            return [states, actions, advantages, returns, old_probs, old_p]
        else:
            return [states, actions, advantages, returns, oldlogproba, old_mu, old_log_std]

    def get_action(self, obs) -> np.ndarray:
        """给定状态，返回动作。状态大小是(batchsize, *obs.dim)
        """
        a = self.sess.run(self.sampled_a, feed_dict={self._obs: obs})
        return a
