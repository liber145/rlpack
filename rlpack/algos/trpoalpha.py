# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from ..utils import diagonal_gaussian_kl, discrete_alpha
from .base import Base


def diagonal_gaussian_alpha(mu, log_std, old_mu, old_log_std, alpha=0.5):
    s2, s1 = tf.exp(log_std), tf.exp(old_log_std)
    tmp1 = alpha*s1**2 + (1-alpha)*s2**2

    tmp = s2**alpha * s1**(1-alpha) * tmp1**(-0.5) * tf.exp(-0.5*alpha*(1-alpha)*(mu-old_mu)**2 / tmp1)
    all_alphas = (1 - tf.reduce_prod(tmp, axis=1)) / (alpha*(1-alpha))
    return tf.reduce_mean(all_alphas)


class TRPOAlpha(Base):
    def __init__(self,
                 rnd=0,
                 dim_obs=None, dim_act=None, alpha=0.5,
                 policy_fn=None, value_fn=None,
                 discount=0.99, gae=0.95, delta=0.1,
                 train_epoch=40, policy_lr=1e-3, value_lr=1e-3, max_grad_norm=40,
                 save_path="./log", log_freq=10, save_model_freq=100,
                 ):
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._alpha = alpha
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
        self._act = tf.placeholder(tf.int32, [None], "action")

        self._old_probs = tf.placeholder(tf.float32, [None, self._dim_act])
        self._old_p = tf.placeholder(tf.float32, [None])
        # self._logp_old = tf.placeholder(tf.float32, [None])
        self._adv = tf.placeholder(tf.float32, [None])
        self._ret = tf.placeholder(tf.float32, [None])
        # self._old_mu = tf.placeholder(tf.float32, [None, self._dim_act])
        # self._old_log_std = tf.placeholder(tf.float32, [self._dim_act])

        with tf.variable_scope("policy"):
            # self.pi, self.logp, self.logp_pi, self.mu, self.log_std = self._policy_fn(self._obs, self._act)
            print("<<<<<<<<", self._obs)
            print("<<<<<<<<", self._act)
            self.sampled_a, self.probs, self.p = self._policy_fn(self._obs, self._act)
            print("herere !!!!!!!!!!!")
            print("<<<<<<<<<<<<<<<", self.sampled_a)
            print("<<<<<<<<<<<<<<<", self.probs)
            print("<<<<<<<<<<<<<<<", self.p)
        with tf.variable_scope("value"):
            self.v = self._value_fn(self._obs)

        self.policy_vars = tf.trainable_variables("policy")
        size_vec = np.sum([np.prod(x.shape.as_list()) for x in self.policy_vars])
        self.vec = tf.placeholder(tf.float32, [size_vec], "vector")

        # self.all_phs = [self._obs, self._act, self._adv, self._ret, self._logp_old, self._old_mu, self._old_log_std]
        self.all_phs = [self._obs, self._act, self._adv, self._ret, self._old_probs, self._old_p]

        for ph in self.all_phs:
            print("??????? ph shape :", ph.shape.as_list())

    def _build_algorithm(self):
        # self.d_kl = diagonal_gaussian_alpha(self.mu, self.log_std, self._old_mu, self._old_log_std, alpha=self._alpha)
        self.d_alpha = discrete_alpha(self._old_probs, self.probs, alpha=self._alpha)
        g_kl = self._flat_param_list(tf.gradients(self.d_alpha, self.policy_vars))
        # Add damping vector.
        self.Hv = self._flat_param_list(tf.gradients(tf.reduce_sum(g_kl * self.vec), self.policy_vars))  # + 0.01 * self.vec

        # ratio = tf.exp(self.logp - self._logp_old)
        ratio = self.p / self._old_p
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
            return self.sess.run([self.policy_loss, self.d_alpha], feed_dict=inputs)

        self._line_search(old_policy_vars, step_direction, max_step_length, func)

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        if global_step % self._save_model_freq == 0:
            self.save_model()

    def _line_search(self, old_theta, step_dir, step_len, target_func, max_backtrack=10):
        fval, d_alpha = target_func(old_theta)
        for i in range(max_backtrack):
            step_frac = 0.5 ** i
            theta = old_theta - step_frac * step_len * step_dir
            new_fval, new_d_alpha = target_func(theta)
            if new_d_alpha < self._delta and new_fval < fval:
                print(f"line search finished in the {i}th backtrack, new_d_alpha={new_d_alpha}")
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
        # oldlogproba, values, old_mu, old_log_std = self.sess.run([self.logp, self.v, self.mu, self.log_std],
        #                                                          feed_dict={self._obs: states, self._act: actions})
        # print("batch size:", batch_size)

        # print(">>>>>>>> states:", states.shape, np.sum(states))
        # print(">>>>>>>> actions:", actions.shape, np.sum(actions))
        oldprobs, oldp, values = self.sess.run([self.probs, self.p, self.v],
                                               feed_dict={self._obs: states, self._act: actions})
        # print(">>>>>>>> odlprobs:", oldprobs.shape, np.sum(oldprobs))
        # print(">>>>>>>> oldp:", oldp.shape, np.sum(oldp))
        

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

        # return [states, actions, advantages, returns, oldlogproba, old_mu, old_log_std]
        return [states, actions, advantages, returns, oldprobs, oldp]

    def get_action(self, obs) -> np.ndarray:
        """给定状态，返回动作。状态大小是(batchsize, *obs.dim)
        """
        a = self.sess.run(self.sampled_a, feed_dict={self._obs: obs})
        return a
