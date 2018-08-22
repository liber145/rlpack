import numpy as np
import copy
import tensorflow as tf
from .tfestimator import TFEstimator
from .networker import Networker
from . import utils
from ..common.log import logger


class TRPO(TFEstimator):
    """Trust Region Policy Optimization."""

    def __init__(self, config):
        self.delta = 0.01
        super().__init__(config)

    def _build_model(self):
        # Build inputs.
        self.input = tf.placeholder(tf.float32, [None]+list(self.dim_ob), "inputs")
        self.action = tf.placeholder(
            tf.float32, [None, self.dim_act], "action")
        self.span_reward = tf.placeholder(tf.float32, [None, 1], "span_reward")
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantages")

        self.old_log_var = tf.placeholder(tf.float32, [self.dim_act], "olvar")
        self.old_mu = tf.placeholder(
            tf.float32, [None, self.dim_act], "old_mu")

        # Build Nets.
        with tf.variable_scope("gauss_net"):
            self.mu, self.log_var = Networker.build_gauss_net(self.input,
                                                              [64, 64, self.dim_act])

        with tf.variable_scope("value_net"):
            self.val = Networker.build_value_net(self.input, [128, 64, 32, 1])

        self.actor_vars = tf.trainable_variables("gauss_net")

        # ------------ Compute g of object. -------------
        logp = -0.5 * tf.reduce_sum(self.log_var)
        logp += -0.5 * tf.reduce_sum(tf.square(self.action - self.mu) / tf.exp(self.log_var),
                                     axis=1,
                                     keepdims=True)

        logp_old = -0.5 * tf.reduce_sum(self.old_log_var)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_mu) / tf.exp(self.old_log_var),
                                         axis=1,
                                         keepdims=True)

        self.obj = -tf.reduce_mean(self.advantage * tf.exp(logp - logp_old))

        # Compute gradient of object.
        self.g = self._flat_param_list(tf.gradients(self.obj, self.actor_vars))

        # ------------ Compute direction by conjugate gradient. ------------
        # Compute gradient of kl divergence.
        log_det_cov_old = tf.reduce_sum(self.old_log_var)
        log_det_cov_new = tf.reduce_sum(self.log_var)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_var - self.log_var))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.mu - self.old_mu) / tf.exp(self.log_var), axis=1) -
                                       self.dim_act)

        g_kl = self._flat_param_list(tf.gradients(self.kl, self.actor_vars))
        logger.debug("g_kl shape: {}".format(g_kl.shape.as_list()))

        # Compute Hessian of KL divergence and the product with a vector.
        size_vec = np.sum([np.prod(v.shape.as_list())
                           for v in self.actor_vars])
        self.vec = tf.placeholder(tf.float32, [size_vec], "vector")
        # add damping vector.
        self.Hv = self._flat_param_list(tf.gradients(tf.reduce_sum(g_kl * self.vec),
                                                     self.actor_vars)) + 0.1 * self.vec

        # Compute update direction by conjugate gradient.

        # Compute update step by line search.

        # ---------- Build critic algorithm ----------
        self.critic_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "value_net")

        self.critic_loss = tf.reduce_mean(
            tf.square(self.val - self.span_reward))

        self.train_critic_op = self.critic_optimizer.minimize(
            self.critic_loss, global_step=tf.train.get_global_step(), var_list=self.critic_vars)

        # ---------- Build action ----------
        self.sampled_act = (self.mu +
                            tf.exp(self.log_var / 2.0) * tf.random_normal(shape=[self.dim_act], dtype=tf.float32))

    def update(self, trajectories):
        data_batch = utils.trajectories_to_batch(trajectories, self.discount)

        self.feeddict = {self.input: data_batch["state"],
                         self.action: data_batch["action"],
                         self.span_reward: data_batch["spanreward"],
                         }

        old_mu_val, old_log_var_val = self.sess.run([self.mu, self.log_var],
                                                    feed_dict=self.feeddict)

        self.feeddict[self.old_mu] = old_mu_val
        self.feeddict[self.old_log_var] = old_log_var_val

        # ---------- Update actor ----------
        # Compute advantage.
        nextstate_val = self.sess.run(self.val,
                                      feed_dict={self.input: data_batch["nextstate"]})
        state_val = self.sess.run(self.val,
                                  feed_dict={self.input: data_batch["state"]})

        advantage = (data_batch["reward"] +
                     self.discount * (1 - data_batch["done"]) * nextstate_val) - state_val
        self.feeddict[self.advantage] = advantage

        # Compute update direction.
        g_obj = self.sess.run(self.g, feed_dict=self.feeddict)
        step_dir = self._conjugate_gradient(-g_obj)  # pylint: disable=E1130

        # Compute max step length.
        self.feeddict[self.vec] = step_dir
        Mx = self.sess.run(self.Hv, feed_dict=self.feeddict)
        max_step_len = np.sqrt(self.delta / (0.5 * np.dot(step_dir, Mx)))

        # Line search to update theta.
        old_theta = self.sess.run(self._flat_param_list(self.actor_vars))
        theta, _ = self._line_search(
            old_theta, step_dir, max_step_len, self._target_func, max_backtrack=5)

        # Assign theta to actor parameters.
        self._recover_param_list(theta)

        obj_val, kl_val = self.sess.run([self.obj, self.kl],
                                        feed_dict=self.feeddict)
        print("obj_val:", obj_val)
        print("kl_val:", kl_val)

        # ---------- Update critic ----------
        # wrong algorihtm.
        # laststate_val = self.sess.run(
        #     self.val, feed_dict={self.input: data_batch["laststate"]})
        # target_val = self.discount * \
        #     data_batch["lastdone"] * laststate_val + data_batch["spanreward"]
        # self.feeddict[self.target_val] = target_val
        # self.feeddict[self.target_val] = data_batch["spanreward"]

        critic_loss = self.sess.run(self.critic_loss, feed_dict=self.feeddict)

        print("old critic loss:", critic_loss)
        for _ in range(10):
            _, total_t, critic_loss = self.sess.run(
                [self.train_critic_op, tf.train.get_global_step(), self.critic_loss], feed_dict=self.feeddict)

        print("new critic loss:", critic_loss, "\n", "-"*20)

        return total_t, {"loss": critic_loss}

    def _target_func(self, theta):

        self._recover_param_list(theta)

        return self.sess.run([self.obj, self.kl], feed_dict=self.feeddict)

    def _line_search(self, old_theta, step_dir, step_len, target_func, max_backtrack=10):
        fval = target_func(old_theta)[0]
        for i in range(max_backtrack):
            step_frac = 0.5 ** i
            theta = step_frac * step_len * step_dir + old_theta
            new_fval, new_kl = target_func(theta)
            if new_kl > 1e-2:
                new_fval += np.inf
            actual_improve = fval - new_fval
            if actual_improve > 0:
                print("step frac:", step_frac)
                return theta, True
        return old_theta, False

    def _flat_param_list(self, ts):
        return tf.concat([tf.reshape(t, [-1]) for t in ts], axis=0)

    def _recover_param_list(self, ts):
        """ts是一个长向量表示的参数。"""
        res = []
        start = 0
        for param in self.actor_vars:
            shape = param.shape.as_list()
            param_np = np.reshape(ts[start: start+np.prod(shape)], shape)
            res.append(param_np)
            start += np.prod(shape)

        assign_weight_op = [x.assign(y) for x, y in zip(self.actor_vars, res)]
        self.sess.run(assign_weight_op)

    def _conjugate_gradient(self, g, residual_tol=1e-8, cg_damping=0.1):
        # Solve Mx = g . Mx is computed by self.Hv.
        # x,p,r are np.array.

        x = np.zeros_like(g)
        p = g.copy()
        r = -g.copy()

        for _ in range(10):
            self.feeddict[self.vec] = p
            Ap = self.sess.run(
                self.Hv, feed_dict=self.feeddict)

            alpha = np.dot(r, r) / np.dot(p, Ap)
            x = x + alpha * p
            r_new = r + alpha * Ap
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = -r_new + beta * p
            r = r_new

            if np.dot(r, r) < residual_tol:
                break

        return x

    def get_action(self, ob, epsilon=None):
        action = self.sess.run(self.sampled_act, feed_dict={self.input: ob})
        return action
