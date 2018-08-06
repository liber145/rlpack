import numpy as np
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
from middleware.log import logger


class PG(TFEstimator):
    """Policy Gradient for coutinuous action."""

    def __init__(self, dim_ob, dim_ac, lr=1e-4, discount=0.999):
        self.dim_ac = dim_ac
        super().__init__(dim_ob, None, lr, discount)

    def _build_model(self):
        self.input = tf.placeholder(tf.float32, [None, self.dim_ob], "inputs")
        self.action = tf.placeholder(tf.float32, [None, self.dim_ac], "action")
        self.qval = tf.placeholder(tf.float32, [None, 1], "Q-values")

        # Build net.
        with tf.variable_scope("gauss_net"):
            self.mu = Networker.build_pg_net(
                self.input, [64, 64, self.dim_ac])

        trainable_variables = tf.trainable_variables("gauss_net")

        # TODO: sigma 可能是负值。
        self.sigma = np.array(1, dtype=np.float32)
        self.logit = -0.5 * self.dim_ac * tf.log(self.sigma) - 0.5 * self.sigma * tf.reduce_sum(
            tf.square(self.action - self.mu), axis=1, keepdims=True)

        logger.debug("logit shape: {}".format(self.logit.shape))

        # maximize target.
        self.target = tf.reduce_mean(self.qval * self.logit)

        self.train_op = self.optimizer.minimize(
            self.target, global_step=tf.train.get_global_step(), var_list=trainable_variables)

    def update(self, trajectories):
        sar = []
        for traj in trajectories:
            # traj = [[S, A, R, S, D], [S, A, R, S, D], ...]
            sar.extend(self._process_traj(traj))

        state_batch, action_batch, reward_batch = map(np.array, zip(*sar))
        reward_batch = reward_batch[:, np.newaxis]

        _, total_t, target, logit, qval = self.sess.run(
            [self.train_op,
             tf.train.get_global_step(),
             self.target,
             self.logit,
             self.qval],
            feed_dict={
                self.input: state_batch,
                self.action: action_batch,
                self.qval: reward_batch
            })

        if total_t % 100 == 0:
            print("step: {} surr:{}".format(total_t, target))

        return total_t, {"loss": target, "logit": logit, "qval": qval}

    def _process_traj(self, traj):
        res = []
        total_reward = 0
        for transition in reversed(traj):
            total_reward = transition[2] + total_reward * self.discount
            res.append([transition[0], transition[1], total_reward])
        return res

    def get_action(self, ob, epsilon=None):
        action = self.sess.run(self.mu, feed_dict={self.input: ob})
        return action
