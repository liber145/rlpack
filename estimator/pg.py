import numpy as np
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
import estimator.utils as utils
from middleware.log import logger


class PG(TFEstimator):
    """Policy Gradient for coutinuous action."""

    def __init__(self, config):
        super().__init__(config)

    def _build_model(self):
        self.input = tf.placeholder(tf.float32, [None]+ list(self.dim_ob), "inputs")
        self.action = tf.placeholder(tf.float32, [None, self.dim_act], "action")
        self.qval = tf.placeholder(tf.float32, [None, 1], "Q-values")

        # Build net.
        with tf.variable_scope("gauss_net"):
            self.mu = Networker.build_pg_net(
                self.input, [64, 64, self.dim_act])

        trainable_variables = tf.trainable_variables("gauss_net")

        # TODO: sigma 可能是负值。
        self.sigma = np.array(1, dtype=np.float32)
        self.logit = -0.5 * self.dim_act * tf.log(self.sigma) - 0.5 * self.sigma * tf.reduce_sum(
            tf.square(self.action - self.mu), axis=1, keepdims=True)

        logger.debug("logit shape: {}".format(self.logit.shape))

        # maximize target.
        self.target = tf.reduce_mean(self.qval * self.logit)

        self.train_op = self.optimizer.minimize(
            self.target, global_step=tf.train.get_global_step(), var_list=trainable_variables)

    def update(self, trajectories):

        data_batch = utils.trajectories_to_batch(trajectories, self.discount)
        batch_generator = utils.generator(data_batch, self.batch_size)

        while True:
            try:
                sample_batch = next(batch_generator)
                state_batch = sample_batch["state"]
                action_batch = sample_batch["action"]
                reward_batch = sample_batch["spanreward"]

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
            except StopIteration:
                del batch_generator
                break

        return total_t, {"loss": target, "logit": logit, "q_value": qval}

    def get_action(self, ob, epsilon=None):
        action = self.sess.run(self.mu, feed_dict={self.input: ob})
        return action
