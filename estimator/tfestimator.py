import os
import numpy as np
import tensorflow as tf
from middleware.log import logger


class TFEstimator(object):
    def __init__(self, config):  # dim_ob, n_act, lr = 1e-4, discount = 0.99):
        self.dim_ob = config.dim_observation
        self.n_act = config.n_action
        self.dim_act = config.dim_action
        self.discount = config.discount
        self.batch_size = config.batch_size
        self.epsilon = config.epsilon
        self.update_target_every = config.update_target_every
        self.n_dqn = config.n_dqn

        self.optimizer = tf.train.AdamOptimizer(config.lr, epsilon=1.5e-8)
        self.critic_optimizer = tf.train.AdamOptimizer(config.critic_lr)
        self._prepare()

    def _prepare(self):
        tf.reset_default_graph()
        tf.Variable(0, name='global_step', trainable=False)
        self._build_model()

        self.saver = tf.train.Saver(max_to_keep=5)
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def _build_model(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_action(self, obs, epsilon):
        raise NotImplementedError

    def save_model(self, outdir):
        total_t = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            os.path.join(outdir, 'model'),
            total_t,
            write_meta_graph=False)

    def load_model(self, outdir):
        latest_checkpoint = tf.train.latest_checkpoint(outdir)
        if latest_checkpoint:
            logger.info("Loading model checkpoint {}...".format(
                latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            logger.info("New start!!")
