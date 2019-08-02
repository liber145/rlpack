import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter


class Base(ABC):
    """Algorithm base class."""

    def __init__(self, save_path=None, rnd=1):
        self.rnd = rnd
        self.save_path = save_path

        # ------------------------ Reset graph ------------------------
        # tf.reset_default_graph()
        tf.set_random_seed(self.rnd)
        np.random.seed(self.rnd)
        # tf.Variable(0, name="global_step", trainable=False)
        # self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)
        self.sw = SummaryWriter(log_dir=self.save_path)

        # ------------------------ Build network ------------------------
        self._build_network()

        # ------------------------ Build algorithm ------------------------
        self._build_algorithm()

        # ------------------------ Initialize model store and reload. ------------------------
        self._prepare()

    @abstractmethod
    def _build_network(self):
        """构建网络。"""
        pass

    @abstractmethod
    def _build_algorithm(self):
        """构建算法。"""
        pass

    def _prepare(self):
        # ------------------------ 初始化保存。 ------------------------
        self.saver = tf.train.Saver(max_to_keep=5)
        tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)

        # ------------------------ 初始化session. ------------------------
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)

        # ------------------------ 初始化变量。  ------------------------
        self.sess.run(tf.global_variables_initializer())

        # ------------------------ 如果有模型，加载模型。 ------------------------
        self.load_model()

    @abstractmethod
    def get_action(self, obs):
        """Return action according to the observations.
        :param obs: the observation that could be image or real-number features
        :return: actions
        """
        pass

    @abstractmethod
    def update(self, minibatch):
        """Update policy using minibatch samples.
        :param minibatch: a minibatch of training data
        :return: update info, i.e. loss.
        """
        pass

    def save_model(self):
        """Save model to `save_path`."""
        save_dir = os.path.join(self.save_path, "model")
        os.makedirs(save_dir, exist_ok=True)
        global_step = self.sess.run(tf.train.get_global_step())
        self.saver.save(
            self.sess,
            os.path.join(save_dir, "model"),
            global_step,
            write_meta_graph=True
        )

    def load_model(self):
        """Load model from `save_path` if there exists."""
        latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.save_path, "model"))
        if latest_checkpoint:
            print("## Loading model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("## New start!")

    def add_scalar(self, *args):
        return self.sw.add_scalar(*args)

    def add_scalars(self, *args):
        return self.sw.add_scalars(*args)
