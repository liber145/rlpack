import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter


class Base(ABC):
    """Algorithm base class."""

    def __init__(self, save_path=None, rnd=1):
        # Save.
        self.rnd = rnd
        self.save_path = save_path

        # ------------------------ Reset graph ------------------------
        tf.reset_default_graph()
        tf.set_random_seed(self.rnd)
        np.random.seed(self.rnd)
        tf.Variable(0, name="global_step", trainable=False)
        self.increment_global_step = tf.assign_add(tf.train.get_global_step(), 1)
        self.sw = SummaryWriter(log_dir=self.save_path)

        # ------------------------ Build network ------------------------
        self._build_network()

        # ------------------------ Build algorithm ------------------------
        self._build_algorithm()

        # ------------------------ Initialize model store and reload. ------------------------
        self._prepare()

    @abstractmethod
    def _build_network(self):
        """Build tensorflow operations for algorithms."""
        pass

    @abstractmethod
    def _build_algorithm(self):
        """Build algorithms using prebuilt networks."""
        pass

    def _prepare(self):
        # ------------------------ Initialize saver. ------------------------
        self.saver = tf.train.Saver(max_to_keep=5)

        # ------------------------ Initialize Session. ------------------------
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)

        # ------------------------ Initialize tensorflow variables.  ------------------------
        self.sess.run(tf.global_variables_initializer())

        # ------------------------ Reload model from the saved path. ------------------------
        self.load_model()

        # ------------------------ 初始化其他 ------------------------
        # self.total_reward = 0

    @abstractmethod
    def get_action(self, obs):
        """Return action according to the observations.
        :param obs: the observation that could be image or real-number features
        :return: actions
        """
        pass

    @abstractmethod
    def update(self, minibatch, update_ratio):
        """Update policy using minibatch samples.
        :param minibatch: a minibatch of training data
        :param update_ratio: the ratio of current update step in total update step
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
