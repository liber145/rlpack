import os
from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf

class Base(ABC):
    """Algorithm base class."""

    def __init__(self, save_path=None, rnd=1):
        self.rnd = rnd
        self.save_path = save_path

        # ------------------------ Reset graph ------------------------
        # tf.reset_default_graph()
        tf.set_random_seed(self.rnd)
        np.random.seed(self.rnd)

        # ------------------------ Initialize model store and reload. ------------------------
        self._prepare()

    def _prepare(self):
        # ------------------------ 初始化保存。 ------------------------
        self.saver = tf.train.Saver(max_to_keep=20)

        # ------------------------ 初始化session. ------------------------
        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)

        # ------------------------ 初始化变量。  ------------------------
        self.sess.run(tf.global_variables_initializer())

        # ------------------------ 如果有模型，加载模型。 ------------------------
        self.load_model()

    def save_model(self):
        """Save model to `save_path`."""
        save_dir = os.path.join(self.save_path, "models")
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
        latest_checkpoint = tf.train.latest_checkpoint(self.save_path)
        if latest_checkpoint:
            print("## Loading model checkpoint {} ...".format(latest_checkpoint))
            self.saver.restore(self.sess, latest_checkpoint)
        else:
            print("## New start!")