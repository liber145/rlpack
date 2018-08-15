import numpy as np
import tensorflow as tf


class Networker(object):
    """工厂函数类，生产不同的网络结构，也可以自己写。"""

    def __init__(self):
        pass

    @staticmethod
    def build_dense_net(x, hidden_units, trainable=True):
        """构建全连接网络。"""

        for n_uint in hidden_units[:-1]:
            x = tf.layers.dense(
                x, n_uint, activation=tf.nn.relu, trainable=trainable)

        fc = tf.layers.dense(
            x, hidden_units[-1], activation=None, trainable=trainable)

        return fc

    @staticmethod
    def build_distdqn_net(x, hidden_units, trainable=True):

        for n_uint in hidden_units[:-1]:
            x = tf.layers.dense(
                x, n_uint, activation=tf.nn.relu, trainable=trainable)

        fc = tf.layers.dense(
            x, hidden_units[-1], activation=None, trainable=trainable)
        return fc

    @staticmethod
    def build_cnn_net(x, n_action, trainable=True):
        """构建卷积网络。"""

        conv1 = tf.layers.conv2d(
            x, 32, 8, 4, activation=tf.nn.relu, trainable=trainable)
        conv2 = tf.layers.conv2d(
            conv1, 64, 4, 2, activation=tf.nn.relu, trainable=trainable)
        conv3 = tf.layers.conv2d(
            conv2, 64, 3, 1, activation=tf.nn.relu, trainable=trainable)
        flattened = tf.layers.flatten(conv3)
        fc1 = tf.layers.dense(
            flattened, 512, activation=tf.nn.relu, trainable=trainable)
        fc2 = tf.layers.dense(
            fc1, 2, activation=None, trainable=trainable)
        return fc2

    @staticmethod
    def build_pg_net(x, hidden_units, trainable=True):
        """构建policy-gradient网络。"""
        for n_uint in hidden_units[:-1]:
            x = tf.layers.dense(
                x, n_uint, activation=tf.nn.relu, trainable=trainable)

        mu = tf.layers.dense(
            x, hidden_units[-1], activation=None, trainable=trainable)

        return mu

    @staticmethod
    def build_gauss_net(x, hidden_units, trainable=True):
        init_size = 11
        init_log_var = -1
        for n_uint in hidden_units[:-1]:
            x = tf.layers.dense(
                x,
                n_uint,
                activation=tf.tanh,
                kernel_initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1/init_size)),
            )
            init_size = n_uint
        mu = tf.layers.dense(x, hidden_units[-1], activation=None,
                             kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1/init_size)))

        log_vars = tf.get_variable('logvars', [mu.shape.as_list()[1]], tf.float32,
                                   tf.constant_initializer(0.0)) + init_log_var

        return mu, log_vars

    @staticmethod
    def build_value_net(x, hidden_units, trainable=True):
        init_size = 11
        for n_uint in hidden_units[:-1]:
            x = tf.layers.dense(
                x,
                n_uint,
                activation=tf.tanh,
                kernel_initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1/init_size)
                )
            )
            init_size = n_uint
        val = tf.layers.dense(x, hidden_units[-1], activation=None,
                              kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(1/init_size)))

        return val

    @staticmethod
    def build_qvalue_net(s, a, trainable=True):
        x = tf.layers.dense(s, 64, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(
            stddev=np.sqrt(1/11)), trainable=trainable)
        x = tf.layers.dense(x, 64, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(
            stddev=np.sqrt(1/64)), trainable=trainable)

        y = tf.layers.dense(a, 64, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(
            stddev=np.sqrt(1/2)), trainable=trainable)

        z = tf.concat([x, y], axis=1)
        z = tf.layers.dense(z, 64, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(
            stddev=np.sqrt(1/128)), trainable=trainable)
        val = tf.layers.dense(z, 1, activation=None, kernel_initializer=tf.random_normal_initializer(
            stddev=np.sqrt(1/64)), trainable=trainable)

        return val

    @staticmethod
    def build_action_net(x, hidden_units, trainable=True):
        init_size = 11
        for n_unit in hidden_units[:-1]:
            x = tf.layers.dense(x, n_unit, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(
                stddev=np.sqrt(1/init_size)), trainable=trainable)
            init_size = n_unit

        act = tf.layers.dense(x, hidden_units[-1], activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(
            stddev=np.sqrt(1/hidden_units[-1])), trainable=trainable)
        return act
