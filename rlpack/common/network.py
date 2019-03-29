import tensorflow as tf


def mlp(x, hidden_sizes=[256, 256, 64, 4]):
    for hsize in hidden_sizes[:-1]:
        x = tf.layers.dense(x, hsize, activation=tf.nn.relu)
    return tf.layers.dense(x, hidden_sizes[-1])


def cnn1d(x, cnn1d_hidden_sizes=[(32, 8, 4), (64, 4, 2)], mlp_hidden_sizes=[64, 4]):
    for n_filter, stride, pad in cnn1d_hidden_sizes:
        x = tf.layers.conv1d(x, n_filter, stride, pad, activation=tf.nn.relu)
    x = tf.contrib.layers.flatten(x)
    for hsize in mlp_hidden_sizes[:-1]:
        x = tf.layers.dense(x, hsize, activation=tf.nn.relu)
    return tf.layers.dense(x, mlp_hidden_sizes[-1])


def cnn2d(x, cnn2d_hidden_sizes=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], mlp_hidden_sizes=[512, 4]):
    for n_filter, stride, pad in cnn2d_hidden_sizes:
        x = tf.layers.conv2d(x, n_filter, stride, pad, activation=tf.nn.relu)
    x = tf.contrib.layers.flatten(x)
    for hsize in mlp_hidden_sizes[:-1]:
        x = tf.layers.dense(x, hsize, activation=tf.nn.relu)
    return tf.layers.dense(x, mlp_hidden_sizes[-1])
