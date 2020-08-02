
#!/usr/bin/python3

from rlpack.algos.gail import Discriminator, PPO, GAIL_wrapper
import numpy as np
import tensorflow as tf
import gym
import argparse


class Policy_net:
    def __init__(self, name, ob_space, act_space):
        """
        :param name: string
        :param env: gym env
        """

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=(None, ob_space), name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', help='log directory', default='log/')
parser.add_argument('--gen_save', help='save directory', default='trained_models/')
parser.add_argument('--disc_save', help='save directory', default='trained_models/')
parser.add_argument('--iters', default=int(1e4), type=int)
args = parser.parse_args()

obs_dims = 4
n_actions = 2

env = gym.make("CartPole-v0")

agent = PPO(args.gen_save, Policy_net, obs_dims, 2)
D = Discriminator(args.disc_save, obs_dims, n_actions)
trainer = GAIL_wrapper(agent, D, env, args.logdir)

trainer.train(args.iters, obs_path='data/trajectory/observations.csv', act_path='data/trajectory/actions.csv')

