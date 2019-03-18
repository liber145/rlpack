import math
import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class TD3(Base):
    def __init__(self,
                 rnd=1,
                 n_env=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=50,
                 log_freq=1,
                 policy_lr=1e-3,
                 value_lr=1e-3,
                 train_epoch=1,
                 policy_delay=2,
                 target_update_rate=0.995,
                 noise_std=0.2,
                 explore_noise_std=0.1
                 ):
        """Implementation of PPO.

        Parameters:
            config: a dictionary for training config.

        Returns:
            None
        """

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount

        self._save_model_freq = save_model_freq

        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._train_epoch = train_epoch
        self._log_freq = log_freq

        self._policy_decay = policy_delay
        self._target_update_ratio = target_update_rate
        self.noise_std = noise_std
        self.explore_noise_std = explore_noise_std

        super().__init__(save_path=save_path, rnd=rnd)

        self.sess.run(self._init_target_policy_op)
        self.sess.run(self._init_target_value_op)

    def _build_network(self):
        """Build networks for algorithm."""
        self._observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")
        self._action = tf.placeholder(tf.int32, [None], name="action")
        self._reward = tf.placeholder(tf.float32, [None], name="reward")
        self._done = tf.placeholder(tf.float32, [None], name="done")
        self._next_observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="next_observation")

        # with tf.variable_scope("policy_net"):
        #     x = tf.layers.dense(self._observation, 400, activation=tf.nn.relu, trainable=True)
        #     x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
        #     self.act = tf.layers.dense(x, self._dim_act, activation=tf.nn.tanh, trainable=True)

        # with tf.variable_scope("target_policy_net"):
        #     x = tf.layers.dense(self._observation, 400, activation=tf.nn.relu, trainable=False)
        #     x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=False)
        #     self.target_act = tf.layers.dense(x, self._dim_act, activation=tf.nn.tanh, trainable=False)

        # with tf.variable_scope("value_net"):
        #     x = tf.concat([self._observation, self._action], axis=1)
        #     x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=True)
        #     x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
        #     self.qval_1 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=True))

        #     x = tf.concat([self._observation, self._action], axis=1)
        #     x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=True)
        #     x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
        #     self.qval_2 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=True))

        # with tf.variable_scope("value_net", reuse=True):
        #     x = tf.concat([self._observation, self.act], axis=1)
        #     x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=True)
        #     x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=True)
        #     self.qval_act = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=True))

        # with tf.variable_scope("target_value_net"):
        #     x = tf.concat([self._observation, self._action], axis=1)
        #     x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=False)
        #     x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=False)
        #     self._target_qval_1 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=False))

        #     x = tf.concat([self._observation, self._action], axis=1)
        #     x = tf.layers.dense(x, 400, activation=tf.nn.relu, trainable=False)
        #     x = tf.layers.dense(x, 300, activation=tf.nn.relu, trainable=False)
        #     self._target_qval_2 = tf.squeeze(tf.layers.dense(x, 1, activation=None, trainable=False))

        with tf.variable_scope("main/policy"):
            x = self._dense(self._observation)
            self._p_act = tf.layers.dense(x, self._dim_act, activation=tf.nn.softmax)

        with tf.variable_scope("main/value_1"):
            x = self._dense(self._observation)
            self._qval_1 = tf.layers.dense(x, self._dim_act)

        with tf.variable_scope("main/value_2"):
            x = self._dense(self._observation)
            self._qval_2 = tf.layers.dense(x, self._dim_act)

        with tf.variable_scope("target/policy"):
            x = self._dense(self._observation)
            self._target_p_act = tf.layers.dense(x, self._dim_act, activation=tf.nn.softmax)

        with tf.variable_scope("target/value_1"):
            x = self._dense(self._observation)
            self._target_qval_1 = tf.layers.dense(x, self._dim_act)

        with tf.variable_scope("target/value_2"):
            x = self._dense(self._observation)
            self._target_qval_2 = tf.layers.dense(x, self._dim_act)

    def _dense(self, obs):
        x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 128, activation=tf.nn.relu)
        x = tf.layers.dense(x, 64, activation=tf.nn.relu)
        return x

    # Update target network.
    def _update_target(self, net1, net2, rho=0):
        params1 = tf.trainable_variables(net1)
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.trainable_variables(net2)
        params2 = sorted(params2, key=lambda v: v.name)
        assert len(params1) == len(params2)
        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param1.assign(rho*param1 + (1-rho)*param2))
        return update_ops

    def _build_algorithm(self):
        policy_optimizer = tf.train.AdamOptimizer(self._policy_lr)
        value_optimizer = tf.train.AdamOptimizer(self._value_lr)
        policy_variables = tf.trainable_variables("main/policy")
        value_variables = tf.trainable_variables("main/value")

        qval_a = tf.reduce_sum(self._p_act * self._qval_1, axis=1)
        policy_loss = -tf.reduce_mean(qval_a)

        nsample = tf.shape(self._observation)[0]
        dist = tf.distributions.Categorical(probs=self._target_p_act)
        target_act = dist.sample()
        actind = tf.stack([tf.range(nsample), target_act], axis=1)
        target_qval_a_1 = tf.gather_nd(self._target_qval_1, actind)
        target_qval_a_2 = tf.gather_nd(self._target_qval_2, actind)
        target_qval_a = tf.minimum(target_qval_a_1, target_qval_a_2)
        qbackup = tf.stop_gradient(self._reward + self._discount * (1-self._done) * target_qval_a)

        actind = tf.stack([tf.range(nsample), self._action], axis=1)
        qval_a_1 = tf.gather_nd(self._qval_1, actind)
        qval_a_2 = tf.gather_nd(self._qval_2, actind)
        value_loss = tf.reduce_mean(tf.squared_difference(qval_a_1, qbackup)) + tf.reduce_mean(tf.squared_difference(qval_a_2, qbackup))

        self._policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy_variables)
        self._value_train_op = value_optimizer.minimize(value_loss, var_list=value_variables)

        self._update_target_policy_op = self._update_target("target/policy", "main/policy", self._target_update_ratio)
        self._update_target_value_op = self._update_target("target/value", "main/value", self._target_update_ratio)

        self._init_target_policy_op = self._update_target("target/policy", "main/policy")
        self._init_target_value_op = self._update_target("target/value", "main/value")

        self._log_op = {"policy_loss": policy_loss, "value_loss": value_loss}

    def get_action(self, obs):
        p_act = self.sess.run(self._p_act, feed_dict={self._observation: obs})
        nsample, nact = p_act.shape
        return [np.random.choice(nact, p=p_act[i, :]) for i in range(nsample)]

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch

        for _ in range(self._train_epoch):
            self.sess.run([self._policy_train_op, self._value_train_op],
                          feed_dict={
                self._observation: s_batch,
                self._action: a_batch,
                self._reward: r_batch,
                self._done: d_batch,
                self._next_observation: next_s_batch
            })

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._policy_decay == 0:
            self.sess.run([self._update_target_value_op, self._update_target_policy_op])

        if global_step % self._save_model_freq == 0:
            self.save_model()

        if global_step % self._log_freq == 0:
            log = self.sess.run(self._log_op,
                                feed_dict={
                                    self._observation: s_batch,
                                    self._action: a_batch,
                                    self._reward: r_batch,
                                    self._done: d_batch,
                                    self._next_observation: next_s_batch
                                })
            self.sw.add_scalars("td3", log, global_step=global_step)
