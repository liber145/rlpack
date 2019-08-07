import numpy as np
import tensorflow as tf

import scipy
import tensorflow as tf

from .base import Base


class DoubleDQN(Base):
    def __init__(self,
                 rnd=0,
                 dim_obs=None, n_act=None,
                 value_fn=None,
                 discount=0.99,
                 train_epoch=1, value_lr=1e-3,
                 update_target_rate=0.8, max_grad_norm=40,
                 save_path="./log", log_freq=10, save_model_freq=1000,
                 ):
        self._dim_obs = dim_obs
        self._n_act = n_act
        self._value_fn = value_fn

        self._discount = discount
        self._train_epoch = train_epoch
        self._value_lr = value_lr
        self._update_target_rate = update_target_rate
        self._max_grad_norm = max_grad_norm

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)
        self.sess.run(self.init_target_op)

    def _build_network(self):
        """Build networks for algorithm."""
        self._obs = tf.placeholder(tf.float32, shape=[None, *self._dim_obs], name="observation")
        self._act = tf.placeholder(dtype=tf.int32, shape=[None], name="action")
        self._reward = tf.placeholder(dtype=tf.float32, shape=[None], name="reward")
        self._done = tf.placeholder(dtype=tf.float32, shape=[None], name="done")
        self._obs2 = tf.placeholder(dtype=tf.float32, shape=[None, *self._dim_obs], name="next_observation")
        self.all_phs = [self._obs, self._act, self._reward, self._done, self._obs2]

        with tf.variable_scope("main/q"):
            self.q = self._value_fn(self._obs)

        with tf.variable_scope("main/q", reuse=True):
            self.q2 = tf.stop_gradient(self._value_fn(self._obs2))

        with tf.variable_scope("target/q"):
            self.q_targ = self._value_fn(self._obs2)

    def _build_algorithm(self):
        trainable_variables = tf.trainable_variables("main/q")

        # Compute state-action value.
        batch_size = tf.shape(self._obs)[0]
        action_index = tf.stack([tf.range(batch_size), self._act], axis=1)
        action_q = tf.gather_nd(self.q, action_index)

        # Compute back up.
        arg_act = tf.argmax(self.q2, axis=1, output_type=tf.int32)
        arg_act_index = tf.stack([tf.range(batch_size), arg_act], axis=1)
        q_backup = self._reward + self._discount * (1 - self._done) * tf.gather_nd(self.q_targ, arg_act_index)

        loss = tf.reduce_mean(tf.squared_difference(q_backup, action_q))
        self._train_op = tf.train.AdamOptimizer(self._value_lr).minimize(loss, var_list=trainable_variables)

        # Update target network.

        def _update_target(net1, net2, rho=0):
            params1 = tf.trainable_variables(net1)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.trainable_variables(net2)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param1.assign(rho*param1 + (1-rho)*param2))
            return update_ops

        self.update_target_op = _update_target("target", "main", rho=self._update_target_rate)
        self.init_target_op = _update_target("target", "main")

    def get_action(self, obs):
        q = self.sess.run(self.q, feed_dict={self._obs: obs})
        return np.argmax(q, axis=1)

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch
        inputs = {k: v for k, v in zip(self.all_phs, [s_batch, a_batch, r_batch, d_batch, next_s_batch])}

        for _ in range(self._train_epoch):
            self.sess.run(self._train_op, inputs)

        self.sess.run(self.update_target_op)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()
