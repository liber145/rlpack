"""Policy Gradient.
1. Return通过倒序discount累加计算所得。最后一个状态的value为0.
2. 训练epoch数目只能是1，因为第二轮训练时，state distribution改变了。
"""


import numpy as np
import tensorflow as tf

from .base import Base


class PG(Base):
    def __init__(self,
                 rnd=0,
                 dim_obs=None, dim_act=None,
                 policy_fn=None,
                 discount=0.99,
                 train_epoch=1, policy_lr=1e-3,
                 save_path="./log", log_freq=10, save_model_freq=1000,
                 ):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._policy_fn = policy_fn

        self._discount = discount
        self._train_epoch = train_epoch
        self._policy_lr = policy_lr

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        self._obs = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self._act = tf.placeholder(shape=[None, self._dim_act], dtype=tf.float32, name="action")
        self._ret = tf.placeholder(shape=[None], dtype=tf.float32, name="return")
        self.all_phs = [self._obs, self._act, self._ret]

        with tf.variable_scope("main"):
            self.pi, self.logp = self._policy_fn(self._obs, self._act)

    def _build_algorithm(self):
        loss = - tf.reduce_mean(self.logp * self._ret)
        self.train_policy_op = tf.train.AdamOptimizer(self._policy_lr).minimize(loss)

    def get_action(self, obs):
        act = self.sess.run(self.pi, feed_dict={self._obs: obs})
        return act

    def update(self, databatch):
        states, actions, returns = self._parse_databatch(*databatch)
        inputs = {k: v for k, v in zip(self.all_phs, [states, actions, returns])}

        for _ in range(self._train_epoch):
            self.sess.run(self.train_policy_op, feed_dict=inputs)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

    def _parse_databatch(self, states, actions, rewards, dones, earlystops, nextstates):

        batch_size = len(dones)
        returns = np.zeros(batch_size)

        for i in reversed(range(batch_size)):

            if dones[i]:
                prev_return = 0
            elif earlystops[i]:
                raise Exception("PG算法采用MCMC方法计算状态值，最后一个状态必须是终止状态。")

            returns[i] = rewards[i] + self._discount * prev_return * (1 - dones[i])
            prev_return = returns[i]

        return [states, actions, returns]
