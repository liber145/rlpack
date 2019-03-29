"""Policy Gradient.
Return通过倒序discount累加计算所得。
训练epoch数目只能是1，因为第二轮训练时，state distribution改变了。
因此，所有的数据一起算梯度。或者分批次算，将梯度累加起来（这种方法实现麻烦）。
"""


from collections import deque
import math
import numpy as np
import tensorflow as tf

from .base import Base


class PG(Base):
    def __init__(self,
                 obs_fn=None,
                 policy_fn=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 batch_size=64,
                 lr=1e-4,
                 train_epoch=1,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=10,
                 ):

        self._obs_fn = obs_fn
        self._policy_fn = policy_fn
        self._dim_act = dim_act
        self._discount = discount
        self._train_epoch = train_epoch
        self._batch_size = batch_size
        self._lr = lr

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        # self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self._observation = self._obs_fn()
        self._action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self._return = tf.placeholder(shape=[None], dtype=tf.float32, name="return")

        with tf.variable_scope("main"):
            # self._p_act = self._dense(self._observation)
            self._p_act = self._policy_fn(self._observation)

    # def _dense(self, obs):
    #     x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #     return tf.layers.dense(x, self._dim_act, activation=tf.nn.softmax)

    def _build_algorithm(self):
        optimizer = tf.train.AdamOptimizer(1e-4)

        batch_size = tf.shape(self._action)[0]
        log_p_act = tf.log(tf.gather_nd(self._p_act, tf.stack([tf.range(batch_size), self._action], axis=1)))

        target = - tf.reduce_mean(log_p_act * self._return)
        self._train_op = optimizer.minimize(target)

    def get_action(self, ob):
        p_act = self.sess.run(self._p_act, feed_dict={self._observation: ob})
        n_sample, n_act = p_act.shape
        return [np.random.choice(n_act, p=p_act[i, :]) for i in range(n_sample)]

    def update(self, databatch):
        s_minibatch, a_minibatch, allr_minibatch = self._parse_minibatch(databatch)

        for _ in range(self._train_epoch):
            self.sess.run(self._train_op,
                          feed_dict={self._observation: s_minibatch,
                                     self._action: a_minibatch,
                                     self._return: allr_minibatch
                                     })

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

    def _parse_minibatch(self, minibatch):
        s_list = deque()
        a_list = deque()
        allr_list = deque()
        for trajectory in minibatch:
            s_batch, a_batch, allr_batch = self._parse_trajectory(trajectory)
            s_list.append(s_batch)
            a_list.append(a_batch)
            allr_list.append(allr_batch)
        return np.concatenate(s_list), np.concatenate(a_list), np.concatenate(allr_list)

    def _parse_trajectory(self, trajectory):
        """trajectory由一系列(s,a,r)构成。最后一组操作之后游戏结束。
        """
        n = len(trajectory)
        s_batch = np.array([t[0] for t in trajectory], dtype=np.float32)
        a_batch = np.array([t[1] for t in trajectory], dtype=np.int32)
        allr_batch = np.zeros(n, dtype=np.float32)
        tsum = 0
        for i, (s, a, r) in enumerate(reversed(trajectory)):
            tsum = self._discount * tsum + r
            allr_batch[i] = tsum
        return s_batch, a_batch, allr_batch
