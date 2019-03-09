"""Policy Gradient.
Return通过倒序discount累加计算所得。
训练epoch数目只能是1，因为第二轮训练时，state distribution改变了。
因此，所有的数据一起算梯度。或者分批次算，将梯度累加起来（这种方法实现麻烦）。
"""


from collections import deque
import math
import numpy as np
import tensorflow as tf


class PG(object):
    def __init__(self,
                 rnd=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 batch_size=64,
                 train_epoch=1):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount
        self._train_epoch = train_epoch
        self._batch_size = batch_size

        tf.reset_default_graph()

        self._build_network()
        self._build_algorithm()

        conf = tf.ConfigProto(allow_soft_placement=True)
        conf.gpu_options.allow_growth = True  # pylint: disable=E1101
        self.sess = tf.Session(config=conf)

        # ------------------------ Initialize tensorflow variables.  ------------------------
        self.sess.run(tf.global_variables_initializer())

    def _build_network(self):

        self._observation = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self._action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self._return = tf.placeholder(shape=[None], dtype=tf.float32, name="return")

        with tf.variable_scope("main"):
            self._p_act = self._net(self._observation)

    def _net(self, obs):
        x = tf.layers.conv2d(self._observation, 8, (1, 3), 1, activation=tf.nn.relu)
        x = tf.contrib.layers.flatten(x)
        return tf.layers.dense(x, self._dim_act, activation=tf.nn.softmax)

    def _build_algorithm(self):
        optimizer = tf.train.AdamOptimizer(1e-4)

        batch_size = tf.shape(self._action)[0]
        log_p_act = tf.log(tf.gather_nd(self._p_act, tf.stack([tf.range(batch_size), self._action], axis=1)))

        target = - tf.reduce_mean(log_p_act * self._return)
        self._train_op = optimizer.minimize(target)

    def update(self, minibatch):
        s_minibatch, a_minibatch, allr_minibatch = self._parse_minibatch(minibatch)

        # n_sample = s_minibatch.shape[0]
        # index = np.arange(n_sample)
        # np.random.shuffle(index)

        # for i in range(math.ceil(n_sample / self._batch_size)):
        #     print(f" {i}th training...")
        #     span_index = slice(i * self._batch_size, min((i+1)*self._batch_size, n_sample))
        #     span_index = index[span_index]

        #     self.sess.run(self._train_op,
        #                   feed_dict={self._observation: s_minibatch[span_index, ...],
        #                              self._action: a_minibatch[span_index],
        #                              self._return: allr_minibatch[span_index]})

        self.sess.run(self._train_op,
                      feed_dict={self._observation: s_minibatch,
                                 self._action: a_minibatch,
                                 self._return: allr_minibatch})

    def get_action(self, ob):
        ob = ob[np.newaxis, :]
        p_act = self.sess.run(self._p_act, feed_dict={self._observation: ob})
        return np.random.choice(len(p_act), 1, p=p_act)[0]

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
        s_batch = np.zeros((n, *self._dim_obs))
        a_batch = np.zeros(n, dtype=np.int32)
        allr_batch = np.zeros(n)
        tsum = 0
        for i, (s, a, r) in enumerate(reversed(trajectory)):
            tsum = self._discount * tsum + r
            allr_batch[i] = tsum
            s_batch[i, ...] = s
            a_batch[i] = a
        return s_batch, a_batch, allr_batch


if __name__ == "__main__":
    import gzip
    import pickle

    data = pickle.load(gzip.open('/data/mahjong/rl/one_step/01a85ed3-2cb6-4c7c-a9d8-dbf27478bbdc', 'rb'))

    agent = PG(dim_obs=(4, 9, 106), dim_act=34)
    agent.update(data)
