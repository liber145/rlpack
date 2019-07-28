"""
Proximal Policy Optimization.
目标loss由三部分组成：1. clipped policy loss；2. value loss；3. entropy。
policy loss需要计算当前policy和old policy在当前state上的ratio。
需要注意的是，state分布依赖于old policy。因此，计算中的old policy是一样的。
"""


import math
from collections import deque, defaultdict
import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class PPO(Base):
    def __init__(self,
                 obs_fn=None,
                 policy_fn=None,
                 value_fn=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 gae=0.95,
                 vf_coef=1.0,
                 entropy_coef=0.01,
                 max_grad_norm=40,
                 train_epoch=3,
                 batch_size=128,
                 lr_schedule=lambda x: 2.5e-4,
                 clip_schedule=lambda x: 0.1 * (10000-x) / 10000,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=10
                 ):
        """PPO settings.

        Keyword Arguments:
            dim_obs {np.ndarray} -- observation shape (default: {None})
            dim_act {np.ndarray} -- action shape (default: {None})
            rnd {int} -- random seed (default: {1})
            discount {float} -- discount factor (default: {0.99})
            gae {float} -- generalized advantage estimation (default: {0.95})
            vf_coef {float} -- value fuction coefficient (default: {1.0})
            entropy_coef {float} -- entropy coefficient (default: {0.01})
            max_grad_norm {float} -- max gradient norm (default: {0.5})
            train_epoch {int} -- train epoch (default: {5})
            batch_size {int} -- batch size (default: {64})
            lr_schedule {lambda} -- learning rate schedule (default: {lambdax:2.5e-4})
            clip_schedule {lamdba} -- epsilon clip schedule (default: {lambdax:0.1})
            save_path {str} -- save path (default: {"./log"})
            save_model_freq {int} -- save model frequency (default: {1000})
        """

        self._obs_fn = obs_fn
        self._policy_fn = policy_fn
        self._value_fn = value_fn
        self._dim_act = dim_act

        self._discount = discount
        self._gae = gae
        self._entropy_coef = entropy_coef
        self._vf_coef = vf_coef
        self._lr_schedule = lr_schedule
        self._clip_schedule = clip_schedule

        self._max_grad_norm = max_grad_norm
        self._train_epoch = train_epoch
        self._batch_size = batch_size

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq
        self._log = defaultdict(deque)

        self._gradients = None

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        # self._observation = tf.placeholder(tf.uint8, [None, *self._dim_obs], name="observation")
        # self._observation = tf.to_float(self._observation) / 255.0
        # self._observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")
        self._observation = self._obs_fn()

        # x = self._dense(self._observation)
        # self._logit_p_act = tf.layers.dense(x, self._dim_act, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01))
        # self._state_value = tf.squeeze(tf.layers.dense(x, 1, kernel_initializer=tf.truncated_normal_initializer()))

        self._logit_p_act = self._policy_fn(self._observation)
        self._state_value = self._value_fn(self._observation)

        # self._logit_p_act = tf.layers.dense(x, self._dim_act, activation=None, kernel_initializer=tf.truncated_normal_initializer(0.0, 0.01))
        # self._state_value = tf.squeeze(tf.layers.dense(x, 1, activation=None, kernel_initializer=tf.truncated_normal_initializer()))

    # def _conv(self, x):
    #     x = tf.layers.conv2d(x, 32, 8, 4, activation=tf.nn.relu)
    #     x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
    #     x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
    #     x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
    #     x = tf.layers.dense(x, 512, activation=tf.nn.relu)

    # def _dense(self, x):
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #     return x

    def _build_algorithm(self):
        self._clip_ratio = tf.placeholder(tf.float32)
        self._lr = tf.placeholder(tf.float32)
        self._optimizer = tf.train.AdamOptimizer(self._lr, epsilon=1e-8)

        self._old_logit_p_act = tf.placeholder(tf.float32, [None, self._dim_act])
        self._action = tf.placeholder(tf.int32, [None], name="action")
        self._advantage = tf.placeholder(tf.float32, [None], name="advantage")
        self._target_state_value = tf.placeholder(tf.float32, [None], name="target_state_value")

        # Get selected action index.
        batch_size = tf.shape(self._observation)[0]
        selected_action_index = tf.stack([tf.range(batch_size), self._action], axis=1)

        # Compute entropy of the action probability.
        log_prob_1 = tf.nn.log_softmax(self._logit_p_act)
        log_prob_2 = tf.stop_gradient(tf.nn.log_softmax(self._old_logit_p_act))
        assert_shape(log_prob_1, [None, self._dim_act])
        assert_shape(log_prob_2, [None, self._dim_act])

        prob_1 = tf.nn.softmax(self._logit_p_act)
        assert_shape(prob_1, [None, self._dim_act])

        entropy = - tf.reduce_mean(tf.reduce_sum(log_prob_1 * prob_1, axis=1))   # entropy = - \sum_i p_i \log(p_i)

        # Compute ratio of the action probability.
        logit_act1 = tf.gather_nd(log_prob_1, selected_action_index)
        logit_act2 = tf.gather_nd(log_prob_2, selected_action_index)
        assert_shape(logit_act1, [None])
        assert_shape(logit_act2, [None])

        ratio = tf.exp(logit_act1 - logit_act2)

        # Get surrogate object.
        surrogate_1 = ratio * self._advantage
        surrogate_2 = tf.clip_by_value(ratio, 1.0 - self._clip_ratio, 1.0 + self._clip_ratio) * self._advantage
        assert_shape(ratio, [None])
        assert_shape(surrogate_1, [None])
        # assert_shape(surrogate_2, [None])
        surrogate = -tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))

        # Compute critic loss.
        vf = tf.reduce_mean(tf.squared_difference(self._state_value, self._target_state_value))

        # Compute gradients.
        total_loss = surrogate + self._vf_coef * vf - self._entropy_coef * entropy
        grads = tf.gradients(total_loss, tf.trainable_variables())

        print("grads:", grads)
        print("shape:")
        print([x.shape for x in grads])

        grad_norm = tf.global_norm(grads)

        # Clip gradients.
        clipped_grads, _ = tf.clip_by_global_norm(grads, self._max_grad_norm)
        print("clipped_grads:", clipped_grads)
        print("shape:")
        print([x.shape for x in clipped_grads])
        input()
        self._train_op = self._optimizer.apply_gradients(zip(clipped_grads, tf.trainable_variables()))

        self._log_op = {"entropy": entropy, "mean_ratio": tf.reduce_mean(ratio), "total_loss": total_loss, "value_loss": vf, "policy_loss": surrogate, "grad_norm": grad_norm}

    def get_action(self, obs) -> np.ndarray:
        n_inference = obs.shape[0]
        logit = self.sess.run(self._logit_p_act, feed_dict={self._observation: obs})
        logit = logit - np.max(logit, axis=1, keepdims=True)
        prob = np.exp(logit) / np.sum(np.exp(logit), axis=1, keepdims=True)
        action = [np.random.choice(self._dim_act, p=prob[i, :]) for i in range(n_inference)]
        return np.array(action)

    def update(self, databatch):
        """
        Arguments:
            databatch {list of list} -- A list of trajectories, each of which is also a list filled with (s,a,r).
        """

        s_batch, a_batch, tsv_batch, adv_batch = self._parse_databatch(databatch)

        # Normalize Advantage.
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
        old_logit_p_act = self.sess.run(self._logit_p_act, feed_dict={self._observation: s_batch})

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        # Train network.
        self._log = defaultdict(deque)
        for _ in range(self._train_epoch):

            n_sample = s_batch.shape[0]
            index = np.arange(n_sample)
            np.random.shuffle(index)

            for i in range(math.ceil(n_sample / self._batch_size)):
                span_index = slice(i*self._batch_size, min((i+1)*self._batch_size, n_sample))
                span_index = index[span_index]

                _, log = self.sess.run([self._train_op, self._log_op],
                                       feed_dict={self._observation: s_batch[span_index, ...],
                                                  self._action: a_batch[span_index],
                                                  self._advantage: adv_batch[span_index],
                                                  self._target_state_value: tsv_batch[span_index],
                                                  self._old_logit_p_act: old_logit_p_act[span_index],
                                                  self._lr: self._lr_schedule(global_step),
                                                  self._clip_ratio: self._clip_schedule(global_step)})

                self._collect_log(log)

        if global_step % self._save_model_freq == 0:
            self.save_model()

        if global_step % self._log_freq == 0:
            self.sw.add_scalars("ppo", self._average_log(), global_step=global_step)

    def _parse_databatch(self, databatch):
        s_list = deque()
        a_list = deque()
        tsv_list = deque()
        adv_list = deque()
        for trajectory in databatch:
            s_batch, a_batch, tsv_batch, adv_batch = self._parse_trajectory(trajectory)
            s_list.append(s_batch)
            a_list.append(a_batch)
            tsv_list.append(tsv_batch)
            adv_list.append(adv_batch)
        return np.concatenate(s_list), np.concatenate(a_list), np.concatenate(tsv_list), np.concatenate(adv_list)

    def _parse_trajectory(self, trajectory):
        """trajectory由一系列(s,a,r)构成。最后一组操作之后游戏结束。
        """
        n = len(trajectory)
        target_sv_batch = np.zeros(n, dtype=np.float32)
        adv_batch = np.zeros(n, dtype=np.float32)

        a_batch = np.array([t[1] for t in trajectory], dtype=np.int32)
        s_batch = np.array([t[0] for t in trajectory], dtype=np.float32)
        sv_batch = self.sess.run(self._state_value, feed_dict={self._observation: s_batch})

        for i, (_, _, r) in enumerate(reversed(trajectory)):   # 注意这里是倒序操作。
            i = n-1-i
            state_value = sv_batch[i]
            if i == n-1:
                adv_batch[i] = r - state_value
                target_sv_batch[i] = r
                last_state_value = state_value
                continue
            delta_value = r + self._discount * last_state_value - state_value
            adv_batch[i] = delta_value + self._discount * self._gae * adv_batch[i+1]
            target_sv_batch[i] = state_value + adv_batch[i]
            last_state_value = state_value

        return s_batch, a_batch, target_sv_batch, adv_batch

    def _collect_log(self, log):
        for k, v in log.items():
            self._log[k].append(v)

    def _average_log(self):
        return {k: np.mean(self._log[k]) for k in self._log.keys()}
