import math
from collections import deque, defaultdict
import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class A2C(Base):
    """Advantage Actor Critic."""

    def __init__(self,
                 obs_fn=None,
                 policy_fn=None,
                 value_fn=None,
                 dim_act=None,
                 rnd=1,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=1000,
                 trajectory_length=2048,
                 gae=0.95,
                 train_epoch=10,
                 lr=1e-4,
                 policy_lr=1e-4,
                 value_lr=1e-3,
                 batch_size=64
                 ):

        self._obs_fn = obs_fn
        self._policy_fn = policy_fn
        self._value_fn = value_fn
        self._dim_act = dim_act
        self._discount = discount
        self._gae = gae
        self._lr = lr
        self._policy_lr = policy_lr
        self._value_lr = value_lr

        self._train_epoch = train_epoch
        self._log_freq = log_freq
        self._save_model_freq = save_model_freq
        self._batch_size = batch_size

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # Build inputs.
        # self._observation = tf.placeholder(tf.float32, [None, *self._dim_obs], "observation")

        self._observation = self._obs_fn()
        self._action = tf.placeholder(tf.int32, [None], "action")
        self._target_state_value = tf.placeholder(tf.float32, [None], "target_state_value")
        self._advantage = tf.placeholder(tf.float32, [None], "advantage")

        with tf.variable_scope("policy"):
            self._logit_a = self._policy_fn(self._observation)
            # x = self._dense(self._observation)
            # self._logit_a = tf.layers.dense(x, self._dim_act)

        with tf.variable_scope("value"):
            # x = self._dense(self._observation)
            # self._state_value = tf.squeeze(tf.layers.dense(x, 1))
            self._state_value = self._value_fn(self._observation)

    # def _dense(self, obs):
    #     x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #     return x

    def _build_algorithm(self):
        policy_optimizer = tf.train.AdagradOptimizer(0.01)  # (self._lr, epsilon=1e-10)
        value_optimizer = tf.train.AdagradOptimizer(0.01)  # (self._lr, epsilon=1e-10)
        policy_variables = tf.trainable_variables("policy")
        value_variables = tf.trainable_variables("value")

        nsample = tf.shape(self._observation)[0]
        actind = tf.stack([tf.range(nsample), self._action], axis=1)
        max_logit_a = tf.reduce_max(self._logit_a, axis=1, keepdims=True)
        logit = self._logit_a - max_logit_a
        log_p_act = logit - tf.reduce_logsumexp(logit, axis=1, keepdims=True)
        log_p_act = tf.gather_nd(log_p_act, actind)
        # actind = tf.stack([tf.range(nsample), self._action], axis=1)
        # log_p_act = tf.log(tf.gather_nd(self._p_act, actind), 1e-20, 1e2)
        assert_shape(log_p_act, [None])

        advantage = tf.stop_gradient(self._target_state_value - self._state_value)
        policy_loss = -tf.reduce_mean(log_p_act * advantage)
        value_loss = tf.reduce_mean((self._target_state_value - self._state_value)**2)

        self._policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy_variables)
        self._value_train_op = value_optimizer.minimize(value_loss, var_list=value_variables)

        self._log_op = {"policy_loss": policy_loss, "value_loss": value_loss}

    def update(self, databatch):

        s_batch, a_batch, tsv_batch, adv_batch = self._parse_databatch(databatch)

        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

        self._log = defaultdict(deque)
        for _ in range(self._train_epoch):
            n_sample = s_batch.shape[0]
            index = np.arange(n_sample)
            np.random.shuffle(index)

            for i in range(math.ceil(n_sample / self._batch_size)):
                span_index = slice(i*self._batch_size, min((i+1)*self._batch_size, n_sample))
                span_index = index[span_index]

                _, _, log = self.sess.run([self._policy_train_op, self._value_train_op, self._log_op],
                                          feed_dict={
                    self._observation: s_batch[span_index, ...],
                    self._action: a_batch[span_index],
                    self._target_state_value: tsv_batch[span_index],
                    self._advantage: adv_batch[span_index]
                })
                self._collect_log(log)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

        if global_step % self._log_freq == 0:
            self.sw.add_scalars("a2c", self._average_log(), global_step=global_step)

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
        n = len(trajectory)
        tsv_batch = np.zeros(n, dtype=np.float32)
        adv_batch = np.zeros(n, dtype=np.float32)

        s_batch = np.array([t[0] for t in trajectory], dtype=np.float32)
        a_batch = np.array([t[1] for t in trajectory], dtype=np.int32)
        sv_batch = self.sess.run(self._state_value, feed_dict={self._observation: s_batch})

        for i, (_, _, r) in enumerate(reversed(trajectory)):
            i = n-1-i
            state_value = sv_batch[i]
            if i == n-1:
                adv_batch[i] = r - state_value
                tsv_batch[i] = r
                last_state_value = state_value
                continue
            delta_value = r + self._discount * last_state_value - state_value
            adv_batch[i] = delta_value + self._discount * self._gae * adv_batch[i+1]
            tsv_batch[i] = state_value + adv_batch[i]
            last_state_value = state_value

        return s_batch, a_batch, tsv_batch, adv_batch

    def _collect_log(self, log):
        for k, v in log.items():
            self._log[k].append(v)

    def _average_log(self):
        return {k: np.mean(self._log[k]) for k in self._log.keys()}

    def get_action(self, ob):
        """Return actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n, action_dimension).
        """
        logit = self.sess.run(self._logit_a, feed_dict={self._observation: ob})
        logit = logit - np.max(logit, axis=1, keepdims=True)
        p_act = np.exp(logit) / np.sum(np.exp(logit), axis=1, keepdims=True)
        # print("p_act:", p_act)
        nsample, nact = p_act.shape
        return [np.random.choice(nact, p=p_act[i, :]) for i in range(nsample)]
