"""
alpha表示reward放缩了多少倍。
"""


import numpy as np
import tensorflow as tf

from ..common.utils import assert_shape
from .base import Base


class SAC(Base):
    def __init__(self,
                 rnd=1,
                 obs_fn=None,
                 policy_fn=None,
                 qval_fn=None,
                 sval_fn=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 log_freq=1000,
                 policy_lr=1e-3,
                 value_lr=1e-3,
                 train_epoch=1,
                 alpha=0.2,
                 update_target_ratio=0.995
                 ):
        self.alpha = alpha
        self._obs_fn = obs_fn
        self._policy_fn = policy_fn
        self._qval_fn = qval_fn
        self._sval_fn = sval_fn
        self._dim_act = dim_act
        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._update_target_ratio = update_target_ratio
        self._train_epoch = train_epoch

        self._discount = discount
        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0
        self.EPS = 1e-8

        super().__init__(save_path=save_path, rnd=rnd)
        self.sess.run(self._init_target_op)

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def clip_but_pass_gradient(self, x, l=-1, u=1):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient(clip_up * (u - x) + clip_low * (l - x))

    def _build_network(self):
        # self._observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")
        self._observation = self._obs_fn()
        self._action = tf.placeholder(tf.int32, [None], name="action")
        self._reward = tf.placeholder(tf.float32, [None], name="reward")
        self._done = tf.placeholder(tf.float32, [None], name="done")
        # self._next_observation = tf.placeholder(tf.float32, [None, *self._dim_obs], name="next_observation")
        self._next_observation = self._obs_fn()

        with tf.variable_scope("main/policy"):
            # x = self._dense(self._observation)
            # self._p_act = tf.layers.dense(x, self._dim_act, activation=tf.nn.softmax)

            self._p_act = self._policy_fn(self._observation)

        with tf.variable_scope("main/action_value_1"):
            # x = self._dense(self._observation)
            # self.q1 = tf.layers.dense(x, self._dim_act)

            self.q1 = self._qval_fn(self._observation)

        with tf.variable_scope("main/action_value_2"):
            # x = self._dense(self._observation)
            # self.q2 = tf.layers.dense(x, self._dim_act)

            self.q2 = self._qval_fn(self._observation)

        with tf.variable_scope("main/state_value"):
            # x = self._dense(self._observation)
            # self.v = tf.squeeze(tf.layers.dense(x, 1))
            self.v = self._sval_fn(self._observation)

        with tf.variable_scope("target/state_value"):
            # x = self._dense(self._next_observation)
            # self.v_targ = tf.squeeze(tf.layers.dense(x, 1))
            self.v_targ = self._sval_fn(self._next_observation)

    # def _dense(self, obs):
    #     x = tf.layers.dense(obs, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 128, activation=tf.nn.relu)
    #     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
    #     return x

    def _build_algorithm(self):
        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self._policy_lr)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=self._value_lr)
        policy_variables = tf.trainable_variables("main/policy")
        value_variables = tf.trainable_variables("main/action_value") + tf.trainable_variables("main/state_value")

        min_q = tf.minimum(self.q1, self.q2)
        v_backup = tf.reduce_sum(self._p_act * (min_q - tf.log(self._p_act)), axis=1)
        v_backup = tf.stop_gradient(v_backup)
        q_backup = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * self.v_targ)

        lse_min_q = tf.reduce_logsumexp(min_q, axis=1, keepdims=True)
        log_p_min_q = min_q - lse_min_q
        policy_loss = tf.reduce_mean(tf.reduce_sum(self._p_act * (tf.log(self._p_act) - log_p_min_q), axis=1))

        nsample = tf.shape(self._observation)[0]
        actind = tf.stack([tf.range(nsample), self._action], axis=1)
        q1_act = tf.gather_nd(self.q1, actind)
        q2_act = tf.gather_nd(self.q2, actind)

        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1_act)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2_act)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        value_loss = q1_loss + q2_loss + v_loss

        self._policy_train_op = policy_optimizer.minimize(policy_loss, var_list=policy_variables)
        self._value_train_op = value_optimizer.minimize(value_loss, var_list=value_variables)

        # Update target network.
        def _update_target(net1, net2, alpha=0):
            params1 = tf.trainable_variables(net1)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.trainable_variables(net2)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param1.assign(alpha * param1 + (1-alpha) * param2))
            return update_ops

        with tf.control_dependencies([self._value_train_op]):
            self._update_target_op = _update_target("target/state_value", "main/state_value", alpha=self._update_target_ratio)
        self._init_target_op = _update_target("target/state_value", "main/state_value", alpha=0)

        self._log_op = {"policy_loss": policy_loss, "value_loss": value_loss}

        # # Soft actor-critic loss
        # self.pi_loss = tf.reduce_mean(self.alpha * self.logp_pi - self.q1_pi)
        # q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        # q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        # v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        # self.value_loss = q1_loss + q2_loss + v_loss

        # # Train policy.
        # pi_optimizer = tf.train.AdamOptimizer(learning_rate=self._policy_lr)
        # self.update_policy = pi_optimizer.minimize(self.pi_loss, var_list=self.get_vars("main/policy"))

        # # Train value.
        # value_optimizer = tf.train.AdamOptimizer(learning_rate=self._value_lr)
        # value_vars = self.get_vars("main/action_value") + self.get_vars("main/state_value")
        # with tf.control_dependencies([self.update_policy]):
        #     self.update_value = value_optimizer.minimize(self.value_loss, var_list=value_vars)

        # main_vars = self.get_vars("main/state_value")
        # target_vars = self.get_vars("target/state_value")
        # with tf.control_dependencies([self.update_value]):
        #     self.update_target = tf.group([tf.assign(v_targ, self._update_target_ratio * v_targ + (1 - self._update_target_ratio)*v_main) for v_main, v_targ in zip(main_vars, target_vars)])

        # self.init_target = tf.group([tf.assign(v_targ, v_main) for v_main, v_targ in zip(main_vars, target_vars)])

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

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

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
            self.sw.add_scalars("sac", log, global_step=global_step)

    def get_action(self, obs):
        p_act = self.sess.run(self._p_act, feed_dict={self._observation: obs})
        nsample, nact = p_act.shape
        return [np.random.choice(nact, p=p_act[i, :]) for i in range(nsample)]
