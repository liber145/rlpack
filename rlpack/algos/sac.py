import numpy as np
import tensorflow as tf

from .base import Base


class SAC(Base):
    def __init__(self,
                 rnd=0,
                 dim_obs=None, dim_act=None,
                 policy_fn=None, value_fn=None, qvalue_fn=None,
                 discount=0.99, gae=0.95, alpha=0.2,
                 train_epoch=1, policy_lr=1e-3, value_lr=1e-3,
                 target_update_rate=0.995,
                 save_path="./log", log_freq=10, save_model_freq=100,
                 update_target_ratio=0.995):
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._policy_fn = policy_fn
        self._value_fn = value_fn
        self._qvalue_fn = qvalue_fn

        self._discount = discount
        self._gae = gae
        self._alpha = alpha
        self._train_epoch = train_epoch
        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._target_update_rate = target_update_rate

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        self._update_target_ratio = update_target_ratio

        super().__init__(save_path=save_path, rnd=rnd)
        self.sess.run(self._init_target_op)

    def get_vars(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def clip_but_pass_gradient(self, x, l=-1, u=1):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient(clip_up * (u - x) + clip_low * (l - x))

    def _build_network(self):
        self._obs = tf.placeholder(tf.float32, [None, *self._dim_obs], name="observation")
        self._act = tf.placeholder(tf.float32, [None, self._dim_act], name="action")
        self._reward = tf.placeholder(tf.float32, [None])
        self._done = tf.placeholder(tf.float32, [None])
        self._obs2 = tf.placeholder(tf.float32, [None, *self._dim_obs], name="next_observation")
        self.all_phs = [self._obs, self._act, self._reward, self._done, self._obs2]

        with tf.variable_scope("main/policy"):
            self.pi, self.logp_pi = self._policy_fn(self._obs, self._act)

        with tf.variable_scope("main/action_value/1"):
            self.q1 = self._qvalue_fn(self._obs, self._act)

        with tf.variable_scope("main/action_value/2"):
            self.q2 = self._qvalue_fn(self._obs, self._act)

        with tf.variable_scope("main/action_value/1", reuse=True):
            self.q1_pi = self._qvalue_fn(self._obs, self.pi)

        with tf.variable_scope("main/action_value/2", reuse=True):
            self.q2_pi = self._qvalue_fn(self._obs, self.pi)

        with tf.variable_scope("main/state_value"):
            self.v = self._value_fn(self._obs)

        with tf.variable_scope("target/state_value"):
            self.v_targ = self._value_fn(self._obs2)

    def _build_algorithm(self):
        policy_vars = tf.trainable_variables("main/policy")
        value_vars = tf.trainable_variables("main/action_value") + tf.trainable_variables("main/state_value")

        min_q = tf.minimum(self.q1_pi, self.q2_pi)
        v_backup = tf.stop_gradient(min_q - self._alpha * self.logp_pi)
        v_loss = 0.5 * tf.reduce_mean((v_backup - self.v)**2)
        q_backup = tf.stop_gradient(self._reward + self._discount * (1 - self._done) * self.v_targ)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - self.q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - self.q2)**2)
        value_loss = q1_loss + q2_loss + v_loss

        policy_loss = tf.reduce_mean(self._alpha * self.logp_pi - self.q1_pi)

        self.train_policy_op = tf.train.AdamOptimizer(self._policy_lr).minimize(policy_loss, var_list=policy_vars)
        self.train_value_op = tf.train.AdamOptimizer(self._value_lr).minimize(value_loss, var_list=value_vars)

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

        with tf.control_dependencies([self.train_value_op]):
            self._update_target_op = _update_target("target/state_value", "main/state_value", alpha=self._target_update_rate)
        self._init_target_op = _update_target("target/state_value", "main/state_value", alpha=0)

    def get_action(self, obs):
        pi = self.sess.run(self.pi, feed_dict={self._obs: obs})
        return pi

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch
        inputs = {k: v for k, v in zip(self.all_phs, [s_batch, a_batch, r_batch, d_batch, next_s_batch])}

        for _ in range(self._train_epoch):
            self.sess.run([self.train_policy_op, self.train_value_op], feed_dict=inputs)

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()
