import numpy as np
import tensorflow as tf

from .base import Base


class A2C(Base):
    """Advantage Actor Critic."""

    def __init__(self,
                 rnd=0,
                 dim_obs=None, dim_act=None,
                 policy_fn=None, value_fn=None,
                 discount=0.99, gae=0.95,
                 train_epoch=10, policy_lr=1e-3, value_lr=1e-3, batch_size=64,
                 save_path="./log", log_freq=10, save_model_freq=1000,
                 ):

        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._policy_fn = policy_fn
        self._value_fn = value_fn

        self._discount = discount
        self._gae = gae
        self._train_epoch = train_epoch
        self._policy_lr = policy_lr
        self._value_lr = value_lr
        self._batch_size = batch_size

        self._log_freq = log_freq
        self._save_model_freq = save_model_freq

        super().__init__(save_path=save_path, rnd=rnd)

    def _build_network(self):
        """Build networks for algorithm."""
        # Build inputs.
        self._obs = tf.placeholder(tf.float32, [None, *self._dim_obs], "observation")
        self._act = tf.placeholder(tf.float32, [None, self._dim_act], "action")
        self._adv = tf.placeholder(tf.float32, [None])
        self._ret = tf.placeholder(tf.float32, [None], "target_state_value")
        self.all_phs = [self._obs, self._act, self._adv, self._ret]

        with tf.variable_scope("policy"):
            self.pi, self.logp = self._policy_fn(self._obs, self._act)

        with tf.variable_scope("value"):
            self.v = self._value_fn(self._obs)

    def _build_algorithm(self):
        policy_vars = tf.trainable_variables("policy")
        value_vars = tf.trainable_variables("value")

        policy_loss = -tf.reduce_mean(self.logp * self._adv)
        value_loss = tf.reduce_mean((self.v - self._ret) ** 2)

        self.train_policy_op = tf.train.AdamOptimizer(self._policy_lr).minimize(policy_loss, var_list=policy_vars)
        self.train_value_op = tf.train.AdamOptimizer(self._value_lr).minimize(value_loss, var_list=value_vars)

    def get_action(self, obs):
        pi = self.sess.run(self.pi, feed_dict={self._obs: obs})
        return pi

    def update(self, databatch):

        states, actions, advantages, returns, oldlogproba = self._parse_databatch(*databatch)
        inputs = {k: v for k, v in zip(self.all_phs, [states, actions, advantages, returns])}

        for _ in range(self._train_epoch):
            self.sess.run(self.train_value_op, feed_dict=inputs)

        for _ in range(self._train_epoch):
            self.sess.run(self.train_policy_op, feed_dict=inputs)

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()

    def _parse_databatch(self, states, actions, rewards, dones, earlystops, nextstates):

        batch_size = len(dones)
        oldlogproba, values = self.sess.run([self.logp, self.v], feed_dict={self._obs: states, self._act: actions})
        nextvalues = self.sess.run(self.v, feed_dict={self._obs: nextstates})

        returns = np.zeros(batch_size)
        deltas = np.zeros(batch_size)
        advantages = np.zeros(batch_size)

        for i in reversed(range(batch_size)):

            if dones[i]:
                prev_return = 0
                prev_value = 0
                prev_advantage = 0
            elif earlystops[i]:
                prev_return = nextvalues[i]
                prev_value = prev_return
                prev_advantage = 0

            returns[i] = rewards[i] + self._discount * prev_return * (1 - dones[i])
            deltas[i] = rewards[i] + self._discount * prev_value * (1 - dones[i]) - values[i]
            # ref: https://arxiv.org/pdf/1506.02438.pdf (generalization advantage estimate)
            advantages[i] = deltas[i] + self._discount * self._gae * prev_advantage * (1 - dones[i])

            prev_return = returns[i]
            prev_value = values[i]
            prev_advantage = advantages[i]

        advantages = (advantages - advantages.mean()) / advantages.std()

        return [states, actions, advantages, returns, oldlogproba]
