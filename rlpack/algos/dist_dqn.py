import numpy as np
import tensorflow as tf


from .base import Base


class DistDQN(Base):
    def __init__(self,
                 rnd=0,
                 dim_obs=None, n_act=None,
                 policy_fn=None,
                 discount=0.99,
                 update_target_rate=0.9,
                 train_epoch=1, policy_lr=1e-3,
                 n_histogram=51, vmax=10, vmin=-10,
                 save_path="./log", log_freq=10, save_model_freq=1000,
                 ):

        self._n_histogram = n_histogram
        self._vmax = vmax
        self._vmin = vmin
        self._delta = (self._vmax - self._vmin) / (self._n_histogram - 1)
        self._split_points = np.linspace(self._vmin, self._vmax, self._n_histogram)

        self._dim_obs = dim_obs
        self._n_act = n_act
        self._policy_fn = policy_fn
        self._discount = discount
        self._train_epoch = train_epoch
        self._policy_lr = policy_lr
        self._update_target_rate = update_target_rate

        self._save_model_freq = save_model_freq
        self._log_freq = log_freq

        super().__init__(save_path=save_path, rnd=rnd)
        self.sess.run(self.init_target_op)

    def _build_network(self):
        """Build networks for algorithm."""
        self._obs = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="observation")
        self._act = tf.placeholder(tf.int32, [None], name="action")
        self._new_p_act = tf.placeholder(tf.float32, [None, self._n_histogram], name="next_input")
        self._obs2 = tf.placeholder(shape=[None, *self._dim_obs], dtype=tf.float32, name="next_observation")

        with tf.variable_scope("main"):
            self.logits = self._policy_fn(self._obs)

        with tf.variable_scope("target"):
            self.logits_targ = tf.stop_gradient(self._policy_fn(self._obs2))

    def _build_algorithm(self):
        """Build networks for algorithm."""
        value_vars = tf.trainable_variables('main')

        batch_size = tf.shape(self._obs)[0]
        self._p_act = tf.nn.softmax(tf.reshape(self.logits, [-1, self._n_act, self._n_histogram]))
        self._p_act_targ = tf.nn.softmax(tf.reshape(self.logits_targ, [-1, self._n_act, self._n_histogram]))

        action_index = tf.stack([tf.range(batch_size), self._act], axis=1)
        self.action_probs = tf.gather_nd(self._p_act, action_index)
        self.action_probs_clip = tf.clip_by_value(self.action_probs, 0.00001, 0.99999)

        loss = -tf.reduce_mean(tf.reduce_sum(self._new_p_act * tf.log(self.action_probs_clip), axis=-1))
        self.train_policy_op = tf.train.AdamOptimizer(self._policy_lr).minimize(loss, var_list=value_vars)

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
        probs = self.sess.run(self._p_act, feed_dict={self._obs: obs})
        qvals = np.sum(probs * self._split_points, axis=-1)
        best_action = np.argmax(qvals, axis=1)
        return best_action

    def update(self, databatch):
        s_batch, a_batch, r_batch, d_batch, next_s_batch = databatch
        next_q_probs = self.sess.run(self._p_act_targ, feed_dict={self._obs2: next_s_batch})
        next_q_vals = np.sum(next_q_probs * self._split_points, axis=-1)
        best_action = np.argmax(next_q_vals, axis=1)

        def compute_histogram(reward, probability, done):
            m = np.zeros(self._n_histogram, dtype=np.float32)
            projection = (np.clip(reward + self._discount * (1 - done) * self._split_points,
                                  self._vmin, self._vmax) - self._vmin) / self._delta
            for p, b in zip(probability, projection):
                l = np.floor(b).astype(np.int32)
                u = np.ceil(b).astype(np.int32)
                m[l] += p * (u - b)
                m[u] += p * (b - l)
            return m

        new_p_act = []
        for rew, prob, d in zip(r_batch, next_q_probs[np.arange(best_action.shape[0]), best_action], d_batch):
            new_p_act.append(compute_histogram(rew, prob, d))
        new_p_act = np.array(new_p_act)

        for _ in range(self._train_epoch):
            self.sess.run(self.train_policy_op,
                          feed_dict={
                              self._obs: s_batch,
                              self._act: a_batch,
                              self._new_p_act: new_p_act
                          })

        self.sess.run(self.update_target_op)

        # Save model.
        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])

        if global_step % self._save_model_freq == 0:
            self.save_model()
