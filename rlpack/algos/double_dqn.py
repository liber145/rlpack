from .base import Base
import tensorflow as tf
import numpy as np
import scipy


class DoubleDQN(Base):
    def __init__(self, config):
        super().__init__(config)

    def build_network(self):
        self.observation = tf.placeholder(shape=[None, *self.dim_observation], dtype=tf.float32, name="observation")

        with tf.variable_scope("qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=True)
            self.qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=True)

        with tf.variable_scope("target_qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=False)
            self.target_qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=False)

    def build_algorithm(self):
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.target = tf.placeholder(shape=[None], dtype=tf.float32, name="target")
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1.5e-8)
        trainable_variables = tf.trainable_variables("qnet")

        batch_size = tf.shape(self.observation)[0]
        gather_indices = tf.range(batch_size) * self.n_action + self.action
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, action_q))
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.train.get_global_step(),
                                                var_list=trainable_variables
                                                )

        # 更新目标网络。
        def _update_target(new_net, old_net):
            params1 = tf.trainable_variables(old_net)
            params1 = sorted(params1, key=lambda v: v.name)
            params2 = tf.global_variables(new_net)
            params2 = sorted(params2, key=lambda v: v.name)
            assert len(params1) == len(params2)
            update_ops = []
            for param1, param2 in zip(params1, params2):
                update_ops.append(param2.assign(param1))
            return update_ops

        self.update_target_op = _update_target("target_qnet", "qnet")

        # ------------------------------------------
        # ------------- 需要记录的中间值 --------------
        # ------------------------------------------
        self.max_qval = tf.reduce_max(self.qvals)

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        self.epsilon -= (self.initial_epsilon - self.final_epsilon) / 100000
        self.epsilon = max(self.final_epsilon, self.epsilon)

        qvals = self.sess.run(self.qvals, feed_dict={self.observation: newobs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = newobs.shape[0]
        actions = np.random.randint(self.n_action, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon
        actions[idx] = best_action[idx]

        if obs.ndim == 1:
            actions = actions[0]
        return actions

    def get_action_boltzman(self, obs):
        if obs.ndim == 1:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            newobs = obs

        # 0.01 是一个不错的参数。
        alpha = 0.001

        qvals = self.sess.run(self.qvals, feed_dict={self.observation: newobs})
        exp_m = scipy.special.logsumexp(qvals / alpha, axis=1)
        exp_m = np.exp(qvals / alpha - exp_m)

        global_step = self.sess.run(tf.train.get_global_step())
        self.summary_writer.add_scalar("action_probability", exp_m[0][0], global_step)

        actions = [np.random.choice(self.n_action, p=exp_m[i]) for i in range(newobs.shape[0])]

        if obs.ndim == 1:
            actions = actions[0]
        return actions

    def update(self, minibatch, update_ratio):

        # 拆分样本。
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        batch_size = s_batch.shape[0]
        current_next_q_vals, target_next_q_vals = self.sess.run(
            [self.qvals, self.target_qvals], feed_dict={self.observation: next_s_batch})
        q_next = target_next_q_vals[range(batch_size), current_next_q_vals.argmax(axis=1)]
        target_batch = r_batch + (1 - d_batch) * self.discount * q_next

        _, global_step, loss, max_q_val = self.sess.run(
            [self.train_op,
             tf.train.get_global_step(),
             self.loss,
             self.max_qval],
            feed_dict={
                self.observation: s_batch,
                self.action: a_batch,
                self.target: target_batch
            }
        )

        # 存储结果。
        self.summary_writer.add_scalar("loss", loss, global_step)
        self.summary_writer.add_scalar("max_q_value", max_q_val, global_step)

        # 存储模型。
        if global_step % self.save_model_freq == 0:
            self.save_model()

        # 更新目标策略。
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return {"loss": loss, "max_q_value": max_q_val, "training_step": global_step}
