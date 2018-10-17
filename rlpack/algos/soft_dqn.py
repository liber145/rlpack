from .baseq import BaseQ
import tensorflow as tf
from tensorflow import keras
import numpy as np
import scipy


class SoftDQN(BaseQ):
    def __init__(self, config):
        super().__init__(config)

    def build_network(self):
        self.observation = tf.placeholder(tf.float32, [None, self.dim_observation], name="observation")
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.target = tf.placeholder(tf.float32, [None], name="target")

        with tf.variable_scope("qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=True)
            self.qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=True)

        with tf.variable_scope("target_qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=False)
            self.target_qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=False)

    def build_algorithm(self):
        self.alpha = 0.1
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1.5-8)
        trainable_variables = tf.trainable_variables("qnet")

        # Compute Q(s,a)
        batch_size = tf.shape(self.observation)[0]
        gather_indices = tf.range(batch_size) * self.n_action + self.action
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)

        # Compute target_v.
        self.target_v = self.alpha * tf.reduce_logsumexp(self.target_qvals / self.alpha, axis=1)
        self.v = self.alpha * tf.reduce_logsumexp(self.qvals / self.alpha, axis=1)

        # Compute action_probability.
        self.action_probability = tf.exp((self.qvals - self.v) / self.alpha)

        # Compute loss.
        self.loss = tf.reduce_mean(tf.squared_difference(self.target, action_q))
        self.train_op = self.optimizer.minimize(self.loss,
                                                global_step=tf.train.get_global_step(),
                                                var_list=trainable_variables)

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
        if obs.ndim == 1:
            newobs = np.array(obs)[np.newaxis, :]
        elif obs.ndim == 2:
            newobs = obs
        else:
            raise RuntimeError

        batch_size = newobs.shape[0]
        assert batch_size == 1

        self.epsilon -= (self.initial_epsilon - self.final_epsilon) / 100000
        self.epsilon = max(self.final_epsilon, self.epsilon)

        # Compute action probability.
        exp_m = self.sess.run(self.action_probability, feed_dict={self.observation: newobs})
        assert exp_m.shape == (newobs.shape[0], self.n_action)

        global_step = self.sess.run(tf.train.get_global_step())
        self.summary_writer.add_scalar("action_probability", exp_m[0][0], global_step)

        # sample according to the above action probability.
        # random_actions = np.random.randint(self.n_action, size=batch_size)
        actions = [np.random.choice(self.n_action, p=exp_m[i]) for i in range(newobs.shape[0])]
        # idx = np.random.uniform(size=batch_size) < self.epsilon
        # actions = np.array(actions)
        # actions[idx] = random_actions[idx]

        if obs.ndim == 1:
            actions = actions[0]
        return actions

    def get_action_greedy(self, obs):
        if obs.ndim == 1:
            newobs = np.array(obs)[np.newaxis, :]
        else:
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
        # alpha = 0.001

        qvals = self.sess.run(self.qvals, feed_dict={self.observation: newobs})
        exp_m = scipy.special.logsumexp(qvals / self.alpha, axis=1)
        exp_m = np.exp(qvals / self.alpha - exp_m)

        global_step = self.sess.run(tf.train.get_global_step())
        self.summary_writer.add_scalar("action_probability", exp_m[0][0], global_step)

        actions = [np.random.choice(self.n_action, p=exp_m[i]) for i in range(newobs.shape[0])]

        if obs.ndim == 1:
            actions = actions[0]
        return actions

    def update(self, minibatch):
        # 拆分样本。
        s_batch, a_batch, r_batch, next_s_batch, d_batch = minibatch

        # 计算目标Q值。
        # target_next_q_vals, target_next_logsumexp_q = self.sess.run([self.target_qvals, self.target_v],
        #                                                             feed_dict={self.observation: next_s_batch})
        # p_a = np.exp((target_next_q_vals - np.array(target_next_logsumexp_q)[:, np.newaxis]) / self.alpha)
        # p_a = np.clip(p_a, 0.001, 1)
        # tmp = (target_next_q_vals / self.alpha) * np.log(p_a)
        # v_batch = self.alpha * scipy.special.logsumexp(tmp, axis=1)

        # print(f"tmp: {tmp.shape}")
        # print(f"p_a shape: {p_a.shape}")
        # print(f"{p_a[0][0]} {p_a[0][1]}")
        # input()

        # v_batch = self.alpha * np.log(np.sum(np.exp(target_next_q_vals / self.alpha), axis=1))

        v_batch = self.sess.run(self.target_v, feed_dict={self.observation: next_s_batch})

        target_batch = r_batch + (1 - d_batch) * self.discount * v_batch

        # print(f"r_batch: {r_batch.shape}")
        # print(f"target_batch: {target_batch.shape}")
        # print(f"v_batch: {v_batch.shape}")
        # input()

        # 更新策略。
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
            self.save_model(self.save_path)

        # 更新目标策略。
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return global_step, {"loss": loss, "max_q_value": max_q_val}
