from .baseq import BaseQ
import tensorflow as tf
import numpy as np
import scipy
from ..common.utils import assert_shape


class PCL(BaseQ):
    def __init__(self, config):
        self.tau = 1.0   # Boltzman exploration中的温度系数。
        self.lr_actor = 0.01
        self.lr_critic = 0.01

        super().__init__(config)

    def build_network(self):
        self.observation = tf.placeholder(shape=[None, self.dim_observation], dtype=tf.float32, name="observation")
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")

        with tf.variable_scope("qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=True)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=True)
            self.qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=True)

        with tf.variable_scope("target_qnet"):
            x = tf.layers.dense(self.observation, 32, activation=tf.nn.relu, trainable=False)
            x = tf.layers.dense(x, 32, activation=tf.nn.relu, trainable=False)
            self.target_qvals = tf.layers.dense(x, self.n_action, activation=None, trainable=False)

    def build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(0.001)
        trainable_variables = tf.trainable_variables("qnet")
        self.td = tf.placeholder(tf.float32, [None], name="temporal_difference")

        # 训练数据的数量。
        batch_size = tf.shape(self.observation)[0]
        # action_index = tf.range(batch_size) * self.n_action + self.action

        # action value.
        self.action_value = self.tau * tf.reduce_logsumexp(self.qvals / self.tau, axis=1)
        self.target_action_value = self.tau * tf.reduce_logsumexp(self.target_qvals / self.tau, axis=1)
        assert_shape(self.action_value, [None])
        assert_shape(self.target_action_value, [None])

        # action probability.
        # print(f"qvals: {self.qvals.shape}")
        # print(f"action_value: {self.action_value.shape}")
        self.action_probability = tf.exp((self.qvals - tf.expand_dims(self.action_value, axis=1)) / self.tau)
        assert_shape(self.action_probability, [None, self.n_action])

        # selected action probability.
        selected_action_index = tf.stack([tf.range(batch_size), self.action], axis=1)
        assert_shape(selected_action_index, [None, 2])

        self.selected_action_probability = tf.gather_nd(self.action_probability, selected_action_index)
        assert_shape(self.selected_action_probability, [None])

        # 计算梯度。
        grad_actor = tf.gradients(tf.log(self.selected_action_probability), trainable_variables, grad_ys=self.td / tf.to_float(batch_size))
        grad_critic = tf.gradients(self.action_value, trainable_variables, grad_ys=self.td / tf.to_float(batch_size))

        grad_composed = [-(self.lr_actor * ga + self.lr_critic * gc) for ga, gc in zip(grad_actor, grad_critic)]
        self.train_op = self.optimizer.apply_gradients(zip(grad_composed, trainable_variables), global_step=tf.train.get_global_step())

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

    def update(self, minibatch, update_ratio):
        # 拆分样本。
        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        batch_size = s_batch.shape[0]

        action_value, target_action_value = self.sess.run([self.action_value, self.target_action_value], feed_dict={self.observation: next_s_batch})
        td_batch = r_batch + (1 - d_batch) * self.discount * target_action_value

        _, global_step = self.sess.run([self.train_op, tf.train.get_global_step()],
                                       feed_dict={self.observation: s_batch,
                                                  self.action: a_batch,
                                                  self.td: td_batch
                                                  })

        # 存储模型。
        if global_step % self.save_model_freq == 0:
            self.save_model(self.save_path)

        # 更新目标策略。
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return global_step

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        qval, actionval, action_probability = self.sess.run([self.qvals, self.action_value, self.action_probability], feed_dict={self.observation: newobs})
        # print(f"action_probability: {action_probability}")
        # print(f"qval: {qval}  actionval: {actionval}")
        actions = [np.random.choice(self.n_action, p=action_probability[i]) for i in range(newobs.shape[0])]

        if obs.ndim == 1:
            actions = actions[0]
        return actions
