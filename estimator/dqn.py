import numpy as np
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
from estimator.utils import gen_batch
from middleware.log import logger


class DQN(TFEstimator):
    def __init__(self, dim_ob, n_act, lr=1e-4, discount=0.99):
        self.cnt = 1
        super().__init__(dim_ob, n_act, lr, discount)
        self._update_target()

    def _build_model(self):

        assert len(self.dim_ob) == 1 or len(self.dim_ob) == 3, "Wrong observation dimension: {}".format(
            self.dim_ob)

        if len(self.dim_ob) == 1:
            self.input = tf.placeholder(
                shape=[None, self.dim_ob], dtype=tf.float32, name="inputs")
        elif len(self.dim_ob) == 3:
            self.input = tf.placeholder(
                shape=[None] + list(self.dim_ob), dtype=tf.float32, name="inputs")

        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")
        self.target = tf.placeholder(
            shape=[None], dtype=tf.float32, name="target")

        # Build net.
        if len(self.dim_ob) == 1:
            with tf.variable_scope("qnet"):
                self.qvals = Networker.build_dense_net(
                    self.input, [512, 256, self.n_act])
            with tf.variable_scope("target_qnet"):
                self.target_qvals = Networker.build_dense_net(
                    self.input, [512, 256, self.n_act], trainable=False)
        elif len(self.dim_ob) == 3:
            with tf.variable_scope("qnet"):
                self.qvals = Networker.build_cnn_net(self.input, self.n_act)
            with tf.variable_scope("target_qnet"):
                self.target_qvals = Networker.build_cnn_net(
                    self.input, self.n_act, trainable=False)
        # with tf.variable_scope("qnet"):
        #     self.qvals=Networker.build_dense_net(
        #         self.input, [512, 256, self.n_act])
        # with tf.variable_scope("target_qnet"):
        #     self.target_qvals=Networker.build_dense_net(
        #         self.input, [512, 256, self.n_act], trainable=False)

        trainable_variables = tf.trainable_variables("qnet")

        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_act + self.actions
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.target, action_q))
        self.max_qval = tf.reduce_max(self.qvals)

        self.train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables
        )
        self.update_target_op = self._get_update_target_op()

    def _get_update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.global_variables('target_qnet')
        params2 = sorted(params2, key=lambda v: v.name)
        assert len(params1) == len(params2)

        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param2.assign(param1))
        return update_ops

    def update(self, trajectories):

        batch_size = 64
        batch_generator = gen_batch(trajectories, batch_size)

        while True:
            try:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = next(
                    batch_generator)

                target_next_q_vals = self.sess.run(self.target_qvals, feed_dict={
                    self.input: next_state_batch})

                target = reward_batch + (
                    1 - done_batch) * self.discount * target_next_q_vals.max(axis=1)

                _, total_t, loss, max_q_value = self.sess.run(
                    [self.train_op,
                     tf.train.get_global_step(),
                     self.loss,
                     self.max_qval],
                    feed_dict={
                        self.input: state_batch,
                        self.actions: action_batch,
                        self.target: target
                    }
                )
            except StopIteration:
                del batch_generator
                break

        # Update target model.
        if total_t > 1000 * self.cnt:
            logger.debug("self.cnt: {}".format(self.cnt))
            self.cnt += 1
            self._update_target()

        return total_t, {"loss": loss, "max_q_value": max_q_value}

    def get_action(self, obs, epsilon):
        qvals = self.sess.run(self.qvals, feed_dict={self.input: obs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        actions = np.random.randint(self.n_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) > epsilon
        actions[idx] = best_action[idx]
        return actions

    def _update_target(self):
        self.sess.run(self.update_target_op)
