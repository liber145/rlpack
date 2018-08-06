import numpy as np
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
from middleware.log import logger


class DQN(TFEstimator):
    def __init__(self, dim_ob, n_act, lr=1e-4, discount=0.99):
        self.cnt = 1
        super().__init__(dim_ob, n_act, lr, discount)
        self._update_target()

    def _build_model(self):

        self.input = tf.placeholder(
            shape=[None, self.dim_ob], dtype=tf.float32, name="inputs")
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")
        self.target = tf.placeholder(
            shape=[None], dtype=tf.float32, name="target")

        # Build net.
        with tf.variable_scope("qnet"):
            self.qvals = Networker.build_dense_net(self.input, [512, 256, 2])
        with tf.variable_scope("target_qnet"):
            self.target_qvals = Networker.build_dense_net(
                self.input, [512, 256, 2], trainable=False)

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
        sarsd = []
        for traj in trajectories:
            sarsd.extend(traj)

        (state_batch,
         action_batch,
         reward_batch,
         next_state_batch,
         done_batch) = map(np.array, zip(*sarsd))

        assert action_batch.ndim == 1, "Unrecognized action dimension."

        batch_size = 64
        n_sample = state_batch.shape[0]
        index = np.arange(n_sample)
        np.random.shuffle(index)

        # print("state batch shape:", state_batch.shape)
        # print("action batch shape:", action_batch.shape)
        # print("reward batch shape:", reward_batch.shape)
        # print("next state batch shape:", next_state_batch.shape)
        # print("done batch shape:", done_batch.shape)

        state_batch = state_batch[index, :]
        action_batch = action_batch[index]
        reward_batch = reward_batch[index]
        next_state_batch = next_state_batch[index, :]
        done_batch = done_batch[index]

        for i in range(int(np.ceil(n_sample/batch_size))):
            span_index = slice(i*batch_size, min((i+1)*batch_size, n_sample))

            target_next_q_vals = self.sess.run(self.target_qvals, feed_dict={
                                               self.input: next_state_batch[span_index, :]})

            target = reward_batch[span_index] + (
                1 - done_batch[span_index]) * self.discount * target_next_q_vals.max(axis=1)

            _, total_t, loss, max_q_value = self.sess.run(
                [self.train_op,
                 tf.train.get_global_step(),
                 self.loss,
                 self.max_qval],
                feed_dict={
                    self.input: state_batch[span_index, :],
                    self.actions: action_batch[span_index],
                    self.target: target
                }
            )

        # Update target model.
        if total_t > 100 * self.cnt:
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
