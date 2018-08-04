import numpy as np
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
from middleware.log import logger


class AveDQN(TFEstimator):
    def __init__(self, dim_ob, n_act, k=2, lr=1e-4, discount=0.99):
        self.k = k
        super().__init__(dim_ob, n_act, lr, discount)

        self._update_target()

    def _build_model(self):
        # placeholders
        self.input = tf.placeholder(
            shape=[None, self.dim_ob], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(
            shape=[None], dtype=tf.float32, name='next_inputs')

        qvals = []
        for i in range(self.k + 1):
            with tf.variable_scope('qnet-{}'.format(i)):
                qvals.append(Networker.build_dense_net(
                    self.input, [512, 256, self.n_act], i == 0))

        self.qvals = qvals[0]
        self.target_qvals = tf.stack(qvals[1:])

        trainable_variables = tf.trainable_variables('qnet-0/')
        batch_size = tf.shape(self.input)[0]
        gather_indices = tf.range(batch_size) * self.n_act + self.actions
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)
        self.loss = tf.reduce_mean(
            tf.squared_difference(self.next_input, action_q))
        self.max_qval = tf.reduce_max(self.qvals)

        self.train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=trainable_variables)

        self.update_target_op = self._get_update_target_op()

    def _get_update_target_op(self):
        update_ops = None
        for i in range(self.k, 0, -1):
            with tf.control_dependencies(update_ops):
                params1 = tf.global_variables('qnet-{}/'.format(i))
                params1 = sorted(params1, key=lambda v: v.name)
                if i != 1:
                    params2 = tf.global_variables('qnet-{}/'.format(i - 1))
                else:
                    # The AdamOptimizer will create some variables which are
                    # named starting with 'qnet-0', so we use
                    # tf.trainable_variables to get correct variables.
                    params2 = tf.trainable_variables('qnet-{}/'.format(i - 1))
                params2 = sorted(params2, key=lambda v: v.name)
                assert len(params1) == len(params2)

                update_ops = []
                for param1, param2 in zip(params1, params2):
                    update_ops.append(param1.assign(param2))

        return update_ops

    def update(self, state_batch, action_batch, reward_batch, next_state_batch,
               done_batch):
        # batch_size = state_batch.shape[0]
        target_next_q_vals = self.sess.run(
            self.target_qvals, feed_dict={
                self.input: next_state_batch
            })

        target_next_q_vals_best = np.array(target_next_q_vals).mean(
            axis=0).max(axis=-1)

        targets = reward_batch + (
            1 - done_batch) * self.discount * target_next_q_vals_best
        _, total_t, loss, max_q_value = self.sess.run(
            [
                self.train_op,
                tf.train.get_global_step(), self.loss, self.max_qval
            ],
            feed_dict={
                self.input: state_batch,
                self.actions: action_batch,
                self.next_input: targets
            })

        # Update target model.
        if total_t % 100 == 0:
            self._update_target()

        return total_t, {'loss': loss, 'max_q_value': max_q_value}

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
