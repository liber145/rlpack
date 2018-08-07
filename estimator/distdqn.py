import numpy as np
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
from estimator.utils import gen_batch


class DistDQN(TFEstimator):
    def __init__(self,
                 dim_ob,
                 n_ac,
                 lr=1e-4,
                 discount=0.99,
                 vmax=10,
                 vmin=-10,
                 n_atoms=51):
        self.n_atoms = n_atoms
        self.vmax = vmax
        self.vmin = vmin
        self.delta = (vmax - vmin) / (n_atoms - 1)
        self.split_points = np.linspace(vmin, vmax, n_atoms)
        self.cnt = 1
        super(DistDQN, self).__init__(dim_ob, n_ac, lr, discount)
        self._update_target()

    def _build_model(self):
        # placeholders
        self.input = tf.placeholder(
            shape=[None, self.dim_ob], dtype=tf.float32, name='inputs')
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name='actions')
        self.next_input = tf.placeholder(
            shape=[None, self.n_atoms], dtype=tf.float32, name='next_inputs')
        # network
        with tf.variable_scope('qnet'):
            self.logits = Networker.build_distdqn_net(
                self.input, [512, 256, self.n_act*self.n_atoms])

        with tf.variable_scope('target'):
            self.target_logits = Networker.build_distdqn_net(
                self.input, [512, 256, self.n_act*self.n_atoms], trainable=False)

        self.trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.input)[0]
        self.probs = tf.nn.softmax(
            tf.reshape(self.logits, [-1, self.n_act, self.n_atoms]))
        self.probs_target = tf.nn.softmax(
            tf.reshape(self.target_logits, [-1, self.n_act, self.n_atoms]))
        gather_indices = tf.range(batch_size) * self.n_act + self.actions
        self.action_probs = tf.gather(
            tf.reshape(self.probs, [-1, self.n_atoms]), gather_indices)
        self.action_probs_clip = tf.clip_by_value(self.action_probs, 0.00001,
                                                  0.99999)
        self.loss = tf.reduce_mean(-tf.reduce_sum(
            self.next_input * tf.log(self.action_probs_clip), axis=-1))
        self.train_op = self.optimizer.minimize(
            self.loss,
            global_step=tf.train.get_global_step(),
            var_list=self.trainable_variables)
        self.update_target_op = self._get_update_target_op()

    def _calc_dist(self, reward, discount, probs):
        m = np.zeros(self.n_atoms, dtype=np.float32)
        projections = (np.clip(reward + discount * self.split_points,
                               self.vmin, self.vmax) - self.vmin) / self.delta
        for (p, b) in zip(probs, projections):
            a = int(b)
            m[a] += p * (1 + a - b)
            if a < self.n_atoms - 1:
                m[a + 1] += p * (b - a)
        return m

    def _get_qvals(self, obs):
        probs = self.sess.run(self.probs, feed_dict={self.input: obs})
        qvals = np.sum(probs * self.split_points, axis=-1)
        return qvals

    def update(self, trajectories):

        batch_size = 64
        batch_generator = gen_batch(trajectories, batch_size)

        while True:
            try:
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = next(
                    batch_generator)

                next_q_probs = self.sess.run(
                    self.probs_target, feed_dict={
                        self.input: next_state_batch
                    })
                next_q_vals = np.sum(next_q_probs * self.split_points, axis=-1)
                best_action = np.argmax(next_q_vals, axis=1)

                targets = []
                for reward, probs, done in zip(
                        reward_batch, next_q_probs[np.arange(
                            best_action.shape[0]), best_action],
                        done_batch):
                    targets.append(
                        self._calc_dist(reward, self.discount * (1 - done), probs))
                targets = np.array(targets)
                _, total_t, loss = self.sess.run(
                    [self.train_op,
                     tf.train.get_global_step(), self.loss],
                    feed_dict={
                        self.input: state_batch,
                        self.actions: action_batch,
                        self.next_input: targets
                    })

            except StopIteration:
                del batch_generator
                break

        # Update target model.
        if total_t > 100 * self.cnt:
            self.cnt += 1
            self._update_target()

        return total_t, {"loss": loss}

    def get_action(self, obs, epsilon):
        qvals = self._get_qvals(obs)
        best_action = np.argmax(qvals, axis=1)
        batch_size = obs.shape[0]
        actions = np.random.randint(self.n_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) > epsilon
        actions[idx] = best_action[idx]
        return actions

    def _update_target(self):
        self.sess.run(self.update_target_op)

    def _get_update_target_op(self):
        params1 = tf.trainable_variables('qnet')
        params1 = sorted(params1, key=lambda v: v.name)
        params2 = tf.global_variables('target')
        params2 = sorted(params2, key=lambda v: v.name)
        assert len(params1) == len(params2)

        update_ops = []
        for param1, param2 in zip(params1, params2):
            update_ops.append(param2.assign(param1))
        return update_ops
