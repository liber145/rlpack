import numpy as np
import tensorflow as tf

from tensorboardX import SummaryWriter

from .base import Base


class DistDQN(Base):
    def __init__(self, config):
        self.n_histogram = 51
        self.vmax = 1
        self.vmin = -10
        self.delta = (self.vmax - self.vmin) / (self.n_histogram - 1)
        self.split_points = np.linspace(self.vmin, self.vmax, self.n_histogram)

        self.lr = config.value_lr_schedule(0)
        self.epsilon_schedule = config.epsilon_schedule
        self.epsilon = self.epsilon_schedule(0)
        self.update_target_freq = config.update_target_freq
        super().__init__(config)

    def build_network(self):
        self.observation = tf.placeholder(shape=[None, *self.dim_observation], dtype=tf.float32, name="observation")

        with tf.variable_scope("qnet"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self.logits = tf.layers.dense(x, self.dim_action * self.n_histogram)

        with tf.variable_scope("target_qnet"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu, trainable=False)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=False)
            self.target_logits = tf.layers.dense(x, self.dim_action * self.n_histogram, trainable=False)

    def build_algorithm(self):
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.action = tf.placeholder(tf.int32, [None], name="action")
        self.target = tf.placeholder(tf.float32, [None], name="target")
        self.next_input = tf.placeholder(tf.float32, [None, self.n_histogram], name="next_input")

        trainable_variables = tf.trainable_variables('qnet')
        batch_size = tf.shape(self.observation)[0]
        self.probs = tf.nn.softmax(tf.reshape(self.logits, [-1, self.dim_action, self.n_histogram]))
        self.probs_target = tf.nn.softmax(tf.reshape(self.target_logits, [-1, self.dim_action, self.n_histogram]))

        gather_indices = tf.range(batch_size) * self.dim_action + self.action
        self.action_probs = tf.gather(tf.reshape(self.probs, [-1, self.n_histogram]), gather_indices)
        self.action_probs_clip = tf.clip_by_value(self.action_probs, 0.00001, 0.99999)

        self.loss = tf.reduce_mean(-tf.reduce_sum(self.next_input * tf.log(self.action_probs_clip), axis=-1))
        self.train_op = self.optimizer.minimize(
            self.loss, global_step=tf.train.get_global_step(), var_list=trainable_variables)

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
        self.epsilon = self.epsilon_schedule(update_ratio)

        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        n_env, batch_size = s_batch.shape[:2]
        s_batch = s_batch.reshape(n_env * batch_size, *self.dim_observation)
        a_batch = a_batch.reshape(n_env * batch_size)
        r_batch = r_batch.reshape(n_env * batch_size)
        d_batch = d_batch.reshape(n_env * batch_size)
        next_s_batch = next_s_batch.reshape(n_env * batch_size, *self.dim_observation)

        next_q_probs = self.sess.run(self.probs_target, feed_dict={self.observation: next_s_batch})
        next_q_vals = np.sum(next_q_probs * self.split_points, axis=-1)
        best_action = np.argmax(next_q_vals, axis=1)

        def compute_histogram(reward, probability, done):
            m = np.zeros(self.n_histogram, dtype=np.float32)
            projection = (np.clip(reward + self.discount * (1 - done) * self.split_points,
                                  self.vmin, self.vmax) - self.vmin) / self.delta

            for p, b in zip(probability, projection):
                a = int(b)
                m[a] += p * (1 + a - b)
                if a < self.n_histogram - 1:
                    m[a + 1] += p * (b - a)
            return m

        targets = []
        for rew, prob, d in zip(r_batch, next_q_probs[np.arange(best_action.shape[0]), best_action], d_batch):
            targets.append(compute_histogram(rew, prob, d))

        target_batch = np.array(targets)
        _, global_step, loss = self.sess.run([self.train_op, tf.train.get_global_step(), self.loss],
                                             feed_dict={self.observation: s_batch, self.action: a_batch, self.next_input: target_batch})

        # 存储模型。
        if global_step % self.save_model_freq == 0:
            self.save_model(self.save_path)

        # 更新目标策略。
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return {"loss": loss, "training_step": global_step}

    def get_action(self, obs):
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        probs = self.sess.run(self.probs, feed_dict={self.observation: newobs})
        qvals = np.sum(probs * self.split_points, axis=-1)

        best_action = np.argmax(qvals, axis=1)

        batch_size = newobs.shape[0]

        actions = np.random.randint(self.dim_action, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon

        actions[idx] = best_action[idx]
        if obs.ndim == 1:
            actions = actions[0]
        return actions
