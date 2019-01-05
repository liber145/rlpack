import numpy as np
import tensorflow as tf

from .base import Base


class DuelDQN(Base):
    """Dueling Archtecture, Double DQN"""

    def __init__(self,
                 rnd=1,
                 n_env=1,
                 dim_obs=None,
                 dim_act=None,
                 discount=0.99,
                 save_path="./log",
                 save_model_freq=1000,
                 update_target_freq=10000,
                 log_freq=1000,
                 epsilon_schedule=lambda x: (1-x)*1,
                 lr=2.5e-4
                 ):

        self.n_env = n_env
        self.dim_obs = dim_obs
        self.dim_act = dim_act
        self.discount = discount
        self.save_model_freq = save_model_freq
        self.log_freq = log_freq

        self.lr = lr
        self.epsilon_schedule = epsilon_schedule
        self.epsilon = self.epsilon_schedule(0)
        self.update_target_freq = update_target_freq
        super().__init__(save_path=save_path, rnd=rnd)

    def build_network(self):
        """Build networks for algorithm."""
        self.observation = tf.placeholder(shape=[None, *self.dim_obs], dtype=tf.uint8, name="observation")
        self.observation = tf.to_float(self.observation) / 256.0

        with tf.variable_scope("net"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu)
            self.v = tf.layers.dense(x, 1)
            self.adv = tf.layers.dense(x, self.dim_act)

        with tf.variable_scope("target_net"):
            x = tf.layers.conv2d(self.observation, 32, 8, 4, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 4, 2, activation=tf.nn.relu, trainable=False)
            x = tf.layers.conv2d(x, 64, 3, 1, activation=tf.nn.relu, trainable=False)
            x = tf.contrib.layers.flatten(x)  # pylint: disable=E1101
            x = tf.layers.dense(x, 512, activation=tf.nn.relu, trainable=False)
            self.target_v = tf.layers.dense(x, 1, trainable=False)
            self.target_adv = tf.layers.dense(x, self.dim_act, trainable=False)

    def build_algorithm(self):
        """Build networks for algorithm."""
        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1.5e-8)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32, name="action")
        self.target = tf.placeholder(shape=[None], dtype=tf.float32, name="target")
        trainable_variables = tf.trainable_variables("net")

        # 计算Q(s,a)。
        self.qvals = self.v + (self.adv - tf.reduce_mean(self.adv, axis=1, keepdims=True))
        self.target_qvals = self.target_v + (self.target_adv - tf.reduce_mean(self.target_adv, axis=1, keepdims=True))

        # 根据action提取Q中的值。
        batch_size = tf.shape(self.observation)[0]
        gather_indices = tf.range(batch_size) * self.dim_act + self.action
        action_q = tf.gather(tf.reshape(self.qvals, [-1]), gather_indices)

        self.loss = tf.reduce_mean(tf.squared_difference(self.target, action_q))
        self.train_op = self.optimizer.minimize(self.loss, var_list=trainable_variables)

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

        self.update_target_op = _update_target("target_net", "net")

        # ------------------------------------------
        # ------------- 需要记录的中间值 --------------
        # ------------------------------------------
        self.max_qval = tf.reduce_max(self.qvals)

    def get_action(self, obs):
        """Get actions according to the given observation.

        Parameters:
            - ob: An ndarray with shape (n, state_dimension).

        Returns:
            - An ndarray for action with shape (n).
        """
        if obs.ndim == 1 or obs.ndim == 3:
            newobs = np.array(obs)[np.newaxis, :]
        else:
            assert obs.ndim == 2 or obs.ndim == 4
            newobs = obs

        qvals = self.sess.run(self.qvals, feed_dict={self.observation: newobs})
        best_action = np.argmax(qvals, axis=1)
        batch_size = newobs.shape[0]
        actions = np.random.randint(self.dim_act, size=batch_size)
        idx = np.random.uniform(size=batch_size) > self.epsilon
        actions[idx] = best_action[idx]

        if obs.ndim == 1:
            actions = actions[0]
        return actions

    def update(self, minibatch, update_ratio) -> dict:
        """Update the algorithm by suing a batch of data.

        Parameters:
            - minibatch: A list of ndarray containing a minibatch of state, action, reward, done, next_state.

                - state shape: (n_env, batch_size, state_dimension)
                - action shape: (n_env, batch_size)
                - reward shape: (n_env, batch_size)
                - done shape: (n_env, batch_size)
                - next_state shape: (n_env, batch_size, state_dimension)

            - update_ratio: float scalar in (0, 1).

        Returns:
            - training infomation.
        """
        # 拆分样本。
        self.epsilon = self.epsilon_schedule(update_ratio)

        s_batch, a_batch, r_batch, d_batch, next_s_batch = minibatch

        mb_s, mb_a, mb_target = [], [], []

        n_env = s_batch.shape[0]
        for i in range(n_env):
            batch_size = s_batch[i, :].shape[0]
            current_next_q_vals, target_next_q_vals = self.sess.run(
                [self.qvals, self.target_qvals], feed_dict={self.observation: next_s_batch[i, :]})
            q_next = target_next_q_vals[range(batch_size), current_next_q_vals.argmax(axis=1)]
            target_batch = r_batch[i, :] + (1 - d_batch[i, :]) * self.discount * q_next

            mb_target.append(target_batch)
            mb_s.append(s_batch[i, :])
            mb_a.append(a_batch[i, :])

        mb_s = np.concatenate(mb_s)
        mb_a = np.concatenate(mb_a)
        mb_target = np.concatenate(mb_target)

        _, loss, max_q_val = self.sess.run(
            [self.train_op,
             self.loss,
             self.max_qval],
            feed_dict={
                self.observation: mb_s,
                self.action: mb_a,
                self.target: mb_target
            }
        )

        global_step, _ = self.sess.run([tf.train.get_global_step(), self.increment_global_step])
        # 存储模型。
        if global_step % self.save_model_freq == 0:
            self.save_model()

        # 更新目标策略。
        if global_step % self.update_target_freq == 0:
            self.sess.run(self.update_target_op)

        return {"loss": loss, "max_q_value": max_q_val, "training_step": global_step}
