import numpy as np
import copy
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
from middleware.log import logger


class PPO(TFEstimator):
    """Proximal Policy Optimization."""

    def __init__(self, dim_ob, dim_act, lr=3e-4, discount=0.99):
        self.dim_act = dim_act
        self.delta = 0.01
        self.beta = 1.0
        super().__init__(dim_ob, None, lr, discount)

    def _build_model(self):
        # Build inputs.
        self.observation = tf.placeholder(
            tf.float32, [None, self.dim_ob], "observation")
        self.action = tf.placeholder(
            tf.float32, [None, self.dim_act], "action")
        self.span_reward = tf.placeholder(tf.float32, [None, 1], "span_reward")
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantages")

        self.old_log_var = tf.placeholder(tf.float32, [self.dim_act], "olvar")
        self.old_mu = tf.placeholder(
            tf.float32, [None, self.dim_act], "old_mu")

        self.beta_holder = tf.placeholder(tf.float32, (), "beta")

        # Build Nets.
        with tf.variable_scope("gauss_net"):
            self.mu, self.log_var = Networker.build_gauss_net(self.observation,
                                                              [64, 64, self.dim_act])

        with tf.variable_scope("value_net"):
            self.val = Networker.build_value_net(
                self.observation, [128, 64, 32, 1])

        # ------------ Build actor algorithm. -------------
        self.actor_vars = tf.trainable_variables("gauss_net")

        logp = -0.5 * tf.reduce_sum(self.log_var)
        logp += -0.5 * tf.reduce_sum(tf.square(self.action - self.mu) / tf.exp(self.log_var),
                                     axis=1,
                                     keepdims=True)

        logp_old = -0.5 * tf.reduce_sum(self.old_log_var)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.action - self.old_mu) / tf.exp(self.old_log_var),
                                         axis=1,
                                         keepdims=True)

        # Compute kl divergence.
        log_det_cov_old = tf.reduce_sum(self.old_log_var)
        log_det_cov_new = tf.reduce_sum(self.log_var)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_var - self.log_var))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.mu - self.old_mu) / tf.exp(self.log_var), axis=1) -
                                       self.dim_act)

        # Build surrgoate loss.
        ratio = tf.exp(logp - logp_old)
        surr1 = ratio * self.advantage
        surr2 = tf.clip_by_value(ratio, 1.0 - 0.2, 1.0 + 0.2) * self.advantage
        # + self.beta_holder * self.kl
        self.surrgoate = -tf.reduce_mean(tf.minimum(surr1, surr2))

        # self.surrgoate = -tf.reduce_mean(self.advantage * tf.exp(logp - logp_old)) + self.beta_holder * self.kl

        # Update actor by adam.
        self.train_actor_op = self.optimizer.minimize(
            self.surrgoate, var_list=self.actor_vars)

        # ---------- Build critic algorithm. ----------
        self.critic_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, "value_net")

        self.critic_loss = tf.reduce_mean(
            tf.square(self.val - self.span_reward))

        self.train_critic_op = self.critic_optimizer.minimize(
            self.critic_loss, global_step=tf.train.get_global_step(), var_list=self.critic_vars)

        # ---------- Build action. ----------
        self.sampled_act = (self.mu + tf.exp(self.log_var / 2.0) *
                            tf.random_normal(shape=[self.dim_act], dtype=tf.float32))

    def update(self, trajectories):
        data_batch = self._trajectories_to_batch(trajectories)

        self.feeddict = {self.observation: data_batch["state"],
                         self.beta_holder: self.beta
                         }

        # Compute old mu, old log var.
        old_mu_val, old_log_var_val = self.sess.run(
            [self.mu, self.log_var], feed_dict=self.feeddict)

        # ---------- Update actor ----------
        # Compute advantage.
        nextstate_val = self.sess.run(
            self.val, feed_dict={self.observation: data_batch["nextstate"]})
        state_val = self.sess.run(
            self.val, feed_dict={self.observation: data_batch["state"]})

        advantage = (data_batch["reward"] + self.discount *
                     (1 - data_batch["done"]) * nextstate_val) - state_val

        # Shuffle batch.
        batch_size = 64
        n_sample = data_batch["state"].shape[0]

        index = np.arange(data_batch["state"].shape[0])
        np.random.shuffle(index)

        data_batch["state"] = data_batch["state"][index, :]
        data_batch["action"] = data_batch["action"][index, :]
        data_batch["spanreward"] = data_batch["spanreward"][index, :]
        old_mu_val = old_mu_val[index, :]
        advantage = advantage[index, :]

        self.feeddict = {self.observation: data_batch["state"][index, :],
                         self.action: data_batch["action"][index, :],
                         self.span_reward: data_batch["spanreward"][index, :],
                         self.old_mu: old_mu_val[index, :],
                         self.old_log_var: old_log_var_val,
                         self.advantage: advantage[index, :]
                         }

        for _ in range(10):

            for i in range(int(np.ceil(n_sample/batch_size))):
                span_index = slice(
                    i*batch_size, min((i+1)*batch_size, n_sample))

                self.feeddict = {self.observation: data_batch["state"][span_index, :],
                                 self.action: data_batch["action"][span_index, :],
                                 self.span_reward: data_batch["spanreward"][span_index, :],
                                 self.old_mu: old_mu_val[span_index, :],
                                 self.old_log_var: old_log_var_val,
                                 self.advantage: advantage[span_index, :]
                                 }

                # Update
                _, kl = self.sess.run(
                    [self.train_actor_op, self.kl], feed_dict=self.feeddict)

                # if kl < self.delta / 1.5:
                #     self.beta /= 2.0
                # elif kl > self.delta * 1.5:
                #     self.beta *= 2.0

                # self.feeddict[self.beta_holder] = self.beta

        # self.beta = 1.0

        # ---------- Update critic ----------
        critic_loss = self.sess.run(self.critic_loss, feed_dict=self.feeddict)

        print("old critic loss:", critic_loss)
        for _ in range(10):

            for i in range(int(np.ceil(n_sample/batch_size))):
                span_index = slice(
                    i*batch_size, min((i+1)*batch_size, n_sample))

                self.feeddict = {self.observation: data_batch["state"][span_index, :],
                                 self.span_reward: data_batch["spanreward"][span_index, :],
                                 }

                _, total_t, critic_loss = self.sess.run(
                    [self.train_critic_op, tf.train.get_global_step(), self.critic_loss], feed_dict=self.feeddict)

        print("new critic loss:", critic_loss)

        return total_t, {"loss": critic_loss}

    def _process_traj(self, traj):
        # traj 的构成: sarsd
        res = []
        # span_reward 表示从traj中最后一个state到当前state之间的discount reward总和。
        span_reward = 0
        laststate = traj[-1][3]
        lastdone = traj[-1][4]
        for transition in reversed(traj):
            span_reward = transition[2] + span_reward * self.discount
            res.append([transition[0], transition[1], transition[2],
                        transition[3], transition[4], span_reward, laststate, lastdone])

        return res

    def _trajectories_to_batch(self, trajectories):
        sarsdts = []
        for traj in trajectories:
            sarsdts.extend(self._process_traj(traj))

        (state_batch,
         action_batch,
         reward_batch,
         nextstate_batch,
         done_batch,
         return_batch,
         laststate_batch,
         lastdone_batch) = map(np.array, zip(*sarsdts))

        reward_batch = reward_batch[:, np.newaxis]
        return_batch = return_batch[:, np.newaxis]
        done_batch = done_batch[:, np.newaxis]
        lastdone_batch = lastdone_batch[:, np.newaxis]

        logger.debug("return batch shape: {}".format(return_batch.shape))

        return {"state": state_batch,
                "action": action_batch,
                "reward": reward_batch,
                "nextstate": nextstate_batch,
                "done": done_batch,
                "spanreward": return_batch,
                "laststate": laststate_batch,
                "lastdone": lastdone_batch
                }

    def get_action(self, ob, epsilon=None):
        action = self.sess.run(self.sampled_act, feed_dict={
                               self.observation: ob})
        return action
