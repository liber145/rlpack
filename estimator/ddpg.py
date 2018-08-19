import os
import numpy as np
import tensorflow as tf
from estimator.tfestimator import TFEstimator
from estimator.networker import Networker
import estimator.utils as utils
from middleware.log import logger


class DDPG(TFEstimator):
    """Deep Deterministic Policy Gradient."""

    def __init__(self, config):
        super().__init__(config)
        self.cnt = None

    def _build_model(self):
        # Build placeholders.
        self.observation_ph = tf.placeholder(
            tf.float32, [None]+list(self.dim_ob), "observation")
        self.action_ph = tf.placeholder(
            tf.float32, (None, self.dim_act), "action")
        self.target_qval_ph = tf.placeholder(
            tf.float32, (None, 1), "next_state_qval")
        self.grad_q_act_ph = tf.placeholder(
            tf.float32, (None, self.dim_act), "grad_q_act")

        # Build Q-value net.
        with tf.variable_scope("qval_net"):
            self.qval = Networker.build_qvalue_net(
                self.observation_ph, self.action_ph)

        with tf.variable_scope("dummy_qval_net"):
            self.dummy_qval = Networker.build_qvalue_net(
                self.observation_ph, self.action_ph, False)

        # Build action net.
        with tf.variable_scope("act_net"):
            self.action = Networker.build_action_net(
                self.observation_ph, [64, 64, self.dim_act])

        with tf.variable_scope("dummy_act_net"):
            self.dummy_action = Networker.build_action_net(
                self.observation_ph, [64, 64, self.dim_act], False)

        # ---------- Build Policy Algorithm ----------
        # Compute gradient of qval with respect to action.
        self.grad_q_a = tf.gradients(self.qval, self.action_ph)

        # Compute update direction of policy parameter.
        actor_vars = tf.trainable_variables("act_net")
        grad_surr = tf.gradients(
            self.action / self.batch_size, actor_vars, -self.grad_q_act_ph)

        # Update actor parameters.
        self.train_actor_op = self.optimizer.apply_gradients(
            zip(grad_surr, actor_vars), global_step=tf.train.get_global_step())

        # ---------- Build Value Algorithm ----------
        critic_vars = tf.trainable_variables("qval_net")
        self.value_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(self.qval - self.target_qval_ph), axis=1))

        self.train_critic_op = self.critic_optimizer.minimize(
            self.value_loss, var_list=critic_vars)

    def update(self, trajectories):
        self.cnt = self.sess.run(tf.train.get_global_step()) // self.update_target_every + 1 \
                   if self.cnt is None else self.cnt

        data_batch = utils.trajectories_to_batch(trajectories, self.batch_size, self.discount)
        logger.debug("data_batch[state]: {}".format(data_batch["state"].shape))
        

        batch_generator = utils.generator(data_batch, self.batch_size)
        while True:
            try:
                sample_batch = next(batch_generator)

                logger.debug("sample_batch shape: {}".format(sample_batch.shape))

                # ---------- Update Actor ----------
                feeddict = {self.observation_ph: sample_batch["state"],
                            self.action_ph: sample_batch["action"]}

                grad = self.sess.run(self.grad_q_a, feed_dict=feeddict)
                feeddict[self.grad_q_act_ph] = grad[0]

                _, tot_step = self.sess.run(
                    [self.train_actor_op, tf.train.get_global_step()], feed_dict=feeddict)
            except StopIteration:
                del batch_generator
                break

        batch_generator = utils.generator(data_batch, self.batch_size)
        while True:
            try: 
                sampe_batch = next(batch_generator)
                
                # ---------- Update Critic ----------
                # Compute taget Q-value.
                next_act = self.sess.run(self.dummy_action, feed_dict={
                                         self.observation_ph: sample_batch["nextstate"]})
                next_qval = self.sess.run(self.dummy_qval, 
                                          feed_dict={self.observation_ph: sample_batch["nextstate"],
                                                     self.action_ph: next_act})

                target_qval = sample_batch["reward"] + \
                    (1 - sample_batch["done"]) * self.discount * next_qval

                # Update critic.
                feeddict[self.target_qval_ph] = target_qval

                _, loss = self.sess.run([self.train_critic_op, self.value_loss], 
                                        feed_dict={
                                            self.observation_ph: sample_batch["state"],
                                            self.action_ph: sample_batch["action"]
                                            })
            except StopIteration:
                del batch_generator
                break

        if tot_step > self.cnt * self.update_target_every:
            self._copy_parameters("qval_net", "dummy_qval_net")
            self._copy_parameters("act_net", "dummy_act_net")
            self.cnt += 1

        return tot_step, {"loss": loss}

    def _copy_parameters(self, netnew, netold):
        """Copy parameters from netnew to netold.

        Parameters: 
            netold: string
            netnew: string
        """

        oldvars = tf.trainable_variables(netold)
        newvars = tf.trainable_variables(netnew)

        assign_op = [x.assign(y) for x, y in zip(oldvars, newvars)]
        self.sess.run(assign_op)

    def _trajectories_to_batch(self, trajectories):

        def func(traj):
            return map(np.array, zip(*traj))

        (state_batch,
         action_batch,
         reward_batch,
         nextstate_batch,
         done_batch) = map(np.concatenate, zip(*map(func, trajectories)))

        reward_batch = reward_batch[:, np.newaxis]
        done_batch = done_batch[:, np.newaxis]

        return {"state": state_batch,
                "action": action_batch,
                "reward": reward_batch,
                "nextstate": nextstate_batch,
                "done": done_batch
                }

    def get_action(self, ob, epsilon=0.1):
        logger.debug("ge action: {}".format(ob.shape))
        if np.random.uniform() < epsilon:
            return np.random.uniform(-1, 1, size=(1, self.dim_act))
        else:
            return self.sess.run(self.action, feed_dict={self.observation_ph: ob})
