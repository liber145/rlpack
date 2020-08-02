#!/usr/bin/python3
"""
wrapper for gail
"""

import numpy as np
import tensorflow as tf
import tqdm
import numpy
import pickle
import os

class GAIL_wrapper:
    def __init__(self, agent, discriminator, env, model_path=None):
        
        self.agent = agent
        self.D = discriminator
        self.saver = tf.train.Saver()
        self.sess = tf.Session() 
        self.sess.run(tf.global_variables_initializer())
        self.env = env
        self.d_save_dir = os.path.join(model_path, "disc")
        self.g_save_dir = os.path.join(model_path, "gen")
        os.makedirs(self.d_save_dir, exist_ok=True)
        os.makedirs(self.g_save_dir, exist_ok=True)

    def train(self, episodes, obs_path, act_path):
        train_fq = 5
        self.rewards_his = []
        expert_observations = np.genfromtxt(obs_path)
        expert_actions = np.genfromtxt(act_path, dtype=np.int32)
        obs_his = []
        act_his = []
        r_his = []
        v_preds_next_his = []
        v_preds_his = []

        for i in tqdm.tqdm(range(episodes)):
            observations = []
            actions = []
            rewards = []
            v_preds = []
            r = 0
            obs = self.env.reset()
            done = False
            while not done:
                act, v_pred = self.agent.act(obs.reshape(-1,4))
                act = act[0]
                next_obs, reward, done, info = self.env.step(act)
                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)
                obs = next_obs
                r += reward

            next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
            _, v_pred = self.agent.act(obs=next_obs, stochastic=True)
            v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
            self.rewards_his.append(r)
            obs_his.extend(observations)
            act_his.extend(actions)
            r_his.extend(rewards)
            v_preds_next_his.extend(v_preds_next)
            v_preds_his.extend(v_preds)

            # update
            # train discriminator
            if (i+1) % train_fq == 0:
                for _ in range(2):
                    self.D.train(expert_s=expert_observations,
                    expert_a=expert_actions,
                    agent_s=obs_his,
                    agent_a=act_his)
                    d_rewards = self.D.get_rewards(agent_s=obs_his, agent_a=act_his).reshape(-1)
                    gaes = self.agent.get_gaes(rewards=d_rewards, v_preds=v_preds_his,v_preds_next=v_preds_next_his)

                    gaes = np.array(gaes).reshape(-1)
                    v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)

                # train policy
                inp = [obs_his, act_his, gaes, d_rewards, v_preds_next_his]
                self.agent.assign_policy_parameters()
                for _ in range(6):
                    sample_indices = np.random.randint(low=0, high=len(observations), size=128)  
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    self.agent.train(obs=sampled_inp[0],
                                        actions=sampled_inp[1],
                                        gaes=sampled_inp[2],
                                        rewards=sampled_inp[3],
                                        v_preds_next=sampled_inp[4])
                
                obs_his = []
                act_his = []
                r_his = []
                v_preds_next_his = []
                v_preds_his = []

            if (i+1) % 1000 == 0:
                r_eval = self.eval(10)
                print("eval r : {}".format(r_eval))
                
        self.agent.saver.save(self.agent.sess, os.path.join(self.g_save_dir, "{}".format(i)))
        self.D.saver.save(self.D.sess, os.path.join(self.d_save_dir, "{}".format(i)))
        print("Model saved")

            
    def eval(self, epo=10):
        r_his = []
        for i in range(epo):
            r = 0
            obs = self.env.reset()
            done = False
            while not done:
                act, v_pred = self.agent.act(obs.reshape(-1,4))
                act = act[0]
                next_obs, reward, done, info = self.env.step(act)
                obs = next_obs
                r += reward
            r_his.append(r)
        return np.mean(r_his)
