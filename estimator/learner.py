import argparse
import gym
import numpy as np
from collections import defaultdict
from environment.atari_wrapper import get_atari_env_fn
from environment.mujoco_wrapper import get_mujoco_env_fn
from environment.master_env import EnvMaker
from estimator import DQN, SoftDQN, DoubleDQN, AveDQN, DistDQN, PG, TRPO, A2C, PPO, PPO, DDPG
from middleware.memory import Memory2, Memory3
import pickle
import time
import os
from tensorboardX import SummaryWriter
from middleware.log import logger
import logging
import traceback


# parser = argparse.ArgumentParser(description="Process parameters.")
# parser.add_argument("--env", default="Reacher-v2", type=str)
# parser.add_argument("--n_env", default=8, type=int)
# parser.add_argument("--model", default="ppo", type=str)
# parser.add_argument("--result_path", default=None, type=str)
# parser.add_argument("--batch_size", default=32, type=int)
# parser.add_argument("--n_iteration", default=10000000, type=int)
# parser.add_argument("--n_trajectory", default=32, type=int)
# parser.add_argument("--n_step", default=1, type=int)
# parser.add_argument("--learning_starts", default=200, type=int)
# parser.add_argument("--memory_size", default=500000, type=int)
# parser.add_argument("--lr", default=0.0001, type=float)
# parser.add_argument("--critic_lr", default=0.01, type=float)
# parser.add_argument("--n_action", default=2, type=int)
# parser.add_argument("--dim_action", default=2, type=int)
# parser.add_argument("--dim_observation", default=11, type=int)
# parser.add_argument("--discount", default=0.99, type=float)
# parser.add_argument("--update_target_every", default=1000, type=int)
# parser.add_argument("--save_model_every", default=1000, type=int)
# args = parser.parse_args()

# args.dim_observation = (84, 84, 4)
# args.result_path = os.path.join(
#     "./results", args.model) if args.result_path is None else args.result_path


class ResultsBuffer(object):
    def __init__(self, rewards_history=[]):
        self.buffer = defaultdict(list)
        assert isinstance(rewards_history, list)
        self.rewards_history = rewards_history

    def update_infos(self, info, total_t):
        for key in info:
            msg = info[key]
            self.buffer['reward'].append(msg[b'reward'])
            self.buffer['length'].append(msg[b'length'])
            if b'real_reward' in msg:
                self.buffer['real_reward'].append(msg[b'real_reward'])
                self.buffer['real_length'].append(msg[b'real_length'])
                self.rewards_history.append(
                    [total_t, key, msg[b'real_reward']])

    def update_summaries(self, summaries):
        for key in summaries:
            self.buffer[key].append(summaries[key])

    def add_summary(self, summary_writer, total_t, time):
        s = {'time': time}
        for key in self.buffer:
            if self.buffer[key]:
                s[key] = np.mean(self.buffer[key])
                self.buffer[key].clear()

        for key in s:
            summary_writer.add_scalar(key, s[key], total_t)


class Learner(object):
    def __init__(self, config):
        self.config = config

    def run(self):
        config = self.config

        # env_fn = get_atari_env_fn(config.env)  # 'BeamRiderNoFrameskip-v4')
        env_fn = get_mujoco_env_fn(config.env)
        num_agents = config.n_env
        env = EnvMaker(num_agents=num_agents, env_fn=env_fn, basename='test')

        config.dim_observation = env.observation_space.shape[1:]
        config.n_action = env.action_space.shape[0]

        model_name = config.model
        if model_name == "dqn":
            model = DQN(config)
        elif model_name == "softdqn":
            model = SoftDQN(config)
        elif model_name == "doubledqn":
            model = DoubleDQN(config)
        elif model_name == "avedqn":
            model = AveDQN(config)
        elif model_name == "distdqn":
            model = DistDQN(config)
        elif model_name == "pg":
            model = PG(config)
        elif model_name == "trpo":
            model = TRPO(config)
        elif model_name == "ppo":
            model = PPO(config)
        elif model_name == "a2c":
            model = A2C(config)
        elif model_name == "ddpg":
            model = DDPG(config)
        else:
            logger.error("Unrecognized model name!")

        events_path = os.path.join(config.result_path, 'events')
        models_path = os.path.join(config.result_path, 'models')
        if not os.path.exists(events_path):
            os.makedirs(events_path)
        if not os.path.exists(models_path):
            os.makedirs(models_path)

        model.load_model(models_path)
        summary_writer = SummaryWriter(events_path)
        rewards_history = []
        pkl_path = '{}/rewards.pkl'.format(config.result_path)
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                rewards_history = pickle.load(f)

        batch_size = config.batch_size
        epsilon = config.epsilon
        learning_starts = config.learning_starts
        save_model_every = config.save_model_every
        memory_buffer = Memory3(config.memory_size)
        results_buffer = ResultsBuffer(rewards_history)
        n_iteration = config.n_iteration
        global_step = 0

        try:
            states = env.reset()
            for i in range(learning_starts):
                actions = model.get_action(states, epsilon)
                next_states, rewards, dones, info = env.step(actions)

                memory_buffer.extend(
                    states, actions, rewards, next_states, dones)
                states = next_states

            states = env.reset()
            start = time.time()
            for i in range(n_iteration):
                actions = model.get_action(states, epsilon)
                next_states, rewards, dones, info = env.step(actions)

                results_buffer.update_infos(info, global_step)
                memory_buffer.extend(
                    states, actions, rewards, next_states, dones)

                global_step, summaries = model.update(
                    memory_buffer.sample(batch_size))

                results_buffer.update_summaries(summaries)

                if global_step % save_model_every == 0:
                    t = time.time() - start
                    model.save_model(models_path)
                    print("Save model, global_step: {}, delta_time: {}.".format(
                        global_step, t))
                    results_buffer.add_summary(summary_writer, global_step, t)
                    start = time.time()

                states = next_states

        except Exception as e:
            # raise e
            traceback.print_exc()

        finally:
            model.save_model(models_path)
            with open(pkl_path, 'wb') as f:
                pickle.dump(results_buffer.rewards_history, f)
            env.close()
