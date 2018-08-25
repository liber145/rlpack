import argparse
import gym
import numpy as np
import pickle
import time
import os
import traceback
from tensorboardX import SummaryWriter
from collections import defaultdict
from ..environment.atari_wrapper import get_atari_env_fn
from ..environment.mujoco_wrapper import get_mujoco_env_fn
from ..environment.master_env import EnvMaker
from . import DQN, SoftDQN, DoubleDQN, AveDQN, DistDQN, PG, TRPO, A2C, PPO, DDPG
from ..common.memory import Memory2, Memory3
from ..common.log import logger

class ResultsBuffer(object):
    """Summary of results here.

    Attributes:
        buffer: A dict mapping keys to the corresponding results along time steps.
        reward_history: A list storing rewards along time steps.
    """

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
    """Run the specified algorithm to interact with the environment to learn policy.

    Attributes:
        config: An argparse object for learning config.
    """
    def __init__(self, config):
        self.config = config

    def run(self):
        """Interact with the environment according to the config."""
        config = self.config

        env_type, env_name = config.env.split(".")
        assert env_type == "Atari" or env_type == "Mujoco", "Unrecognized environment."
        if env_type == "Atari":
            env_fn = get_atari_env_fn(env_name)
        elif env_type == "Mujoco":
            env_fn = get_mujoco_env_fn(env_name)

        num_agents = config.n_env
        env = EnvMaker(num_agents=num_agents, env_fn=env_fn, basename='test')

        if env_type == "Atari":
            config.dim_observation = env.observation_space.shape[1:]
            config.n_action = env.action_space.n
        elif env_type == "Mujoco":
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
