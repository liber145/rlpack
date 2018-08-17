import argparse
import gym
import numpy as np
from collections import defaultdict
from environment.atari_wrapper import get_atari_env_fn
from environment.master_env import EnvMaker
from estimator import DQN
from middleware.memory import Memory2, Memory3
import pickle
import time
import os
from tensorboardX import SummaryWriter
from middleware.log import logger
import logging
import traceback

logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="Process parameters.")
parser.add_argument("--env", default="Reacher-v2", type=str)
parser.add_argument("--model", default="ppo", type=str)
parser.add_argument("--result_path", default=None, type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--memory_size", default=10000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--critic_lr", default=0.01, type=float)
parser.add_argument("--n_env", default=8, type=int)
parser.add_argument("--n_action", default=2, type=int)
parser.add_argument("--dim_action", default=2, type=int)
parser.add_argument("--dim_observation", default=11, type=int)
parser.add_argument("--discount", default=0.99, type=float)
parser.add_argument("--n_step", default=1, type=int)
parser.add_argument("--update_target_every", default=100, type=int)
parser.add_argument("--save_model_every", default=1000, type=int)
args = parser.parse_args()

args.dim_observation = (84, 84, 4)
args.result_path = os.path.join(
    "./results", args.model) if args.result_path is None else args.result_path


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


env_fn = get_atari_env_fn('BeamRiderNoFrameskip-v4')
num_agents = args.n_env
env = EnvMaker(num_agents=num_agents, env_fn=env_fn, basename='test')
states = env.reset()
dim_ob = env.observation_space.shape
n_act = env.action_space.shape[0]
args.n_action = n_act
model = DQN(args)  # (84, 84, 4), n_act)

# env = EnvMaker(num_agents=num_agents, env_fn=env_fn, basename='test')

# print('action_space')
# print(env.action_space.shape)
# print(env.action_space.nvec.dtype, type(env.action_space))

# print('observation_space')
# assert env.observation_space.shape == (num_agents, 84, 84, 4)
# print(env.observation_space.high[0, 0, 0, 0],
#       env.observation_space.low[0, 0, 0, 0],
#       env.observation_space.shape,
#       env.observation_space.dtype)

# assert env.reset().shape == (num_agents, 84, 84, 4)
# s, r, d, info = env.step(env.action_space.sample())
# assert s.shape == (num_agents, 84, 84, 4)
# assert r.shape == (num_agents,)
# assert d.shape == (num_agents,) and d.dtype == np.bool
# print(info)

# env.close()


# env = EnvMaker(num_agents=num_agents, env_fn=env_fn, basename='test')
# env.reset()
# dim_ob = env.observation_space.shape
# n_act = env.action_space.shape[0]
# reward = np.zeros(num_agents)
# for i in range(1000):
#     s, r, d, info = env.step(env.action_space.sample())
#     reward += r
#     if d.any():
#         print(reward[d])
#         print(info)
#         reward[d] = 0.0
# env.close()

base_path = "train_log"
events_path = os.path.join(base_path, 'events')
models_path = os.path.join(base_path, 'models')
if not os.path.exists(events_path):
    os.makedirs(events_path)
if not os.path.exists(models_path):
    os.makedirs(models_path)

model.load_model(models_path)
summary_writer = SummaryWriter(events_path)
rewards_history = []
pkl_path = '{}/rewards.pkl'.format(base_path)
if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        rewards_history = pickle.load(f)

batch_size = 32
epsilon = 0.01
learning_starts = 200
save_model_every = 1000
update_target_every = 1000
memory_buffer = Memory3(500000)
results_buffer = ResultsBuffer(rewards_history)
num_iterations = 6250000
global_step = 0

try:
    for i in range(learning_starts):
        actions = model.get_action(states, epsilon)
        next_states, rewards, dones, info = env.step(actions)

        # logger.debug("next_states: {}".format(next_states.shape))
        # logger.debug("type: {} | {} | {}".format(
        # rewards.shape, actions.shape, dones.shape))

        # memory_buffer.extend(
        #     zip(states, actions, rewards, next_states, dones))
        memory_buffer.extend(states, actions, rewards, next_states, dones)
        states = next_states

    states = env.reset()
    start = time.time()
    for i in range(num_iterations):
        actions = model.get_action(states, epsilon)
        next_states, rewards, dones, info = env.step(actions)

        results_buffer.update_infos(info, global_step)
        # memory_buffer.extend(
        #     zip(states, actions, rewards, next_states, dones))
        memory_buffer.extend(states, actions, rewards, next_states, dones)

        # global_step, summaries = model.update(
        #     *memory_buffer.sample(batch_size))

        global_step, summaries = model.update(
            memory_buffer.sample(batch_size))

        results_buffer.update_summaries(summaries)

        # if global_step % update_target_every == 0:
        #     model.update_target()

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
