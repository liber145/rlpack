import argparse
import logging
import numpy as np
import torch
import random
import os
import pdb
import pickle

from rlpack.algos.sarsa import SARSA
from rlpack.envs.classic_control import make_classic_control


class ReplayBuffer:
    def __init__(self, capacity, dim_obs):
        self.capacity = capacity
        self.dim_obs = dim_obs

        self.clear()

        self.size = 0
        self.capacity = capacity

    def save_sards(self, state, action, reward, done, next_state):
        assert self.size < self.capacity
        ind = self.size % self.capacity
        self.state[ind, :] = state
        self.action[ind] = action
        self.reward[ind] = reward
        self.done[ind] = done
        self.next_state[ind, :] = next_state
        self.size += 1

    def clear(self):
        self.state = np.empty((self.capacity, self.dim_obs), dtype=np.float32)
        self.action = np.empty(self.capacity, dtype=np.int32)
        self.reward = np.empty(self.capacity, dtype=np.float32)
        self.done = np.empty(self.capacity, dtype=np.float32)
        self.next_state = np.empty((self.capacity, self.dim_obs), dtype=np.float32)
        self.size = 0


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_classic_control(args.env)
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Model.
    replay_buffer = ReplayBuffer(100000, env.dim_obs)
    agent = SARSA(dim_obs=env.dim_obs, num_act=env.num_act, discount=0.99)

    # optimizer = torch.optim.SGD(agent.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    num_update_steps = int(args.num_train_steps / (args.update_every_n_episode * 100))
    gamma = (1 / 1e3) ** (1 / num_update_steps)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    # Train.
    epsilon = 0.1
    epsilon_end = 0.001
    epsilon_decay = 0.9995
    history_rews = []
    history_lens = []
    history_losses = [0]
    obs = env.reset()
    cnt_episode = 0
    for step in range(1, 1 + args.num_train_steps):
        if np.random.rand() < epsilon:
            act = env.sample_action()
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        else:
            obs = torch.from_numpy(obs).float().to(device)
            act = agent.get_action(obs)
            act = act.item()

        next_obs, rew, done, info = env.step(act)

        replay_buffer.save_sards(obs, act, rew, done, next_obs)

        obs = next_obs
        if done:
            cnt_episode += 1
            obs = env.reset()
            history_rews.append(info["episode_reward"])
            history_lens.append(info["episode_length"])

        if cnt_episode > 0 and cnt_episode % args.update_every_n_episode == 0:
            cnt_episode = 0

            agent.train()
            ind = replay_buffer.size
            s = replay_buffer.state[:ind, :]
            a = replay_buffer.action[:ind]
            r = replay_buffer.reward[:ind]
            d = replay_buffer.done[:ind]
            replay_buffer.clear()

            target_R = []
            R = 0
            for i in reversed(range(r.size)):
                R = r[i] + (1 - d[i]) * args.discount * R
                target_R.append(R)
            target_R.reverse()

            s = torch.from_numpy(s).float().to(device)
            a = torch.from_numpy(a).long().to(device)
            target_R = torch.tensor(target_R).float().to(device)

            loss = agent.compute_loss(s, a, target_R)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            # lr_scheduler.step()
            agent.eval()

            history_losses.append(loss.item())

        if step % args.log_freq == 0:
            lr = lr_scheduler.get_last_lr()[0]
            print(
                f"step:{step}, mean rew:{np.mean(history_rews[-10:]):.2f}, loss:{np.mean(history_losses[-10:]):.2e}, lr:{lr:.2e}, epsilon:{epsilon:.2e}"
            )

        if step % 10 * args.log_freq == 0:
            os.makedirs(args.log_dir, exist_ok=True)
            agent.save_model(os.path.join(args.log_dir, "pytorch_model.bin"))

    with open(os.path.join(args.log_dir, "result.pk"), "wb") as f:
        pickle.dump({"reward": history_rews, "loss": history_losses}, f)


def get_args():
    parser = argparse.ArgumentParser(description="ClassicControl")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--env", default="CartPole-v0", type=str)
    parser.add_argument("--alg", default="SARSA", type=str)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--log_freq", default=10000, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--update_every_n_episode", default=10, type=int)
    parser.add_argument("--num_train_steps", default=1000000, type=int)
    parser.add_argument("--num_warmup_steps", default=10000, type=int)

    # Optimization.
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = get_args()
        for k, v in args.__dict__.items():
            print(f"{k}: {v}")
        main(args)
    except Exception as e:
        logging.exception("Debug:")
