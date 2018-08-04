import argparse
import logging
import gym
from environment.agent import Agent
from environment.env import MujocoEnv
from middleware.log import logger

logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(description="Process parameters.")
parser.add_argument("--result_path", default="./results/ppo/", type=str)
parser.add_argument("--env_name", default="Reacher-v2", type=str)
parser.add_argument("--model", default="ppo", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--memory_size", default=10000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--n_env", default=4, type=int)
parser.add_argument("--n_action", default=2, type=int)
parser.add_argument("--dim_action", default=2, type=int)
parser.add_argument("--dim_ob", default=11, type=int)
parser.add_argument("--discount", default=0.99, type=float)
parser.add_argument("--n_step", default=100, type=int)
parser.add_argument("--update_target_every", default=100, type=int)
parser.add_argument("--save_model_every", default=1000, type=int)
args = parser.parse_args()


def main():
    for i in range(args.n_env):
        env = MujocoEnv("{}".format(i).encode(
            "ascii"), args.n_step, args.env_name)
        env.start()

    game = gym.make(args.env_name)
    args.dim_ob = game.observation_space.shape[0]
    if game.action_space.shape == ():
        args.n_action = game.action_space.n
    else:
        args.n_action = game.action_space.shape[0]
    del game

    agent = Agent(args.result_path, args.model, args.lr, args.n_env, args.discount,
                  args.batch_size, args.memory_size, args.n_action, args.dim_ob)
    agent.run()


if __name__ == "__main__":
    main()
