import argparse
import logging
import os
from rlpack.common.log import logger
from rlpack.estimator.learner import Learner

logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="Process parameters.")
parser.add_argument("--env", default="Atari.BeamRiderNoFrameskip-v4", type=str)
parser.add_argument("--n_env", default=4, type=int)
parser.add_argument("--model", default="dqn", type=str)
parser.add_argument("--epsilon", default=0.01, type=float)
parser.add_argument("--result_path", default=None, type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--n_iteration", default=10000000, type=int)
parser.add_argument("--n_trajectory", default=32, type=int)
parser.add_argument("--n_step", default=1, type=int)
parser.add_argument("--n_dqn", default=2, type=int)
parser.add_argument("--learning_starts", default=200, type=int)
parser.add_argument("--memory_size", default=500000, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument("--critic_lr", default=0.01, type=float)
parser.add_argument("--n_action", default=2, type=int)
parser.add_argument("--dim_action", default=2, type=int)
parser.add_argument("--dim_observation", default=11, type=int)
parser.add_argument("--discount", default=0.99, type=float)
parser.add_argument("--update_target_every", default=1000, type=int)
parser.add_argument("--save_model_every", default=1000, type=int)
args = parser.parse_args()

args.result_path = os.path.join(
    "./results", args.model) if args.result_path is None else args.result_path


def main():
    learner = Learner(args)
    learner.run()


if __name__ == "__main__":
    main()
