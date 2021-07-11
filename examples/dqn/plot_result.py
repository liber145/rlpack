import matplotlib.pyplot as plt
import pickle
import argparse
import logging
import numpy as np
import os 


def main(args):
    with open(args.result_path, "rb") as f:
        result = pickle.load(f)

    losses = result["loss"]
    rewards = result["reward"]

    xrange = np.arange(len(losses))
    plt.plot(xrange, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(f"{args.env} loss")
    plt.savefig(os.path.join(args.log_dir, f"{args.env}_loss.pdf"), bbox_inches="tight")
    plt.close()

    xrange = np.arange(len(rewards))
    plt.plot(xrange, rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title(f"{args.env} reward")
    plt.savefig(os.path.join(args.log_dir, f"{args.env}_reward.pdf"), bbox_inches="tight")
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description="Plot.")
    parser.add_argument("--env", default="cartpole_v0", type=str)
    parser.add_argument("--result_path", default="./cartpole_v0_log/result.pk", type=str)
    parser.add_argument("--log_dir", default="./cartpole_v0_log", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        logging.exception("Debug:")
