import torch
import random
import numpy as np
from tqdm import tqdm
from IPython import embed
from tensorboardX import SummaryWriter

from rlpack.algos.utils.parser import Parser
from rlpack.algos.a2c import A2C_discrete
from rlpack.algos.utils.tools import pack_info
from rlpack.algos.utils.buffer import ReplayBuffer
from rlpack.envs.classic_control import make_classic_control


if __name__ == "__main__":
    args = Parser().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = make_classic_control(args.env_name)
    log_dir = f"./log/{args.env_name},dicount:{args.discount}/"

    greedy_info, target_info, net_info, opt_info = pack_info(args)
    log_dir += f"actor:{net_info['actor_type']}-{net_info['actor_fc_hidden']}-{opt_info['actor_type']}-{opt_info['actor_lr']};"
    log_dir += f"critic:{net_info['critic_type']}-{net_info['critic_fc_hidden']}-{opt_info['critic_type']}-{opt_info['critic_lr']};"
    writer = SummaryWriter(log_dir)

    A2C = A2C_discrete(
         device, env._dim_obs, env._num_act, args.discount,
        greedy_info, net_info, opt_info)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    epoch_cnt = 0
    for i in tqdm(range(args.Nepoch)):
        s, R, traj = env.reset(), 0, []

        for _ in range(10):
            while True:
                a = A2C.take_step(s)
                ns, r, done, info = env.step(a)
                traj.append((s, a, r, ns, done, info))
                R += r

                if done:
                    s = env.reset()
                    break
                
                s = ns

        writer.add_scalar("Epoch Reward", R/10, i)

        aloss, closs, n = A2C.train(traj)
        writer.add_scalar("Actor Loss", aloss, n)
        writer.add_scalar("Critic Loss", closs, n)
