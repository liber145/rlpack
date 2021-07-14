import torch
import random
import numpy as np
from tqdm import tqdm
from IPython import embed
from tensorboardX import SummaryWriter

from rlpack.algos.utils.parser import Parser
from rlpack.algos.doubledqn import DoubleDQN
from rlpack.algos.utils.tools import pack_info
from rlpack.algos.utils.buffer import ReplayBuffer
from rlpack.envs.classic_control import make_classic_control


if __name__ == "__main__":
    args = Parser().parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env = make_classic_control(args.env_name)
    Buffer = ReplayBuffer(args.buffer_size)
    log_dir = f"./log/{args.env_name},dicount:{args.discount}/"

    greedy_info, target_info, net_info, opt_info = pack_info(args)
    log_dir += f"net:{net_info['type']}-{net_info['fc_hidden']};"
    log_dir += f"opt:{opt_info['type']}-{opt_info['lr']}"
    writer = SummaryWriter(log_dir)

    DoubleDQN = DoubleDQN(
         device,env._dim_obs, env._num_act, args.discount, args.loss_fn, args.double_way,
        greedy_info, target_info, net_info, opt_info)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for i in tqdm(range(args.Nepoch)):
        s = env.reset()
        R = 0
        while True:
            a = DoubleDQN.take_step(s)
            ns, r, done, info = env.step(a)
            Buffer.append(s, a, r, ns, done, info)
            R += r

            if done:
                writer.add_scalar("Epoch Reward", R, i)
                break
            
            s = ns
            if i > args.warmup:
                batch = Buffer.sample(args.batch_size)
                loss, loss2, n = DoubleDQN.train(batch)
                writer.add_scalar("Average Loss", loss+loss2/2, n)
