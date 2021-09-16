import os
import torch
import random
import numpy as np
from tqdm import tqdm
from IPython import embed
from tensorboardX import SummaryWriter

from rlpack.algos.utils.parser import Parser
from rlpack.algos.a2c import A2C_discrete
from rlpack.algos.utils.tools import pack_info, get_opt, pack_env_info
from rlpack.algos.utils.buffer import ReplayBuffer
from rlpack.envs.classic_control import make_classic_control


if __name__ == "__main__":
    args = Parser().parse()
    if os.path.exists(f"./default_config/{args.env_name}.jh"):
        config = torch.load(f"./default_config/{args.env_name}.jh")
        args = config
    else:
        if not os.path.exists(f"./default_config"):
            os.makedirs(f"./default_config")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_info = pack_env_info(args)
    greedy_info, target_info, net_info, opt_info = pack_info(args)
    
    if args.env_name in {"Acrobot-v1", "CartPole-v1", "CartPole-v0", "MountainCar-v0"}:
        env = make_classic_control(args.env_name)
    elif 'ram' in args.env_name:
        env = Atari_Raw_Env(args.env_name)
    else:
        env = Atari_Env(args.env_name, env_info)

    Buffer = ReplayBuffer(args.buffer_size)
    log_dir = f"./log/{args.env_name},dicount:{args.discount}/"

    if env.use_cnn:
        log_dir += f"net:{net_info['type']}-{net_info['cnn_hidden']}"
    else:
        log_dir += f"net:{net_info['type']}-{net_info['fc_hidden']};"
    log_dir += f"opt:{opt_info['type']}-{opt_info['lr']}"
    writer = SummaryWriter(log_dir)

    A2C = A2C_discrete(
         device, env._dim_obs, env._num_act, args.discount,
        greedy_info, net_info)

    aopt = get_opt(A2C.parameters[0], {'type': opt_info['actor_type'], 'lr': opt_info['actor_lr']})
    copt = get_opt(A2C.parameters[1], {'type': opt_info['critic_type'], 'lr': opt_info['critic_lr']})

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for i in tqdm(range(args.Nepoch)):
        s, R, traj = env.reset(), 0, []

        for j in range(10):
            while True:
                a = A2C.take_step(s)
                ns, r, done, info = env.step(a)
                traj.append((s, a, r, ns, done, info))
                R += r

                if done:
                    writer.add_scalar("Epoch Reward", R, i*10+j)
                    s, R = env.reset(), 0
                    break
                
                s = ns

        aloss, closs, n = A2C.compute_loss(traj)
        aopt.zero_grad()
        aloss.backward()
        aopt.step()

        copt.zero_grad()
        closs.backward()
        copt.step()

        writer.add_scalar("Actor Loss", aloss.item(), n)
        writer.add_scalar("Critic Loss", closs.item(), n)

    torch.save(args, f"./default_config/{args.env_name}.jh")