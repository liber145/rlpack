import os
import torch
import random
import numpy as np
from tqdm import tqdm
from IPython import embed
from tensorboardX import SummaryWriter

from rlpack.algos.utils.parser import Parser
from rlpack.algos.ddpg import DDPG
from rlpack.algos.utils.tools import *
from rlpack.algos.utils.buffer import ReplayBuffer
from rlpack.envs.classic_control import make_classic_control
from rlpack.envs.mujoco_control import make_mujoco_control
from rlpack.envs.atari_wrapper import Atari_Env, Atari_Raw_Env


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
    elif args.env_name in {'Hopper-v2', 'Ant-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2'}:
        env = make_mujoco_control(args.env_name)
    elif 'ram' in args.env_name:
        env = Atari_Raw_Env(args.env_name)
    else:
        env = Atari_Env(args.env_name, env_info)

    Buffer = ReplayBuffer(args.buffer_size)
    log_dir = f"./log/{args.env_name},dicount:{args.discount}/"

    if env.use_cnn:
        log_dir += f"net:{net_info['type']}-{net_info['cnn_hidden']}"
    else:
        log_dir += f"actor:{net_info['actor_type']}-{net_info['actor_fc_hidden']},{opt_info['actor_type']}-{opt_info['actor_lr']};"
        log_dir += f"critic:{net_info['critic_type']}-{net_info['critic_fc_hidden']},{opt_info['critic_type']}-{opt_info['critic_lr']};"


    writer = SummaryWriter(log_dir)

    act_range = {
        'low':env._range_act['low'], 
        'high':env._range_act['high']}
    clip = clip_w_range(act_range)

    DDPG = DDPG(
        device, env._dim_obs, env._dim_act, act_range,
        args.discount, greedy_info, target_info, net_info)

    opta_info = {'type': opt_info['actor_type'], 'lr': opt_info['actor_lr']}
    optc_info = {'type': opt_info['critic_type'], 'lr': opt_info['critic_lr']}

    opta = get_opt(DDPG.parameters[0], opta_info)
    optc = get_opt(DDPG.parameters[1], optc_info)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cnt = 0
    for i in tqdm(range(args.Nepoch)):
        s = env.reset()
        R = 0
        while True:
            a = DDPG.take_step(s)
            ns, r, done, info = env.step(a)
            Buffer.append(s, a, r, ns, done, info)
            R += r
            cnt += 1

            if done:
                writer.add_scalar("Epoch Reward", R, i)
                break
            
            s = ns
            if i > args.warmup and cnt % 50 == 0:
                for t in range(50):
                    batch = Buffer.sample(args.batch_size)
                    closs, aloss, n = DDPG.compute_loss(batch)

                    opta.zero_grad()
                    aloss.backward()
                    opta.step()

                    optc.zero_grad()
                    closs.backward()
                    optc.step()

                    writer.add_scalar("Actor Loss", aloss.item(), n)
                    writer.add_scalar('Critic Loss', closs.item(), n)

                    DDPG.update_target()

    torch.save(args, f"./default_config/{args.env_name}.jh")
