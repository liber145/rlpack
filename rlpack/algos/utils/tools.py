import torch
import numpy as np
import torch.nn as nn

def pack_info(args):
    greedy_info = {
        'type': args.greedy_type,
        'eps-max': args.greedy_eps_max,
        'eps-min': args.greedy_eps_min,
        'eps-decay': args.greedy_eps_decay,
    }
    target_info = {
        'tau': args.target_tau,
        'nupdate': args.target_nupdate,
    }
    net_info = {
        'type': args.net_type,
        'fc_hidden': args.net_fc_hidden,
        'actor_type': args.actor_net_type,
        'actor_fc_hidden': args.actor_fc_hidden,
        'critic_type': args.critic_net_type,
        'critic_fc_hidden': args.critic_fc_hidden,     
    }
    opt_info = {
        'type': args.opt_type,
        'lr': args.opt_lr,
        'actor_type': args.actor_opt,
        'actor_lr': args.actor_lr,
        'critic_type': args.critic_opt,
        'critic_lr': args.critic_lr
    }
    return greedy_info, target_info, net_info, opt_info

def pack_dist_info(args):
    dist_info = {
        'Vmin': args.dist_Vmin, 
        'Vmax': args.dist_Vmax,
        'n': args.dist_n
    }
    return dist_info

def get_aggregate(aggregate_way):
    def get_min(a, b):
        return torch.min(a, b)
    def get_avg(a, b):
        return a+b/2
    
    if aggregate_way == "min":
        return get_min
    elif aggregate_way == "avg":
        return get_avg
    else:
        raise NotImplementedError

def get_opt(param, opt_info):
    if opt_info["type"] == "Adam":
        return torch.optim.Adam(param, lr=opt_info['lr'])
    elif opt_info["type"] == "SGD":
        return torch.optim.SGD(param, lr=opt_info['lr'])
    else:
        raise NotImplementedError

def get_loss(loss_type):
    if loss_type == "MSE":
        return nn.MSELoss()
    elif loss_type == "SmoothL1":
        pass
    else:
        raise NotImplementedError

def get_greedy(greedy_info):
    def epsilon_greedy(target, n, t):
        eps = greedy_info['eps-max'] * np.power(greedy_info['eps-decay'], t)
        eps = max(eps, greedy_info['eps-min'])
        if np.random.rand() < eps:
            return np.random.randint(n)
        else:
            return target

    if greedy_info['type'] == 'epsilon':
        return epsilon_greedy
    else:
        raise NotImplementedError