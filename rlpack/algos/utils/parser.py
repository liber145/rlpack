import argparse

class Parser():
    def parse(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--warmup", default=10, type=int)

        # Environment Setting
        parser.add_argument("--env_name", default="CartPole-v1", type=str)
        parser.add_argument("--discount", default=0.995, type=float)

        # General Setting
        parser.add_argument("--seed", default=233, type=int)
        parser.add_argument("--Nepoch", default=5000, type=int)
        parser.add_argument("--loss_fn", default="MSE", type=str)
        parser.add_argument("--buffer_size", default=10000, type=int)
        parser.add_argument("--batch_size", default=32, type=float)

        parser.add_argument("--greedy_type", default="epsilon", type=str)
        parser.add_argument("--greedy_eps_max", default=0.1, type=float)
        parser.add_argument("--greedy_eps_min", default=0.001, type=float)
        parser.add_argument("--greedy_eps_decay", default=0.995, type=float)

        parser.add_argument("--target_tau", default=0.01, type=float)
        parser.add_argument("--target_nupdate", default=100, type=int)

        parser.add_argument("--net_type", default='FC', type=str)
        parser.add_argument('--net_fc_hidden', default=[128, 64], type=str)
        
        parser.add_argument('--opt_type', default='Adam', type=str)
        parser.add_argument('--opt_lr', default=1e-3, type=float)

        # Double DQN Setting
        parser.add_argument('--double_way', default="min", type=str)

        # Advantage Actor Critic
        parser.add_argument('--Ntraj', default=500, type=int)

        parser.add_argument('--actor_net_type', default="FC", type=str)
        parser.add_argument('--critic_net_type', default="FC", type=str)
        parser.add_argument('--actor_fc_hidden', default=[128, 64], type=list)
        parser.add_argument('--critic_fc_hidden', default=[128, 64], type=list)
        
        parser.add_argument('--actor_opt', default="Adam", type=str)
        parser.add_argument('--critic_opt', default="Adam", type=str)
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--critic_lr', default=1e-3, type=float)

        return parser.parse_args()