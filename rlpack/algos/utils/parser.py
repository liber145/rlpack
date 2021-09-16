import argparse

class Parser():
    def parse(self):
        parser = argparse.ArgumentParser()

        # Environment Setting
        parser.add_argument("--warmup", default=10, type=int)
        parser.add_argument("--env_name", default="CartPole-v0", type=str)
        parser.add_argument("--discount", default=0.995, type=float)
        parser.add_argument("--noop_max", default=30, type=int)
        parser.add_argument('--env_skip', default=4, type=int)

        # General Setting
        parser.add_argument("--seed", default=233, type=int)
        parser.add_argument("--Nepoch", default=20000, type=int)
        parser.add_argument("--loss_fn", default="MSE", type=str)
        parser.add_argument("--buffer_size", default=10000, type=int)
        parser.add_argument("--batch_size", default=32, type=float)

        parser.add_argument("--greedy_type", default="epsilon", type=str)
        parser.add_argument("--greedy_eps_max", default=1.0, type=float)
        parser.add_argument("--greedy_eps_min", default=0.1, type=float)
        parser.add_argument("--greedy_eps_decay", default=0.999, type=float)

        parser.add_argument("--target_tau", default=0.1, type=float)
        parser.add_argument("--target_nupdate", default=1000, type=int)

        parser.add_argument("--net_type", default='FC', type=str)
        parser.add_argument('--net_fc_hidden', default=[128, 64], type=list)
        parser.add_argument('--net_cnn_hidden', default=[[32, 8, 4], [64, 4, 2], [64, 3, 1]], type=list)        

        parser.add_argument('--opt_type', default='Adam', type=str)
        parser.add_argument('--opt_lr', default=0.001, type=float)

        # Double DQN Setting
        parser.add_argument('--double_way', default="min", type=str)

        # Advantage Actor Critic
        parser.add_argument('--Ntraj', default=500, type=int)

        parser.add_argument('--actor_net_type', default="FC", type=str)
        parser.add_argument('--critic_net_type', default="FC", type=str)
        parser.add_argument('--actor_fc_hidden', default=[128, 64], type=list)
        parser.add_argument('--critic_fc_hidden', default=[128, 64], type=list)
        parser.add_argument('--actor_cnn_hidden', default=[[32, 8, 4], [64, 4, 2], [64, 3, 1]], type=list)
        parser.add_argument('--critic_cnn_hidden', default=[[32, 8, 4], [64, 4, 2], [64, 3, 1]], type=list)

        parser.add_argument('--actor_opt', default="SGD", type=str)
        parser.add_argument('--critic_opt', default="SGD", type=str)
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--critic_lr', default=1e-3, type=float)

        # Categorical distribution
        parser.add_argument('--dist_Vmin', default=-10, type=int)
        parser.add_argument('--dist_Vmax', default=10, type=int)
        parser.add_argument('--dist_n', default=51, type=int)

        return parser.parse_args()
