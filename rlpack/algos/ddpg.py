import torch
import torch.nn as nn
from IPython import embed

from .nets.fc import FC, Ensemble_FC
from .utils.tools import *

class DDPG:
    def __init__(
        self,
        device,
        dim_obs = None,
        dim_act = None,
        act_range = None,
        discount = 0.9,
        greedy_info = {'type':'gaussian', 'delta':0.1, 'eps-decay':0.999, 'delta-min':0.001},
        target_info = {'tau': 0.01, 'nupdate': 100},
        net_info = {
            'actor_type': 'FC', 'actor_fc_hidden': [128, 64],
            'critic_type': 'FC', 'critic_fc_hidden': [128, 64]
        },
    ):
        self.n = 0
        self.nupdate = target_info['nupdate']
        self._tau = target_info['tau']
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._act_range = act_range
        self._discount = discount
        self._critic_loss = nn.MSELoss()
        self._random_select = get_greedy(greedy_info)
        self._device = device

        if (type(dim_obs) == int or len(dim_obs) == 1) and (net_info['critic_type'] == 'FC') and (net_info['actor_type']=='FC'):
            self._policy = FC(dim_obs, dim_act, net_info['actor_fc_hidden'], device, torch.tanh)
            self._policy_ = FC(dim_obs, dim_act, net_info['actor_fc_hidden'], device, torch.tanh)
            self._critic = FC(dim_obs+dim_act, 1, net_info['actor_fc_hidden'], device)
            self._critic_ = FC(dim_obs+dim_act, 1, net_info['actor_fc_hidden'], device)
        elif (len(dim_obs) == 3) and (net_info['critic_type'] == 'CNN') and (net_info['actor_type'] == 'CNN'):
            raise NotImplementedError
        else:
            raise NotImplementedError

    def _parse_batch(self, batch, device):
        bs, ba, br, bns, bd, _ = zip(*batch)
        bs = torch.FloatTensor(bs).to(device)
        ba = torch.FloatTensor(ba).to(device)
        bns = torch.FloatTensor(bns).to(device)
        br = torch.FloatTensor(br).to(device)
        bd = torch.LongTensor(bd).to(device)
        return bs, ba, br, bd, bns

    def take_step(self, obs):
        self._evaluate()
        obs = torch.unsqueeze(torch.FloatTensor(obs), 0).to(self._device)
        action = self._random_select(self._policy(obs)[0].detach(), self.n).cpu()
        return np.array(action)

    def compute_loss(self, batch):
        self.n += 1
        self._train()

        bs, ba, br, bd, bns = self._parse_batch(batch, self._device)
        
        next_a = self._policy_(bns).to(self._device)
        next_q = self._critic_(torch.cat((bns,next_a),dim=-1)).squeeze()
        target = br+(1-bd)*next_q*self._discount
        cur_q = self._critic(torch.cat((bs,ba),dim=-1)).squeeze()
        closs = self._critic_loss(cur_q, target)

        cur_a = self._policy(bs).to(self._device)
        aloss = -torch.mean(self._critic(torch.cat((bs,cur_a), dim=-1)))

        return closs, aloss, self.n

    def save_model(self, ckpt_path):
        torch.save({
            "policy": self._policy.state_dict(),
            "policy_target": self._policy_.state_dict(),
            "critic": self._critic.state_dict(), 
            "critic_target": self._critic_.state_dict()}, ckpt_path)

    def load_model(self, ckpt_path):
        load_dict = torch.load(ckpt_path)
        self._policy.load_state_dict(load_dict["policy"])
        self._policy_.load_state_dict(load_dict["policy_target"])
        self._critic.load_state_dict(load_dict["critic"])
        self._critic_.load_state_dict(load_dict["critic_target"])

    def update_target(self):
        for e, t in zip(self._policy.parameters(), self._policy_.parameters()):
            t.data.copy_(self._tau*e+(1-self._tau)*t)
        for e, t in zip(self._critic.parameters(), self._critic_.parameters()):
            t.data.copy_(self._tau*e+(1-self._tau)*t)

    def _evaluate(self):
        self._policy.eval()
        self._critic.eval()
    
    def _train(self):
        self._policy.train()
        self._critic.train()

    @property
    def parameters(self):
        return [self._policy.parameters(), self._critic.parameters()]