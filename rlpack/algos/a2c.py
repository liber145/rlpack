import torch
import torch.nn as nn
from IPython import embed

from .nets.fc import FC
from .utils.tools import *


class A2C_discrete:
    def __init__(
        self,
        device,
        dim_obs = None,
        dim_act = None,
        discount = 0.9,
        greedy_info = {'type':"epsilon", 'eps-max':0.1, 'eps-min':0.001, 'eps-decay':0.995},
        net_info = {
            'actor_type': 'FC', 'actor_fc_hidden': [128, 64],
            'critic_type': 'FC', 'critic_fc_hidden': [128, 64]
        },
        opt_info = {
            'actor_type': 'Adam', 'actor_lr': 1e-3,
            'critic_type': 'Adam', 'critic_lr': 1e-3
        }
    ):
        self.n = 0
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount
        self._critic_loss = nn.MSELoss()
        self._greedy_select = get_greedy(greedy_info)
        self._device = device

        if (type(dim_obs) == int or len(dim_obs.shape) == 1) and (net_info['critic_type'] == 'FC') and (net_info['actor_type']=='FC'):
            self._actor = FC(dim_obs, dim_act, net_info['actor_fc_hidden'], device)
            self._critic = FC(dim_obs, 1, net_info['critic_fc_hidden'], device)
            self._aopt = get_opt(self._actor.parameters(), {'type': opt_info['actor_type'], 'lr': opt_info['actor_lr']})
            self._copt = get_opt(self._critic.parameters(), {'type': opt_info['critic_type'], 'lr': opt_info['critic_lr']})
        else:
            raise NotImplementedError

    def take_step(self, obs):
        self._evaluate()
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(torch.FloatTensor(obs), 0).to(self._device)
        qvals = self._actor(obs)[0].detach()
        return self._greedy_select(qvals.argmax().item(), self._dim_act, self.n)
        
    def _compute_loss(self, s, a, r, d, ns):
        adv, ret = np.zeros(len(d)), np.zeros(len(d))
        Vref = self._critic(s).detach()
        
        for i in reversed(range(len(d))):
            if d[i]:
                prev_ret = 0
            ret[i] = r[i] + self._discount*prev_ret*(1-d[i])
            adv[i] = ret[i]-Vref[i]
            prev_ret = ret[i]

        pi = torch.softmax(self._actor(s), dim=1)[np.arange(len(d)), a]

        adv = torch.FloatTensor(adv).to(self._device)
        ret = torch.FloatTensor(ret).to(self._device)
        aloss = -torch.mean(torch.log(pi)*adv)

        V = self._critic(s).squeeze()
        closs = self._critic_loss(V, ret)
        
        return aloss, closs

    def save_model(self, ckpt_path):
        torch.save({
            "actor": self._actor.state_dict(),
            "critic": self._critic.state_dict()}, ckpt_path)

    def load_model(self, ckpt_path):
        load_dict = torch.load(ckpt_path)
        self._actor.load_state_dict(load_dict["actor"])
        self._critic.load_state_dict(load_dict["critic"])

    def _parse_traj(self, batch, device):
        bs, ba, br, bns, bd, _ = zip(*batch)
        bs = torch.FloatTensor(bs).to(device)
        ba = torch.LongTensor(ba).to(device)
        bns = torch.FloatTensor(bns).to(device)
        br = torch.FloatTensor(br).to(device)
        return bs, ba, br, bd, bns

    def train(self, traj):
        self._train()

        bs, ba, br, bd, bns = self._parse_traj(traj, self._device)
        aloss, closs = self._compute_loss(bs, ba, br, bd, bns)

        self._aopt.zero_grad()
        aloss.backward()
        self._aopt.step()

        self._copt.zero_grad()
        closs.backward()
        self._copt.step()

        self.n += 1
        
        return aloss.item(), closs.item(), self.n

    def _evaluate(self):
        self._actor.eval()
        self._critic.eval()
    
    def _train(self):
        self._actor.train()
        self._critic.train()


    @property
    def parameters(self):
        pass