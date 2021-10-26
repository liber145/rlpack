import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from IPython import embed

from .nets.fc import FC, Ensemble_FC
from .utils.tools import *

class double_critic(nn.Module):
    def __init__(self, idim, odim, hidden_list, device):
        super(double_critic, self).__init__()
        self.critic1 = FC(idim, odim, hidden_list, device)
        self.critic2 = FC(idim, odim, hidden_list, device)

    def forward(self, x):
        q1, q2 = self.critic1(x), self.critic2(x)
        return q1, q2

class soft_actor(nn.Module):
    def __init__(self, idim, odim, hidden_list, act_mean, act_width,
                    min_logstd, max_logstd, device):
        super(soft_actor, self).__init__()
        self.policy_embed = FC(idim, hidden_list[-1], hidden_list[:-1], device)
        self.mean_out = FC(hidden_list[-1], odim, [], device)
        self.log_std_out = FC(hidden_list[-1], odim, [], device)

        self.action_width = act_width
        self.action_mean = act_mean
        self.min_logstd = torch.tensor(min_logstd).to(device)
        self.max_logstd = torch.tensor(max_logstd).to(device)

    def forward(self, x):
        x = self.policy_embed(x)
        mean = self.mean_out(x)
        log_std = torch.clamp(self.log_std_out(x), self.min_logstd, self.max_logstd)
        std = log_std.exp()

        dist = Normal(mean, std)
        z = dist.rsample()
        log_prob = dist.log_prob(z)
        a = torch.tanh(z)
        log_prob -= torch.log(1-a.pow(2)+1e-6)
        mean = torch.tanh(mean)*self.action_width+self.action_mean
        a = a*self.action_width+self.action_mean
        return a, log_prob.sum(dim=-1, keepdim=True), mean

class SAC:
    def __init__(
        self,
        device,
        init_alpha,
        args,
        dim_obs = None,
        dim_act = None,
        act_range = None,
        discount = 0.99,
        target_info = {'tau': 0.01, 'nupdate': 100},
        net_info = {
            'actor_type': 'FC', 'actor_fc_hidden': [128, 64],
            'critic_type': 'FC', 'critic_fc_hidden': [128, 64]
        },
    ):
        self.n = 0
        self.nupdate = target_info['nupdate']
        self._tau = target_info['tau']
        self._alpha = 0.2
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._act_range = act_range
        self._discount = discount
        self._critic_loss = nn.MSELoss()
        self._device = device
        self._log_alpha = torch.zeros(1, requires_grad=True, device=self._device)
        self._act_mean = 0.5*torch.tensor(self._act_range['low']+self._act_range['high']).to(self._device)
        self._act_width = 0.5*torch.tensor(self._act_range['high']-self._act_range['low']).to(self._device)
        self._tentropy = torch.tensor(-self._dim_act, device=self._device)

        self._min_logstd, self._max_logstd = -20.0, 2.0

        if (type(dim_obs) == int or len(dim_obs) == 1) and (net_info['critic_type'] == 'FC') and (net_info['actor_type']=='FC'):
            self._actor = soft_actor(
                dim_obs, dim_act, net_info['actor_fc_hidden'], self._act_mean,
                self._act_width, self._min_logstd, self._max_logstd, device)
            self._critics = double_critic(dim_obs+dim_act, 1, net_info['critic_fc_hidden'], device)
            self._critics_ = double_critic(dim_obs+dim_act, 1, net_info['critic_fc_hidden'], device)
            self._update_target(1.0)
        elif (len(dim_obs) == 3) and (net_info['critic_type'] == 'CNN') and (net_info['actor_type'] == 'CNN'):
            raise NotImplementedError
        else:
            raise NotImplementedError



    def take_step(self, obs, evaluate=False):
        self._evaluate()
        obs = torch.unsqueeze(torch.FloatTensor(obs), 0).to(self._device)
        action, _, mean = self._actor(obs)
        action = action[0].detach().cpu().numpy()
        if evaluate:
            action = mean[0].detach().cpu().numpy()
        return action

    def _compute_alpha_loss(self, cur_logp):
        alpha_loss = (-self._log_alpha.exp()*(cur_logp+self._tentropy).detach()).mean()
        return alpha_loss

    def compute_loss(self, batch):
        self.n += 1
        self._train()

        bs, ba, br, bd, bns = self._parse_batch(batch, self._device)
        br, bd = br.unsqueeze(1), bd.unsqueeze(1)

        with torch.no_grad():
            nexta, next_logp, _ = self._actor(bns)
            q1_, q2_ = self._critics_(torch.cat((bns,nexta),dim=-1))
            target = br + self._discount*(1-bd)*(torch.min(q1_,q2_)-self._alpha*next_logp)
        q1, q2 = self._critics(torch.cat((bs,ba),dim=-1))
        closs1 = F.mse_loss(q1, target)
        closs2 = F.mse_loss(q2, target)
        closs = (closs1+closs2)/2.0

        cur_a, cur_logp, _ = self._actor(bs)
        q1, q2 = self._critics(torch.cat((bs,cur_a),dim=-1))
        aloss = (-torch.min(q1,q2)+self._alpha*cur_logp).mean()   

        self._update_target(self._tau)

        alpha_loss = self._compute_alpha_loss(cur_logp)
        self._alpha = torch.exp(self._log_alpha)

        return closs, aloss, alpha_loss, self._alpha, self.n

    def _parse_batch(self, batch, device):
        bs, ba, br, bns, bd, _ = zip(*batch)
        bs = torch.FloatTensor(bs).to(device)
        ba = torch.FloatTensor(ba).to(device)
        bns = torch.FloatTensor(bns).to(device)
        br = torch.FloatTensor(br).to(device)
        bd = torch.LongTensor(bd).to(device)
        return bs, ba, br, bd, bns

    def save_model(self, ckpt_path):
        policy_param = self._actor.state_dict()
        critic_param = [self._critics.state_dict(), self._critics_.state_dict()]

        torch.save({
            "policy": policy_param ,
            "critic": critic_param}, ckpt_path)

    def load_model(self, ckpt_path):
        load_dict = torch.load(ckpt_path)
        self._actor.load_state_dict(load_dict["policy"])
        self._critics.load_state_dict(load_dict["critic"][0])
        self._critics_.load_state_dict(load_dict["critic"][1])

    def _update_target(self, tau):
        for e, t in zip(self._critics.parameters(), self._critics_.parameters()):
            t.data.copy_(tau*e+(1-tau)*t)

    def _evaluate(self):
        self._actor.eval()
        self._critics.eval()
    
    def _train(self):
        self._actor.train()
        self._critics.train()

    @property
    def parameters(self):
        return [self._actor.parameters(), self._critics.parameters(), [self._log_alpha]]