import torch

from .nets.fc import FC
from .utils.tools import *


class DoubleDQN:
    def __init__(
        self,
        dim_obs=None,
        dim_act=None,
        discount=0.9,
        loss_fn = "MSE",
        double_way = "min",
        target_info = {'tau': 0.01, 'nupdate': 100},
        net_info = {'type': 'FC', 'fc_hidden': [256, 128, 64, 64]},
        opt_info = {'type': 'SGD', 'lr': 1e-3},
    ):
        self.n = 0
        self._dim_obs = dim_obs
        self._dim_act = num_act
        self._discount = discount
        self._loss_fn = get_loss(loss_fn)
        self._aggregate = get_aggregate(double_way)
        self._tau = target_info['tau']
        self._nupdate = target_info['nupdate']


        if (type(dim_obs) == int or len(dim_obs.shape) == 1) and (net_info['type'] == 'FC'):
            self._eval, self._target = FC(dim_obs, dim_act, net_info['fc_hidden']), FC(dim_obs, dim_act, net_info['fc_hidden'])
            self._eval2, self._target2 = FC(dim_obs, dim_act, net_info['fc_hidden']), FC(dim_obs, dim_act, net_info['fc_hidden'])
            self._opt = get_opt(self._eval.parameters(), opt_info)
            self._opt2 = get_opt(self._eval2.parameters(), opt_info)
        else:
            raise NotImplementedError

    def act(self, obs):
        self._eval()
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(torch.FloatTensor(obs), 0)
        qvals = self._aggregate(self._eval(obs), self._eval(obs))[0].detach()
        return qvals.argmax()
        
    def _compute_loss(self, bs, ba, br, bd, bns):
        qcur = self._eval(bs).gather(1, ba.unsqueeze(1)).squeeze()
        qcur2 = self._eval2(bs).gather(1, ba.unsqueeze(1)).squeeze()

        qnext = self._aggregate(self._target(bns), self._target2(bns)).detach()
        qtarget = b_r + self._discount * qnext.max(1)[0]

        loss = self._loss_fn(qcur, qtarget)
        loss2 = self._loss_fn(qcur2, qtarget)

        return loss, loss2

    def save_model(self, ckpt_path):
        torch.save({
            "eval": self._eval.state_dict(),
            "target": self._target.state_dict(),
            "eval2": self._eval2.state_dict(), 
            "target2": self._target2.state_dict()}, ckpt_path)

    def load_model(self, ckpt_path):
        load_dict = torch.load(ckpt_path)
        self._eval.load_state_dict(load_dict["eval"])
        self._eval2.load_state_dict(load_dict["eval2"])
        self._target.load_state_dict(load_dict["target"])
        self._target2.load_state_dict(load_dict["target2"])
    
    def _update_target(self):
        for e, t in zip(self._eval.parameters(), self._target.parameters()):
            t.data.copy_(self.tau * e + (1-self.tau) * t)
        for e, t in zip(self._eval2.parameters(), self._target2.parameters()):
            t.data.copy_(self.tau * e + (1-self.tau) * t)

    def _parse_batch(self, batch, device):


    def train(self, batch, device):
        self._train()

        bs, ba, br, bd, bns = self._parse_batch(self, batch, device)
        loss, loss2 = self._compute_loss(bs, ba, br, bd, bns)

        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

        self._opt2.zero_grad()
        loss2.backward()
        self._opt2.step()

        self.n += 1
        if self.n % self._nupdate == 0:
            self._update_target()
        
        return loss.item(), loss2.item()

    def _eval(self):
        self._eval.eval()
        self._eval2.eval()
    
    def _train(self):
        self._eval.train()
        self._eval2.train()


    @property
    def parameters(self):
        pass
