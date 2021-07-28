import torch
import torch.nn as nn
from IPython import embed

from .nets.fc import FC
from .utils.tools import *


class DoubleDQN:
    def __init__(
        self,
        device,
        dim_obs=None,
        dim_act=None,
        discount=0.9,
        loss_fn = "MSE",
        greedy_info = {'type': "epsilon", 'eps-max': 0.1, 'eps-min': 0.001, 'eps-decay': 0.995},
        target_info = {'tau': 0.01, 'nupdate': 100},
        dist_info = {'Vmin': -10, 'Vmax': 10, 'n': 51},
        net_info = {'type': 'FC', 'fc_hidden': [256, 128, 64, 64]},
        opt_info = {'type': 'Adam', 'lr': 1e-3},
    ):
        self.n = 0
        self._dim_obs = dim_obs
        self._dim_act = dim_act
        self._discount = discount
        self._loss_fn = get_loss(loss_fn)
        self._greedy_select = get_greedy(greedy_info)
        self._tau = target_info['tau']
        self.nupdate = target_info['nupdate']
        self._device = device
        self._Vmin, self._Vmax = dist_info['Vmin'], dist_info['Vmax']
        self._natoms, self._dz = dist_info['n'], (self._Vmax-self._Vmin)/(dist_info['n']-1)
        self._support = torch.arange(self._Vmin, self._Vmax+self._dz, self._dz)

        if (type(dim_obs) == int or len(dim_obs.shape) == 1) and (net_info['type'] == 'FC'):
            self._eval = FC(dim_obs, dim_act*self._natoms, net_info['fc_hidden'], device)
            self._target = FC(dim_obs, dim_act*self._natoms, net_info['fc_hidden'], device) 
        else:
            raise NotImplementedError

    def take_step(self, obs):
        self._evaluate()
        if len(obs.shape) == 1:
            obs = torch.unsqueeze(torch.FloatTensor(obs), 0).to(self._device)
        qout = F.softmax(self._eval(obs).view(obs.shape[0], self._dim_act, self._natoms), dim=2).squeeze(0).detach()
        qvals = torch.sum(qout*self._support, dim=1)
        return self._greedy_select(qvals.argmax().item(), self._dim_act, self.n)
        
    def compute_loss(self, batch):
        self.n += 1
        self._train()
        bs, ba, br, bd, bns = self._parse_batch(batch, self._device)

        next_qout = F.softmax(self._target(bns).view(bs.shape[0], self._dim_act, self._natoms), dim=2).detach()
        next_qvals = torch.sum(next_qout*self._support, dim=1)
        next_qdist = next_qout[np.arange(bs.shape[0]), next_qvals.max(1)[1]]

        cur_qout = F.softmax(self._eval(bs).view(bs.shape[0], self._dim_act, self._natoms), dim=2)
        cur_qdist = cur_qout[np.arange(bs.shape[0]), ba]
        
        m = torch.zeros(cur_qdist.shape)
        for i in range(bs.shape[0]):
            for j in range(self._natoms):
                Tzj = (br[i]+self._gamma*next_qdist[i][j]).clamp(self._Vmin, self._Vmax)
                bj = (Tzj-self._Vmin)/self._dz
                l, u = bj.floor, bj.ceil()
                m[i][l] += (u-bj)*next_qdist[i][j]
                m[i][u] += (bj-l)*next_qdist[i][j]
        m = m.to(self._device)

        loss = -torch.sum(m*torch.log(cur_qdist))

        return loss, self.n

    def save_model(self, ckpt_path):
        torch.save({
            "eval": self._eval.state_dict(),
            "target": self._target.state_dict()}, ckpt_path)

    def load_model(self, ckpt_path):
        load_dict = torch.load(ckpt_path)
        self._eval.load_state_dict(load_dict["eval"])
        self._target.load_state_dict(load_dict["target"])
    
    def update_target(self):
        for e, t in zip(self._eval.parameters(), self._target.parameters()):
            t.data.copy_(self._tau * e + (1-self._tau) * t)

    def _parse_batch(self, batch, device):
        bs, ba, br, bns, bd, _ = zip(*batch)
        bs = torch.FloatTensor(bs).to(device)
        ba = torch.LongTensor(ba).to(device)
        bns = torch.FloatTensor(bns).to(device)
        br = torch.FloatTensor(br).to(device)
        return bs, ba, br, bd, bns

    def _evaluate(self):
        self._eval.eval()
    
    def _train(self):
        self._eval.train()

    @property
    def parameters(self):
        return self._eval.parameters()
