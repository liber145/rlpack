# Multistep TD SARSA.
import torch

from .nets.qnet import QNet

import pdb


class SARSA:
    def __init__(
        self,
        dim_obs=None,
        num_act=None,
        discount=0.9,
    ):
        self._dim_obs = dim_obs
        self._num_act = num_act
        self._discount = discount

        self.model = QNet(dim_obs, num_act)

    def get_action(self, obs):
        qvals = self.model(obs)
        return qvals.argmax()

    def compute_loss(self, s_episode, a_episode, target_R_episode):
        qvals = self.model(s_episode).gather(1, a_episode.unsqueeze(1)).squeeze()
        loss = torch.nn.functional.smooth_l1_loss(qvals, target_R_episode)

        return loss

    def save_model(self, ckpt_path):
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path))

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    @property
    def parameters(self):
        return self.model.parameters()
