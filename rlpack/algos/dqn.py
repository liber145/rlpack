# -*- coding: utf-8 -*-
import torch

from .nets.qnet import QNet


class DQN:
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
        self.target_model = QNet(dim_obs, num_act)
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, obs):
        qvals = self.model(obs)
        return qvals.argmax()

    def compute_loss(self, s_batch, a_batch, r_batch, d_batch, next_s_batch):
        # Compute current Q value based on current states and actions.
        qvals = self.model(s_batch).gather(1, a_batch.unsqueeze(1)).squeeze()

        # Compute next Q value based on which action gives max Q values.
        # Detach variable from the current graph since we don't want gradients for next Q to propagated.
        next_qvals = r_batch + (1 - d_batch) * self._discount * self.target_model(next_s_batch).detach().max(1)[0]

        # delta_qvals = next_qvals - qvals
        # loss = torch.mean(torch.square(delta_qvals))
        loss = torch.nn.functional.smooth_l1_loss(next_qvals, qvals)
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

    def delay_update_target_model(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(0.01 * param.data + (1 - 0.01) * target_param.data)
