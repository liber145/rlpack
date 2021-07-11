import torch

from .nets.qnet import QNet


class DoubleDQN:
    def __init__(
        self,
        dim_obs=None,
        num_act=None,
        discount=0.9,
    ):
        self._dim_obs = dim_obs
        self._num_act = num_act
        self._discount = discount

        self.model = None

    def get_action(self, obs):
        pass

    def compute_loss(self):
        pass

    def save_model(self, ckpt_path):
        pass

    def load_model(self, ckpt_path):
        pass

    @property
    def parameters(self):
        pass
