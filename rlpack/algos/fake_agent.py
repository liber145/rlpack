import numpy as np


class FakeAgent(object):
    def __init__(self, n_action):
        self.n_action = n_action

    def get_action(self, processed_observation):
        """"Return action by inferenc on the observation."""
        return np.random.randint(self.n_action)

    def update(self, minibatch):
        """Update policy according to the minibatch."""
        pass
