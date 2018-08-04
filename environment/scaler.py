import numpy as np


class Scaler(object):
    """ 
    https://en.wikipedia.org/wiki/Normalization_(statistics)
    计算观测值的均值和方差。
    输入的观测值的大小为[N, obs_dim]。
    """

    def __init__(self, obs_dim):
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.first_pass = True

    def update(self, x):
        """x is [N, obs_dim]"""

        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) +
                         (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            # occasionally goes negative, clip
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    # def get(self):
    #     return 1/(np.sqrt(self.vars) + 0.1)/3, self.means

    @property
    def mean(self):
        return self.means

    @property
    def var(self):
        """element-wize variance."""
        return self.vars

    @property
    def std(self):
        return np.sqrt(self.vars)

    def scale(self, x, low=None, high=None):
        assert x.shape[-1] == self.mean.shape[0], "Unmatched shape in state scale."

        if low is None and high is None:
            return (x - self.mean) / (self.std + 1e-5)
        else:
            return np.clip((x - self.mean) / (self.std + 1e-5), low, high)
