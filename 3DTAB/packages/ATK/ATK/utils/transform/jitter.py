# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import numpy as np


class RandomJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        assert clip > 0
        self.sigma = sigma
        self.clip = clip

    def __call__(self, data_dict):
        if "xyz" in data_dict.keys():
            jitter = np.clip(
                self.sigma * np.random.randn(data_dict["xyz"].shape[0], 3),
                -self.clip,
                self.clip,
            )
            data_dict["xyz"] += jitter
        return data_dict


class ClipGaussianJitter(object):
    def __init__(self, scalar=0.02, store_jitter=False):
        self.scalar = scalar
        self.mean = np.mean(3)
        self.cov = np.identity(3)
        self.quantile = 1.96
        self.store_jitter = store_jitter

    def __call__(self, data_dict):
        if "xyz" in data_dict.keys():
            jitter = np.random.multivariate_normal(
                self.mean, self.cov, data_dict["xyz"].shape[0]
            )
            jitter = self.scalar * np.clip(jitter / 1.96, -1, 1)
            data_dict["xyz"] += jitter
            if self.store_jitter:
                data_dict["jitter"] = jitter
        return data_dict

