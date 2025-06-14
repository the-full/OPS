# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import numpy as np


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_dict):
        if np.random.rand() < self.p:
            if "xyz" in data_dict.keys():
                data_dict["xyz"][:, 0] = -data_dict["xyz"][:, 0]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 0] = -data_dict["normal"][:, 0]
        if np.random.rand() < self.p:
            if "xyz" in data_dict.keys():
                data_dict["xyz"][:, 1] = -data_dict["xyz"][:, 1]
            if "normal" in data_dict.keys():
                data_dict["normal"][:, 1] = -data_dict["normal"][:, 1]
        return data_dict

