# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import numpy as np

import ipdb
class NormalizeCoord(object):
    def __call__(self, data_dict):
        if "xyz" in data_dict.keys():
            centroid = np.mean(data_dict["xyz"], axis=0)
            data_dict["xyz"] -= centroid
            m = np.max(np.sqrt(np.sum(data_dict["xyz"]**2, axis=1)))
            data_dict["xyz"] = data_dict["xyz"] / m
        return data_dict

