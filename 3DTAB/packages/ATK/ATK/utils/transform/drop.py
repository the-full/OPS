# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import random
import numpy as np


class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
        upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["xyz"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "xyz" in data_dict.keys():
                data_dict["xyz"] = data_dict["xyz"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
        return data_dict

