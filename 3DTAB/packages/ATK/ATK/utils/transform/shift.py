# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import numpy as np


class PositiveShift(object):
    def __call__(self, data_dict):
        if "xyz" in data_dict.keys():
            xyz_min = np.min(data_dict["xyz"], 0)
            data_dict["xyz"] -= xyz_min
        return data_dict


class RandomShift(object):
    def __init__(self, shift=((-0.2, 0.2), (-0.2, 0.2), (0, 0))):
        self.shift = shift

    def __call__(self, data_dict):
        if "xyz" in data_dict.keys():
            shift_x = np.random.uniform(self.shift[0][0], self.shift[0][1])
            shift_y = np.random.uniform(self.shift[1][0], self.shift[1][1])
            shift_z = np.random.uniform(self.shift[2][0], self.shift[2][1])
            data_dict["xyz"] += [shift_x, shift_y, shift_z]
        return data_dict


