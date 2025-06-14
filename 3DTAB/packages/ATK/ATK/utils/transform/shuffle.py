# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import numpy as np


class ShufflePoint(object):
    def __call__(self, data_dict):
        assert "xyz" in data_dict.keys()
        shuffle_index = np.arange(data_dict["xyz"].shape[0])
        np.random.shuffle(shuffle_index)
        if "xyz" in data_dict.keys():
            data_dict["xyz"] = data_dict["xyz"][shuffle_index]
        if "grid_coord" in data_dict.keys():
            data_dict["grid_coord"] = data_dict["grid_coord"][shuffle_index]
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"][shuffle_index]
        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"][shuffle_index]
        return data_dict

