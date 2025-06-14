# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import random
import warnings

import numpy as np

from .rot_utils import *


class RandomRotate(object):
    def __init__(self, angle=(-1, 1), center=None, axis="z", always_apply=False, p=0.5):
        assert len(angle) == 2, f"angle should be a tuple of 2 values, but got{angle}"
        if angle[0] > angle[1]:
            raise ValueError(f"angle[0] should be less than angle[1] but got {angle}")
        if angle[0] < -1 or angle[1] > 1:
            message = f"{angle} is out of [-1, 1], please check your input"
            warnings.warn(message, category=UserWarning)

        self.angle = angle
        
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center


    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == "x":
            rot_t = np.array([[1,       0,        0], 
                              [0, rot_cos, -rot_sin], 
                              [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos,  0, rot_sin], 
                              [      0,  1,       0], 
                              [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin,  0], 
                              [rot_sin,  rot_cos,  0], 
                              [      0,        0,  1]])
        else:
            raise NotImplementedError
        if "xyz" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["xyz"].min(axis=0)
                x_max, y_max, z_max = data_dict["xyz"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["xyz"] -= center
            data_dict["xyz"] = np.dot(data_dict["xyz"], np.transpose(rot_t))
            data_dict["xyz"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict


class RandomRotateTargetAngle(object):
    def __init__(
        self, angle=(1 / 2, 1, 3 / 2), center=None, axis="z", always_apply=False, p=0.75
    ):
        if np.any(angle < -1) or np.any(angle > 1):
            message = f"there exist some angle is out of [-1, 1], please check your input"
            warnings.warn(message, category=UserWarning)
        self.angle = angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.choice(self.angle) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)

        if self.axis == "x":
            rot_t = np.array([[1,       0,        0], 
                              [0, rot_cos, -rot_sin], 
                              [0, rot_sin, rot_cos]])
        elif self.axis == "y":
            rot_t = np.array([[rot_cos,  0, rot_sin], 
                              [      0,  1,       0], 
                              [-rot_sin, 0, rot_cos]])
        elif self.axis == "z":
            rot_t = np.array([[rot_cos, -rot_sin,  0], 
                              [rot_sin,  rot_cos,  0], 
                              [      0,        0,  1]])
        else:
            raise NotImplementedError

        if "xyz" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["xyz"].min(axis=0)
                x_max, y_max, z_max = data_dict["xyz"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["xyz"] -= center
            data_dict["xyz"] = np.dot(data_dict["xyz"], np.transpose(rot_t))
            data_dict["xyz"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict
