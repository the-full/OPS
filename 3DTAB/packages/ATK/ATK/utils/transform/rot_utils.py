# @Cite: https://github.com/Pointcept/Pointcept
# @Ori_author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)

import numpy as np

def rotation_by_x(pc, angle):
    one_, zero = np.ones(pc.shape[0]), np.zeros(pc.shape[0])
    sin_, cos_ = np.sin(angle), np.cos(angle)

    R = np.array([
        [one_, zero,  zero],
        [zero, cos_, -sin_],
        [zero, sin_,  cos_],
    ])

    return pc @ R.T

def rotation_by_y(pc, angle):
    one_, zero = np.ones(pc.shape[0]), np.zeros(pc.shape[0])
    sin_, cos_ = np.sin(angle), np.cos(angle)

    R = np.array([
        [ cos_, zero, sin_],
        [ zero, one_, zero],
        [-sin_, zero, cos_],
    ])

    return pc @ R.T

def rotation_by_z(pc, angle):
    one_, zero = np.ones(pc.shape[0]), np.zeros(pc.shape[0])
    sin_, cos_ = np.sin(angle), np.cos(angle)

    R = np.array([
        [cos_, -sin_, zero],
        [sin_,  cos_, zero],
        [zero,  zero, one_],
    ])

    return pc @ R.T

def rotation_by_xyz(pc, x_angle, y_angle, z_angle):
    pc = rotation_by_x(pc, x_angle)
    pc = rotation_by_y(pc, y_angle)
    pc = rotation_by_z(pc, z_angle)
    return pc
