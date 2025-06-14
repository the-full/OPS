import random
import functools

import torch
import numpy as np
from scipy.spatial.transform import Rotation

from ATK.utils.ops import farthest_point_sample

from .attack_template import SampleAttack


def identity(pcs):
    return pcs


class scaling():
    def __init__(self, scale) -> None:
        self.scale = scale
    
    def __call__(self, pcs):
        return pcs * self.scale


class rotate_x():
    def __init__(self, angle) -> None:
        self.angle = angle
        self.Rot = Rotation.from_euler('x', angle, degrees=True).as_matrix()

    def __call__(self, pcs):
        rot = torch.tensor(self.Rot).float().to(pcs.device)
        return torch.einsum('Cc, bnc -> bnC', rot, pcs)


class rotate_y():
    def __init__(self, angle) -> None:
        self.angle = angle
        self.Rot = Rotation.from_euler('y', angle, degrees=True).as_matrix()

    def __call__(self, pcs):
        rot = torch.tensor(self.Rot).float().to(pcs.device)
        return torch.einsum('Cc, bnc -> bnC', rot, pcs)


class rotate_z():
    def __init__(self, angle) -> None:
        self.angle = angle
        self.Rot = Rotation.from_euler('z', angle, degrees=True).as_matrix()

    def __call__(self, pcs):
        rot = torch.tensor(self.Rot).float().to(pcs.device)
        return torch.einsum('Cc, bnc -> bnC', rot, pcs)

class rotate():
    def __init__(self, angle_x=0., angle_y=0., angle_z=0.) -> None:
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        Rot_x = Rotation.from_euler('x', angle_x, degrees=True).as_matrix()
        Rot_y = Rotation.from_euler('y', angle_y, degrees=True).as_matrix()
        Rot_z = Rotation.from_euler('z', angle_z, degrees=True).as_matrix()
        self.Rot = Rot_x @ Rot_y @ Rot_z

    def __call__(self, pcs):
        rot = torch.tensor(self.Rot).float().to(pcs.device)
        return torch.einsum('Cc, bnc -> bnC', rot, pcs)

class drop_point():
    def __init__(self, drop_num, ratio=0.95) -> None:
        self.drop_num = drop_num
        self.drop_ratio = ratio

    def __call__(self, pcs):
        B, N, _ = pcs.shape
        _, drop_idx = farthest_point_sample(pcs, self.drop_num) # <B, K>
        mask = torch.zeros((B, N, 1), dtype=torch.bool).to(pcs.device)
        mask.scatter_(dim=1, index=drop_idx.unsqueeze(-1), value=True)
        drop_pcs = pcs.masked_fill(mask, 0)
        return torch.where(drop_pcs == 0, pcs * self.drop_ratio, pcs)

class shift():
    def __init__(self, shift_x, shift_y, shift_z) -> None:
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z
        self.shift_  = (shift_x, shift_y, shift_z)

    def __call__(self, pcs):
        shift_ = torch.tensor((self.shift_)).float().to(pcs.device)
        return pcs + shift_


class OPS(SampleAttack):
    ''' Operator-Perturbation-based Stochastic optimization (OPS). 
    '''
    def __init__(
        self, 
        model, 
        beta = 1.0,
        num_sample_neighbor = 1,
        sample_ratios = np.arange(0., 1.5, 0.25) + 0.25,
        num_sample_operator = 20,
        sample_levels = range(1, 5),
        **kwargs,
    ):
        super().__init__(
            model, 
            beta = beta,
            num_sample = num_sample_neighbor,
            **kwargs
        )
        self.num_sample_neighbor = num_sample_neighbor
        self.num_sample_operator = num_sample_operator
        self.using_sampling = (num_sample_operator * num_sample_neighbor > 0)

        if self.using_sampling:
            # NOTE: Operator Sampling
            self.num_sample_operator = num_sample_operator
            self.sample_levels = sample_levels
            self.op_list = []
            self.basic_ops = [
                identity, 
                drop_point(1, 0), drop_point(1, 0.3), drop_point(1, 0.6), drop_point(1, 0.9),
                drop_point(1, 1.2), drop_point(1, 1.5),
                scaling(0.8), scaling(0.85), scaling(0.9), scaling(0.95), 
                scaling(1.05), scaling(1.1), scaling(1.15),  
                rotate(0.5, -1.0, 0), rotate(0, 0, 0.5), rotate(-1.0, 0.5, 0), rotate(1.5, 0, -1.0), 
                rotate(1.5, 0, -0.5), rotate(1.0, 1.5, -0.5), rotate(-1.0, 1.5, 1.0),
                shift(0.02, 0.05, -0.04), shift(0.04, 0, 0), shift(0, 0.01, -0.05), shift(-0.04, 0.03, 0), 
                shift(-0.04, 0.03, -0.03), shift(0.03, 0.05, 0.01), shift(-0.03, 0, 0.05),
            ]
            self.num_extra_ops = len(self.basic_ops)

            # NOTE: Perturbation Sampling
            self.num_sample_neighbor = num_sample_neighbor
            sample_ratios = np.array(sample_ratios)
            self.sample_radius = beta * self.budget * sample_ratios
            self.eps_list = []
            self.num_extra_eps = self.num_sample_neighbor

    # NOTE: Operator Sampling
    @property
    def op_num(self):
        return len(self.op_list)

    def get_new_ops(self, k=2):
        sel_ops = random.choices(self.basic_ops, k=k)
        new_op = lambda x: x
        new_op = functools.reduce(lambda f, g: lambda x: f(g(x)), sel_ops, new_op)
        return new_op

    def expand_op_list(self, k=2):
        for _ in range(self.num_extra_ops):
            self.op_list.append(self.get_new_ops(k=k))

    def init_op_list(self):
        self.op_list = []
        for level in self.sample_levels:
            if level == 1:
                self.op_list.append(self.basic_ops.copy())
            else:
                self.expand_op_list(level)

    # NOTE: Perturbation Sampling
    @property
    def eps_num(self):
        return len(self.eps_list)

    def expand_eps_list(self, delta, radius=1.):
        shape = (self.num_extra_eps, *delta.shape[1:])
        noise = torch.zeros(shape).uniform_(-radius, radius).to(self.device)
        self.eps_list.extend(noise)

    def init_eps_list(self, delta):
        self.eps_list = []
        for radius in self.sample_radius:
            self.expand_eps_list(delta, radius)

    def get_surrogate_gradient(self, ori_pcs, delta, labels, target=None):
        logits = self.get_logits(ori_pcs + delta)
        loss   = self.get_loss(logits, labels, target=target)
        grad   = self.get_grad(loss, delta)
        return grad

    def get_sampled_grad(self, ori_pcs, delta, labels, target=None):
        averaged_gradient = self.get_surrogate_gradient(ori_pcs, delta, labels, target=target)
        if not self.using_sampling:
            return averaged_gradient

        selected_eps = random.sample(self.eps_list, min(self.num_sample_neighbor, self.eps_num))
        for eps in selected_eps:
            x_near = ori_pcs + delta + eps

            self.init_op_list()
            selected_ops = random.sample(self.op_list, min(self.num_sample_operator, self.op_num))
            for op in selected_ops:
                logits = self.get_logits(op(x_near))
                loss   = self.get_loss(logits, labels, target=target)
                grad   = self.get_grad(loss, delta)
                averaged_gradient += grad

        return averaged_gradient / (self.num_sample_neighbor * self.num_sample_operator + 1)

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        if self.using_sampling:
            self.init_eps_list(delta)
        del delta
        return super().attack(ori_pcs, labels, target=target, **kwargs)
