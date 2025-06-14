from typing import Any
from copy import copy

import torch
from pytorch3d.ops import estimate_pointcloud_normals
from qqdm import qqdm, format_str

from ATK.utils.common import return_first_item, check_option


class BasicAttack(object):
    def __init__(
        self, 
        model, 
        device: str = 'cuda', 
        budget: float = 0.,
        budget_type: str = 'none',
        renorm_result: bool = True,
        renorm_type: str = 'default'
    ):
        # NOTE: attack setting 
        self.model  = model
        self.device = device

        # NOTE: budget setting
        check_option(budget_type, ['none', 'linfty', 'point_linfty'])
        self.budget = budget
        self.budget_type = budget_type

        # NOTE: result setting
        check_option(renorm_type, ['default', 'pcs_normalization'])
        self.renorm_result = renorm_result
        self.renorm_type = renorm_type


    def set_task(
        self,
        model,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.device = device

    def set_budget(
        self, 
        budget: float = 0.,
        budget_type: str = 'none',
    ):
        check_option(budget_type, ['none', 'linfty', 'point_linfty'])
        self.budget = budget
        self.budget_type = budget_type

    def set_renorm(
        self,
        renorm_result: bool = True,
        renorm_type: str = 'default',
    ):
        check_option(renorm_type, ['default', 'pcs_normalization'])
        self.renorm_result = renorm_result
        self.renorm_type = renorm_type

    def attack(self, ori_pcs, labels, target=None, **kwargs) -> Any:
        return torch.zeros_like(ori_pcs).detach()

    def _proj(self, ori_pcs, delta, budget_type, budget, **kwargs):
        if budget_type == 'none':
            return delta

        if budget_type == 'linfty':
            return torch.clip(delta, -budget, budget)

        if budget_type == 'point_linfty':
            norm = delta.norm(p=2, dim=-1) # <B, N, 3>
            scale_factor = self.budget / (norm + 1e-9)  # <B, K>
            scale_factor = torch.clamp(scale_factor, max=1.)  # <B, K>
            return delta * scale_factor.unsqueeze(-1)

        raise NotImplemented

    def proj_delta(self, ori_pcs, delta, **kwargs):
        delta = self._proj(
            ori_pcs, delta, self.budget_type, self.budget, **kwargs
        )
        return delta

    def renorm_adv_pcs(self, ori_pcs, delta):
        adv_pcs  = ori_pcs + delta
        if self.renorm_type == 'default':
            adv_center = adv_pcs.mean(dim=1, keepdim=True)
            adv_pcs  = adv_pcs - adv_center
            adv_norm = adv_pcs.norm(dim=-1, p=2, keepdim=True)
            adv_pcs  = adv_pcs / (adv_norm + 1e-10)
            adv_norm = torch.clip(adv_norm, min=0, max=1.)
            adv_pcs  = adv_pcs * adv_norm
            return adv_pcs
        
        if self.renorm_type == 'pcs_normalization':
            adv_center = adv_pcs.mean(dim=1, keepdim=True)
            adv_pcs  = adv_pcs - adv_center
            adv_norm = adv_pcs.norm(dim=-1, p=2, keepdim=True)
            adv_pcs  = adv_pcs / (adv_norm + 1e-10)
            return adv_pcs


    def __call__(self, data_dict, target=None):

        self.model.to(self.device)
        self.model.eval()

        data_dict = copy(data_dict)
        pcs = data_dict.pop('xyz') # <B, N, 3>
        labels = data_dict.pop('category').view(-1) # <B,>

        ori_pcs, labels = self.to_device(pcs, labels, device=self.device)

        result = self.attack(ori_pcs, labels, target, **data_dict)
        delta  = return_first_item(result)
        delta  = self.proj_delta(ori_pcs, delta, normal=data_dict.get('normal', None))

        if self.renorm_result:
            adv_pcs = self.renorm_adv_pcs(ori_pcs, delta)
        else:
            adv_pcs = ori_pcs + delta

        if 'feat' in data_dict.keys():
            feat = data_dict.get('feat')
            feat[:, :, :3] = adv_pcs
        else:
            feat = adv_pcs

        _, N, _ = delta.shape
        offset = torch.ones_like(labels).unsqueeze(-1) * N

        data_dict['delta']    = delta
        data_dict['xyz']      = adv_pcs
        data_dict['feat']     = feat
        data_dict['offset']   = offset
        data_dict['category'] = labels

        return data_dict

    @staticmethod
    def get_pbar(iterator, attack_name):
        pbar = qqdm(
            iterator,
            desc=format_str('bold', attack_name),
            leave=False,
        ) 
        return pbar

    @staticmethod
    def estimate_pcs_normals(pcs, k=50):
        return -estimate_pointcloud_normals(pcs, k)

    def get_logits(self, pcs, **kwargs):
        pcs = pcs.to(self.device)
        feat = kwargs.get('feat', pcs)
        return self.model.__predict__(dict(xyz = pcs, feat = feat))


    def get_grad(self, loss, data, norm=None):
        grad = torch.autograd.grad(loss, [data])[0].detach()
        if norm is not None:
            grad = self.norm_grad(grad, norm=norm)
        return grad


    def init_delta(self, pcs, **kwargs):
        delta = torch.zeros_like(pcs).to(self.device)
        delta.requires_grad_(True)
        return delta

    @staticmethod
    def norm_grad(grad, norm=1):
        grad_norm = torch.norm(grad, p=norm, dim=(-1, -2), keepdim=True) # type: ignore
        grad      = grad / (grad_norm + 1e-20)
        return grad

    @staticmethod
    def center_grad(grad):
        grad = grad - grad.mean(dim=(-1, -2), keepdim=True)
        return grad

    @staticmethod
    def to_device(*data, device='cuda'):
        return [x.to(device) for x in data]


    @staticmethod
    def init_adv_pcs(ori_pcs, device):
        adv_pcs = ori_pcs.clone().detach()
        adv_pcs = adv_pcs.requires_grad_(True).to(device)
        return adv_pcs

    @staticmethod
    def clamp_adv_pcs(adv_pcs, ori_pcs, eps):
        delta   = adv_pcs - ori_pcs
        delta   = torch.clamp(delta, -eps, eps)
        adv_pcs = ori_pcs + delta
        return adv_pcs
