import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np
from pytorch3d.ops import knn_points

from .ops import ball_query, farthest_point_sample, get_pcs_by_idx
from .metric import (
    l2_norm_distance,
    chamfer_distance,
    hausdorff_distance,
    curvature_diff,
    curvature_distance,
    curvature_std_distance,
    displacement_diff,
    repulsion_index,
    knn_outlier_distance,
    knn_mean_distance,
    uniform_index,
)


def to_tensor(data) -> torch.Tensor:
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    return data




#  |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣>
#  |         special loss            >
#  |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿>
#              (\__/) ||
#              (•ㅅ•) ||
#              / 　 づ
class NoneLoss(nn.Module):
    def __init__(self):
        super(NoneLoss, self).__init__()

    def forward(self, input_data: torch.FloatTensor) -> torch.Tensor:
        device = input_data.device
        B = input_data.shape[0]
        loss = torch.zeros(B).long().to(device)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.objectives = {}
        self.weights = {}

    def add_objective(self, name: str, objective: nn.Module, weight: float):
        self.objectives[name] = objective
        self.weights[name] = weight

    def forward(self, loss_info=None, **kwargs):
        if loss_info is None:
            loss_info = types.SimpleNamespace(loss=0.)

        for name, args in kwargs.items():
            objective = self.objectives.get(name, None)
            weight = self.weights[name]
            if objective is not None:
                if weight == 0.:
                    loss = torch.tensor(0.).float()
                else:
                    loss = objective(*args)
                    loss_info.loss += weight * loss
                setattr(loss_info, name, loss)
            else:
                raise ValueError(f"No arguments provided for loss '{name}'")

        if isinstance(loss_info.loss, float):
            loss_info.loss = torch.tensor(0.).float()
        return loss_info


#  |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣>
#  |        distance loss            >
#  |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿>
#              (\__/) ||
#              (•ㅅ•) ||
#              / 　 づ
class MeanSquareLoss(nn.Module):
    def __init__(self):
        super(MeanSquareLoss, self).__init__()

    def forward(self, adv_pcs: torch.FloatTensor, ori_pcs: torch.FloatTensor) -> torch.Tensor:
        return torch.square(adv_pcs - ori_pcs).sum(-1).mean(-1)


class L2NormLoss(nn.Module):
    def __init__(self):
        super(L2NormLoss, self).__init__()

    def forward(self, adv_pcs: torch.FloatTensor, ori_pcs: torch.FloatTensor) -> torch.Tensor:
        return l2_norm_distance(adv_pcs, ori_pcs)


class ChamferLoss(nn.Module):
    def __init__(self, single=False, reduce='max'):
        super(ChamferLoss, self).__init__()
        self.single = single
        self.reduce = reduce

    def forward(
        self,
        ori_pcs: torch.Tensor,
        adv_pcs: torch.Tensor,
        knn_idx: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        if knn_idx is None:
            return chamfer_distance(ori_pcs, adv_pcs, single=self.single, reduce=self.reduce)
        else:
            ori_pcs = torch.gather(ori_pcs, dim=-1, index=knn_idx)
            adv_pcs = torch.gather(adv_pcs, dim=-1, index=knn_idx)
            return chamfer_distance(ori_pcs, adv_pcs, single=self.single, reduce=self.reduce)


class HausdorffLoss(nn.Module):
    def __init__(self, single=False, reduce='max'):
        super(HausdorffLoss, self).__init__()
        self.single = single
        self.reduce = reduce

    def forward(
        self,
        ori_pcs: torch.Tensor,
        adv_pcs: torch.Tensor,
        knn_idx: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if knn_idx is None:
            return hausdorff_distance(ori_pcs, adv_pcs, single=self.single, reduce=self.reduce)
        else:
            ori_pcs = torch.gather(ori_pcs, dim=-1, index=knn_idx)
            adv_pcs = torch.gather(adv_pcs, dim=-1, index=knn_idx)
            return hausdorff_distance(ori_pcs, adv_pcs, single=self.single, reduce=self.reduce)


class CurvatureLoss(nn.Module):
    def __init__(self, k=2):
        super(CurvatureLoss, self).__init__()
        self.k = k

    def forward(
        self,
        ori_pcs: torch.FloatTensor,
        adv_pcs: torch.FloatTensor,
        ori_normals: torch.FloatTensor,
    ) -> torch.Tensor:
        return curvature_distance(ori_pcs, adv_pcs, ori_normals)


class CurvatureStdLoss(nn.Module):
    def __init__(self, k=2):
        super(CurvatureStdLoss, self).__init__()
        self.k = k

    def forward(
        self,
        ori_pcs: torch.FloatTensor,
        adv_pcs: torch.FloatTensor,
        ori_normals: torch.FloatTensor,
    ) -> torch.Tensor:
        return curvature_std_distance(ori_pcs, adv_pcs, ori_normals, self.k)


class CurvatureStdLossV2(nn.Module):
    def __init__(self):
        super(CurvatureStdLossV2, self).__init__()

    def forward(
        self,
        ori_pcs: torch.FloatTensor,
        adv_pcs: torch.FloatTensor,
        ori_curve: torch.FloatTensor,
        adv_curve: torch.FloatTensor,
    ) -> torch.Tensor:
        return curvature_diff(ori_pcs, adv_pcs, ori_curve, adv_curve)


class DisplacementLoss(nn.Module):
    def __init__(self, k: int = 16):
        super(DisplacementLoss, self).__init__()
        self.k = k

    def forward(
        self, 
        adv_pcs: torch.Tensor, 
        ori_pcs: torch.Tensor
    ) -> torch.Tensor:
        return displacement_diff(adv_pcs, ori_pcs, self.k)


class RepulsionLoss(nn.Module):
    def __init__(self, k: int = 4, h: float = 0.03):
        super(RepulsionLoss, self).__init__()
        self.k = k
        self.h = h

    def forward(self, pcs: torch.Tensor) -> torch.Tensor:
        return repulsion_index(pcs, self.k, self.h)


class DistanceKMeanLoss(nn.Module):
    def __init__(self, k: int):
        super(DistanceKMeanLoss, self).__init__()
        self.k = k

    def forward(self, pcs: torch.Tensor) -> torch.Tensor:
        return knn_mean_distance(pcs, k=self.k)


class KNNSmoothingLoss(nn.Module):
    """
    Ref: 
        - Robust Adversarial Objects against Deep Learning Models (AAAI 2020)
        - https://ojs.aaai.org/index.php/AAAI/article/download/5443/5299j
        - https://github.com/jinyier/ai_pointnet_attack/blob/master/attack.py#L206-227
    """

    def __init__(self, k: int, alpha: float = 1.05):
        super(KNNSmoothingLoss, self).__init__()
        self.k = k
        self.alpha = alpha

    def forward(self, pcs: torch.Tensor) -> torch.Tensor:
        return knn_outlier_distance(pcs, k=self.k, alpha=self.alpha)


class UniformLoss(torch.nn.Module):
    def __init__(self,
                 percentages: list = [0.004, 0.006, 0.008, 0.010, 0.012],
                 radius: float = 1.0,
                 shape: str = 'square'):
        super(UniformLoss, self).__init__()
        self.percentages = percentages
        self.radius = radius
        self.shape = shape

    def forward(self, adv_pcs: torch.FloatTensor) -> torch.Tensor:
        device   = adv_pcs.device
        B, N, _  = adv_pcs.size()
        seed_num = int(N * 0.05)

        loss = torch.tensor(0.).float().to(device)
        for p in self.percentages:
            exp_num = N * p
            r = np.sqrt(p * self.radius * self.radius)

            if self.shape == 'square':
                expect_dis = (np.pi * (self.radius ** 2)) / exp_num * p
            elif self.shape == 'hexagon':
                expect_dis = (2 * np.pi * (self.radius ** 2)) / (np.sqrt(3) * exp_num) * p
            else:
                raise ValueError('Unsupported shape: {}'.format(self.shape))
            expect_dis = torch.sqrt(torch.Tensor([expect_dis])).to(device)

            sub_pcs = get_pcs_by_idx(adv_pcs, farthest_point_sample(adv_pcs, seed_num)[1]) # <B, seed_num, 3>
            grouped_pcs, grouped_idx = ball_query(sub_pcs, adv_pcs, r, int(2 * exp_num), use_nearest=False) # <B, seed_num, 2 * exp_num, 3>
            grouped_pcs = rearrange(grouped_pcs, 'B seed_num exp_num_2 C -> (B seed_num) exp_num_2 C')
            grouped_idx = rearrange(grouped_idx, 'B seed_num exp_num_2   -> (B seed_num) exp_num_2')

            true_nn_mask = (grouped_idx != -1) # <B * seed_num, 2 * exp_num>
            grouped_num  = torch.sum(true_nn_mask, dim=-1) # <B * seed_num>

            grouped_nn_dis = knn_points(grouped_pcs, grouped_pcs, lengths1=grouped_num, K=2).dists # <B * seed_num, 2 * exp_num, 2>
            grouped_nn_dis = grouped_nn_dis[:, :, 1:].contiguous() # <B * seed_num, 2 * exp_num, 1>
            grouped_nn_dis = torch.sqrt(grouped_nn_dis).squeeze(-1) # <B * seed_num, 2 * exp_num>

            uniform_clutter = torch.square(grouped_nn_dis - expect_dis) / (expect_dis + 1e-12) # <B * seed_num, 2 * exp_num>
            uniform_clutter = torch.sum(uniform_clutter * true_nn_mask) / grouped_num # <B * seed_num>

            uniform_imbalance = torch.square(grouped_num - exp_num) / exp_num # <B * seed_num>

            uniform_loss = uniform_clutter * uniform_imbalance # <B * seed_num>
            uniform_loss = rearrange(uniform_loss, '(B seed_num) -> B seed_num', B=B) # <B, seed_num>
            uniform_loss = torch.mean(uniform_loss, dim=-1) # <B,>

            loss += uniform_loss

        return loss / len(self.percentages)


class UniformLossGeoA3(torch.nn.Module):
    def __init__(
        self,
        percentages: list = [0.004, 0.006, 0.008, 0.010, 0.012],
        radius: float = 1.0,
        shape: str = 'square'
    ):
        super(UniformLossGeoA3, self).__init__()
        self.percentages = percentages
        self.radius = radius
        self.shape = shape

    def forward(self, adv_pcs: torch.FloatTensor) -> torch.Tensor:
        device = adv_pcs.device
        B, N, _ = adv_pcs.size()
        npoint = int(N * 0.05)

        loss = torch.tensor(0.).float().to(device)
        for p in self.percentages:
            p = p * 4
            nsample = int(N * p)
            r = np.sqrt(p * self.radius)

            if self.shape == 'square':
                expect_dis = (np.pi * (self.radius ** 2)) / nsample * p
            elif self.shape == 'hexagon':
                expect_dis = (2 * np.pi * (self.radius ** 2)) / (np.sqrt(3) * nsample) * p
            else:
                raise ValueError('Unsupported shape: {}'.format(self.shape))
            expect_dis = torch.sqrt(torch.Tensor([expect_dis])).to(device)

            sub_pcs = adv_pcs[farthest_point_sample(adv_pcs, npoint)[1]]  # <B, npoint, 3>
            grouped_pcs, _ = ball_query(sub_pcs, adv_pcs, r, nsample)  # <B, npoint, nsample, 3>
            grouped_pcs = grouped_pcs.permute(0, 2, 1, 3).contiguous().view(B * npoint, nsample, 3)  # <B * npoint, nsample, 3>

            grouped_nn_dis = knn_points(grouped_pcs, grouped_pcs, K=2).dists  # <B * npoint, nsample, 2>
            grouped_nn_dis = grouped_nn_dis[:, :, 1:].contiguous()  # <B * npoint, nsample, 1>
            grouped_nn_dis = torch.sqrt(grouped_nn_dis).squeeze(-1)  # <B * npoint, nsample>

            uniform_clutter = torch.square(grouped_nn_dis - expect_dis) / (expect_dis + 1e-12)  # <B * npoint, nsample>
            uniform_clutter = uniform_clutter.view(B, npoint * nsample)  # <B, npoint * nsample>
            uniform_dis_mean = uniform_clutter.mean(dim=-1)  # <B,>
            uniform_dis_mean = uniform_dis_mean * (p * 100) ** 2  # <B,>

            loss += uniform_dis_mean

        return loss / len(self.percentages)



#  |￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣>
#  |        adversarial loss         >
#  |＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿>
#              (\__/) ||
#              (•ㅅ•) ||
#              / 　 づ
class CWAdvLoss(nn.Module):
    def __init__(self, kappa=0.):
        super(CWAdvLoss, self).__init__()
        self.kappa = kappa

    def get_m_logits(self, logits, one_hot):
        m_logits = torch.max(logits - one_hot * 1e10, dim=1).values # <B,>
        return m_logits

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        # m: other most likely, g: gt, t: target
        B, N = logits.shape

        if target is None:
            labels_one_hot = F.one_hot(labels, num_classes=N)
            g_logits = torch.sum(logits * labels_one_hot, dim=-1)
            m_logits = self.get_m_logits(logits, labels_one_hot)
            loss = torch.relu(g_logits - m_logits + self.kappa)
        else:
            device = logits.device
            target_tensor = to_tensor(target).long().to(device)

            targets_one_hot = F.one_hot(target_tensor, num_classes=N).to(device)
            targets_one_hot = repeat(targets_one_hot, 'cls_num -> B cls_num', B=B)
            t_logits = torch.sum(logits * targets_one_hot, dim=-1) # <B,>
            m_logits = self.get_m_logits(logits, targets_one_hot)
            loss = torch.relu(m_logits - t_logits + self.kappa)
        return loss


class MLCAdvLoss(nn.Module):
    def __init__(self,
                 kappa=0.):
        super(MLCAdvLoss, self).__init__()
        self.kappa    = kappa
        self.m_labels = None

    def fix_mlc(self, ori_logits, labels):
        labels_one_hot = F.one_hot(labels, num_classes=ori_logits.shape[-1])
        self.m_labels = torch.argmax(ori_logits - labels_one_hot * 1e10, dim=-1) # <B,>

    def get_m_logits(self, logits, one_hot):
        if self.m_labels is not None:
            m_logits = torch.gather(logits, dim=-1, index=self.m_labels) # <B,>
        else:
            m_logits = torch.max(logits - one_hot * 1e10, dim=1).values # <B,>
        return m_logits

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        # m: other most likely, g: gt, t: target
        B, N = logits.shape

        if target is None:
            labels_one_hot = F.one_hot(labels, num_classes=N)
            g_logits = torch.sum(logits * labels_one_hot, dim=-1)
            m_logits = self.get_m_logits(logits, labels_one_hot)
            loss = torch.relu(g_logits - m_logits + self.kappa) - self.kappa
        else:
            device = logits.device
            target_tensor = to_tensor(target).long().to(device)

            targets_one_hot = F.one_hot(target_tensor, num_classes=N).to(device)
            targets_one_hot = repeat(targets_one_hot, 'cls_num -> B cls_num', B=B)
            t_logits = torch.sum(logits * targets_one_hot, dim=-1) # <B,>
            m_logits = self.get_m_logits(logits, targets_one_hot)
            loss = torch.relu(m_logits - t_logits + self.kappa) - self.kappa
        return loss


class LLCAdvLoss(nn.Module):
    def __init__(self, kappa=0.):
        super(LLCAdvLoss, self).__init__()
        self.kappa    = kappa
        self.l_labels = None

    def fix_llc(self, ori_logits, labels):
        labels_one_hot = F.one_hot(labels, num_classes=ori_logits.shape[-1])
        self.l_labels = torch.argmin(ori_logits + labels_one_hot * 1e10, dim=-1) # <B,>

    def get_l_logits(self, logits, one_hot):
        if self.l_labels is not None:
            l_logits = torch.gather(logits, dim=-1, index=self.l_labels) # <B,>
        else:
            l_logits = torch.min(logits + one_hot * 1e10, dim=1).values # <B,>
        return l_logits

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        # l: other least likely, g: gt, t: target
        B, N = logits.shape
        if target is None:
            labels_one_hot = F.one_hot(labels, num_classes=N)
            g_logits = torch.sum(logits * labels_one_hot, dim=-1)
            l_logits = self.get_l_logits(logits, labels_one_hot)
            loss = torch.relu(g_logits - l_logits + self.kappa) - self.kappa
        else:
            device = logits.device
            target_tensor = to_tensor(target).long().to(device)

            labels_one_hot = F.one_hot(target_tensor, num_classes=N) # <B, N>
            target_one_hot = F.one_hot(target_tensor, num_classes=N).to(device) # <N>
            target_one_hot = repeat(target_one_hot, 'cls_num -> B cls_num', B=B)
            g_logits = torch.sum(logits * labels_one_hot, dim=-1) # <B,>
            t_logits = torch.sum(logits * target_one_hot, dim=-1) # <B,>
            loss = torch.relu(g_logits - t_logits + self.kappa) - self.kappa
        return loss


class NegtiveAdvLoss(nn.Module):
    def __init__(self, loss_fn=None):
        super(NegtiveAdvLoss, self).__init__()
        if loss_fn is None:
            self.Loss = nn.CrossEntropyLoss(reduction='none')
        else:
            self.Loss = loss_fn

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        if target is None:
            loss = -self.Loss(logits, labels)
        else:
            B = logits.shape[0]
            device = logits.device

            target_tensor = to_tensor(target).long().to(device)
            target_tensor = repeat(target_tensor.view(-1), '1 -> B', B=B)
            loss = self.Loss(logits, target_tensor)
        return loss


class LogitsAdvLoss(nn.Module):
    """
    Ref:
        - https://blog.csdn.net/AITIME_HY/article/details/124335650
    """
    def __init__(self, use_gt=False):
        super(LogitsAdvLoss, self).__init__()
        self.use_gt = use_gt

    def forward(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        B = logits.shape[0]
        batch_idx = torch.arange(B)
        if target is None:
            loss = logits[batch_idx, labels] # <B,>
        else:
            device = logits.device
            target_tensor = to_tensor(target).long().to(device)
            batch_target  = repeat(target_tensor.view(-1), '1 -> B', B=B)

            if self.use_gt:
                loss =  logits[batch_idx, labels] - logits[batch_idx, batch_target]
            else:
                loss = -logits[batch_idx, batch_target]
        return loss
