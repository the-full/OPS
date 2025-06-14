import torch
import torch.nn as nn


# NOTE: modify from AOF
class ClipPointsLinf(nn.Module):

    def __init__(self, budget):
        """Clip point cloud with a given l_inf budget.

        Args:
            budget (float): perturbation budget
        """
        super(ClipPointsLinf, self).__init__()

        self.budget = budget

    def forward(self, adv_pcs, ori_pcs):
        """Clipping every point in a point cloud.

        Args:
            adv_pcs (torch.FloatTensor): batch pc, [B, K, 3]
            ori_pc (torch.FloatTensor): original point cloud
        """
        diff = adv_pcs - ori_pcs  # <B, K, 3>
        norm = diff.norm(dim=-1) # <B, K>
        scale_factor = self.budget / (norm + 1e-9)  # <B, K>
        scale_factor = torch.clamp(scale_factor, max=1.)  # <B, K>
        diff = diff * scale_factor.unsqueeze(-1) # <B, K, 3>
        adv_pcs = ori_pcs + diff # <B, K, 3>
        return adv_pcs


class ProjectInnerPoints(nn.Module):

    def __init__(self):
        """Eliminate points shifted inside an object.
        Introduced by AAAI'20 paper.
        """
        super(ProjectInnerPoints, self).__init__()

    def forward(self, adv_pcs, ori_pcs, normal=None):
        """Clipping "inside" points to the surface of the object.

        Args:
            adv_pcs (torch.FloatTensor): batch pc, [B, K, 3]
            ori_pc (torch.FloatTensor): original point cloud
            normal (torch.FloatTensor, optional): normals. Defaults to None.
        """
        if normal is None:
            return adv_pcs

        diff = adv_pcs - ori_pcs # <B, K, 3>
        inner_diff_normal = torch.sum(diff * normal, dim=-1)  # <B, K>
        inner_mask = (inner_diff_normal < 0.)  # <B, K>

        # clip to surface!
        # 1) vng = Normal x Perturb
        vng = torch.cross(normal, diff)  # <B, K, 3>
        vng_norm = vng.norm(dim=-1) # <B, K>

        # 2) vref = vng x Normal
        vref = torch.cross(vng, normal)  # <B, K, 3>
        vref_norm = vref.norm(dim=-1) # <B, K>

        # 3) Project Perturb onto vref
        diff_proj = diff * vref / (vref_norm.unsqueeze(-1) + 1e-9)  # <B, K, 3>

        # some diff is completely opposite to normal
        # just set them to (0, 0, 0)
        opposite_mask = inner_mask & (vng_norm < 1e-6) # <B, K>
        opposite_mask = opposite_mask.unsqueeze(-1).expand_as(diff_proj) # <B, K, 3>
        diff_proj[opposite_mask] = 0.

        # set inner points with projected perturbation
        inner_mask = inner_mask.unsqueeze(-1).expand_as(diff) # <B, K, 3>
        diff[inner_mask] = diff_proj[inner_mask]
        adv_pcs = ori_pcs + diff
        return adv_pcs


class ProjectInnerClipLinf(nn.Module):

    def __init__(self, budget):
        """Project inner points to the surface and
        clip the l_inf norm of perturbation.

        Args:
            budget (float): l_inf norm budget
        """
        super(ProjectInnerClipLinf, self).__init__()

        self.project_inner = ProjectInnerPoints()
        self.clip_linf = ClipPointsLinf(budget=budget)

    def forward(self, ori_pcs, adv_pcs, normal=None):
        """Project to the surface and then clip.

        Args:
            ori_pcs (torch.FloatTensor): original point cloud
            adv_pcs (torch.FloatTensor): batch pc, [B, K, 3]
            normal (torch.FloatTensor, optional): normals. Defaults to None.
        """
        # project
        adv_pcs = self.project_inner(adv_pcs, ori_pcs, normal)
        # clip
        adv_pcs = self.clip_linf(adv_pcs, ori_pcs)
        return adv_pcs

