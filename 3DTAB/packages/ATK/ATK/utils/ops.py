from collections import namedtuple
from typing import Optional

import torch
import numpy as np
import pytorch3d.ops as pops
from einops import rearrange, repeat


def BCN_to_BNC(x):
    return rearrange(x, 'B C N -> B N C')


def BNC_to_BCN(x):
    return rearrange(x, 'B C N -> B N C')


def batch_minmax_normalize(input_tensor):
    min_values = input_tensor.min(dim=0, keepdim=True)[0] # <B, C, ...>
    max_values = input_tensor.max(dim=0, keepdim=True)[0] # <B, C, ...>

    eps = 1e-8
    normalized_tensor = (input_tensor - min_values) / (max_values - min_values + eps) # <B, C, ...>

    return normalized_tensor

def minmax_normalize(input_tensor, lower_bound=None, upper_bound=None):
    if lower_bound is None:
        lower_bound = input_tensor.min()
    if upper_bound is None:
        upper_bound = input_tensor.max()
    return (input_tensor - lower_bound) / (upper_bound - lower_bound + 1e-8)


def pcs_normalize(pcs):
    """
    pcs <B, N, 3>
    """
    pcs = pcs - pcs.mean(1, keepdim=True) # <B, N, 3>
    m = torch.norm(pcs, dim=-1).max() # <B,>
    pcs = pcs / (m + 1e-12)
    return pcs


def random_point_sample(pcs, k=128):
    B, N, _ = pcs.size()
    if k > N:
        raise ValueError("num_pts cannot be greater than N")

    permuted_indices = torch.randperm(N, device=pcs.device).unsqueeze(0).repeat(B, 1)
    for b in range(B):
        permuted_indices[b] = torch.randperm(N, device=pcs.device)

    sampled_indices = permuted_indices[:, :k]
    return sampled_indices


def farthest_point_sample(pcs, k=128):
    return pops.sample_farthest_points(pcs, K=k)


def knn_points(query_pcs, source_pcs, k):
    """
    Ref:
        https://github.com/pytorch/pytorch/pull/2775
    """
    _, idx, points = pops.knn_points(query_pcs, source_pcs, K=k, return_nn=True)
    idx = idx.squeeze(-1)
    return points, idx


def knn_gather(pcs, knn_idx):
    return pops.knn_gather(pcs, knn_idx)


def ball_query(query_pcs, source_pcs, radius, k, use_nearest=True):
    _, N, _ = source_pcs.shape
    _, idx, points = pops.ball_query(query_pcs, source_pcs, K=k, radius=radius)
    if use_nearest:
        idx = torch.where(idx == -1, N, idx)
        idx = idx.sort(dim=-1)[0][:, :, :k]
        nearest_idx = repeat(idx[:, :, 0], 'B Q -> B Q K', K=k)
        idx = torch.where(idx == N, nearest_idx, idx)
        points = pops.knn_gather(source_pcs, idx)
    return points, idx


def get_knn_idx(pcs, idx, k=10):
    idx     = repeat(idx, 'b m -> b m c', c=3) # <B, M, 3>
    sel_pcs = torch.gather(pcs, dim=1, index=idx) # <B, M, 3>
    knn_idx = pops.knn_points(sel_pcs, pcs, K=k+1).idx # <B, M, k+1>
    return knn_idx


def get_pcs_by_idx(pcs, idx):
    B, K = idx.shape
    batch_idx = torch.arange(B, dtype=torch.long).to(pcs.device)
    batch_idx = repeat(batch_idx, 'B -> B K', K=K)
    sel_pcs = pcs[batch_idx, idx, :]
    return sel_pcs


def get_graph_feature(data, idx=None, k=10):
    """
    Ref:
        https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
    """
    data = rearrange(data, 'B C N -> B N C')
    if idx is not None:
        knn_idx = idx
    else:
        knn_idx = pops.knn_points(data, data, K=k).idx # <B, N, k>
    feature = pops.knn_gather(data, knn_idx) # <B, N, k, C>
    data    = repeat(data, 'B N C -> B N k C', k=k) # <B, N, k, C>
    feature = torch.cat((feature-data, data), dim=-1) # <B, N, k, 2*C>
    feature = rearrange(feature, 'B N k C -> B C N k').contiguous() # <B, 2*C, N, k>
    return feature


SGReturn = namedtuple('SGReturn', ['fps_xyz', 'grouped_features', 'fps_idx', 'grouped_idx'])
def sample_and_group_knn(
    npoint: int,
    nsample: int,
    xyz: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    return_idx: bool = False
) -> SGReturn:
    """
    Ref:
        https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py#L120-L148
    """
    fps_xyz, fps_idx = farthest_point_sample(xyz, npoint) # [<B, S, C>, <B, S>] # S = npoint
    grouped_xyz, idx = knn_points(fps_xyz, xyz, k=nsample) # [<B, S, K, C>, <B, S, K>] # K = nsample
    grouped_xyz = grouped_xyz - fps_xyz.unsqueeze(-2) # <B, S, K, C>

    if features is not None:
        grouped_features = knn_gather(features, idx) # <B, S, K, D>
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1) # <B, S, K, C+D>
    else:
        grouped_features = grouped_xyz # <B, S, K, C>

    if return_idx:
        return SGReturn(fps_xyz, grouped_features, fps_idx, idx)
    else:
        return SGReturn(fps_xyz, grouped_features, None, None)


def sample_and_group_ball_query(
    npoint: int,
    nsample: int,
    radius: float,
    xyz: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    return_idx: bool = False
):
    """
    Ref:
        https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py#L110-L138
    """
    fps_xyz, fps_idx = farthest_point_sample(xyz, npoint) # [<B, S, C>, <B, S>], S = npoint
    grouped_xyz, idx = ball_query(query_pcs=fps_xyz,
                                  source_pcs=xyz,
                                  radius=radius,
                                  k=nsample,
                                  use_nearest=True) # [<B, S, K, C>, <B, S, K>, K = nsample
    grouped_xyz = grouped_xyz - fps_xyz.unsqueeze(-2) # <B, S, K, C>

    if features is not None:
        grouped_features = knn_gather(features, idx) # <B, S, K, D>
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1) # <B, S, K, C+D>
    else:
        grouped_features = grouped_xyz # <B, S, K, C>

    if return_idx:
        return SGReturn(fps_xyz, grouped_features, fps_idx, idx)
    else:
        return SGReturn(fps_xyz, grouped_features, None, None)


SGAllReturn = namedtuple('SGAllReturn', ['zero_xyz', 'grouped_features'])
def sample_and_group_all(
    xyz: torch.Tensor,
    features: Optional[torch.Tensor] = None
) -> SGAllReturn:
    """
    Ref:
        https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py#L141-L158
        https://github.com/charlesq34/pointnet2/blob/master/utils/pointnet_util.py#L59-L84
    """

    device = xyz.device
    B, _, C = xyz.shape
    zero_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.unsqueeze(1) # <B, 1, N, C>
    if features is not None:
        grouped_features = features.unsqueeze(1) # <B, 1, N, D>
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1) # <B, 1, N, C+D>
    else:
        grouped_features = grouped_xyz # <B, 1, N, C>
    return SGAllReturn(zero_xyz, grouped_features)


def estimate_normal_pytorch3d(pcs, k=50):
    return -pops.estimate_pointcloud_normals(pcs, k)


def find_offset(ori_pcs: torch.Tensor, adv_pcs: torch.Tensor) -> torch.Tensor:
    knn_idx = pops.knn_points(adv_pcs, ori_pcs, K=1).idx  # <B, N, 1, C>, C=3
    adv_nn_in_ori_pcs = pops.knn_gather(ori_pcs, knn_idx).squeeze(2).contiguous()  # <B, N, C>
    offset = adv_pcs - adv_nn_in_ori_pcs  # <B, N, C>
    return offset


def offset_proj(
    offset: torch.Tensor,
    ori_pcs: torch.Tensor,
    ori_normals: torch.Tensor
) -> torch.Tensor:
    knn_idx = pops.knn_points(offset, ori_pcs, K=1).idx  # <B, N, 1, C>
    normal  = pops.knn_gather(ori_normals, knn_idx).squeeze(2).contiguous()  # <B, N, C>

    normal_norm = normal.norm(p=2, dim=-1, keepdim=True)  # <B, N, 1>
    offset_proj = (offset * normal / (normal_norm + 1e-6)).sum(dim=-1, keepdim=True) * normal / (normal_norm + 1e-6) # <B, N, 3>
    return offset_proj


def lp_clip(offset: torch.Tensor, epsilon: float, p: float=2) -> torch.Tensor:
    if epsilon == 0.:
        return torch.zeros_like(offset)

    if p == np.inf:
        offset = torch.clamp(offset, min=-epsilon, max=epsilon)
    else:
        offset_norm = offset.norm(p=p, dim=-1, keepdim=True)  # <B, N, 1>
        offset = torch.where(offset_norm < epsilon, offset, offset / offset_norm * epsilon)  # <B, N, C>, C=3
    return offset

