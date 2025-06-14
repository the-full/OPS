import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points, knn_gather
from einops import rearrange
import numpy as np

from .ops import ball_query, farthest_point_sample, get_pcs_by_idx


def l2_norm_distance(pcs1: torch.Tensor, pcs2: torch.Tensor) -> torch.Tensor:
    return torch.norm(pcs1 - pcs2, dim=-1).mean(-1)  # <B,>


def chamfer_distance(pcs1, pcs2, single=False, reduce='max'):
    nn_dis_12 = knn_points(pcs1, pcs2, K=1).dists # <B, N, 1>
    nn_dis_21 = knn_points(pcs2, pcs1, K=1).dists # <B, N, 1>
    dis_12 = nn_dis_12.squeeze(-1).mean(-1) # <B,>
    dis_21 = nn_dis_21.squeeze(-1).mean(-1) # <B,>

    if single:
        chamfer = dis_12
    else:
        if reduce == 'max':
            chamfer = torch.max(dis_12, dis_21)
        elif reduce == 'mean':
            chamfer = torch.mean(dis_12, dis_21)
        elif reduce == 'sum':
            chamfer = dis_12 + dis_21
        else:
            raise NotImplemented
    return chamfer


def hausdorff_distance(pcs1, pcs2, single=False, reduce='max') -> torch.Tensor:
    nn_dis_12 = knn_points(pcs1, pcs2, K=1).dists # <B, N, 1>
    nn_dis_21 = knn_points(pcs2, pcs1, K=1).dists # <B, N, 1>
    dis_12 = nn_dis_12.squeeze(-1).max(-1).values # <B,>
    dis_21 = nn_dis_21.squeeze(-1).max(-1).values # <B,>

    if single:
        hausdorff_dis = dis_12
    else:
        if reduce == 'max':
            hausdorff_dis = torch.max(dis_12, dis_21)
        elif reduce == 'mean':
            hausdorff_dis = torch.mean(dis_12, dis_21)
        elif reduce == 'sum':
            hausdorff_dis = dis_12 + dis_21
        else:
            raise NotImplemented

    return hausdorff_dis


def _normalize(pcs, p=2, dim=-1, eps=1e-12):
    return pcs / (pcs.norm(p=p, dim=dim, keepdim=True).clip(min=eps))

def _get_curvature(pcs, normals, k=2):
    knn_pts = knn_points(pcs, pcs, K=k+1, return_nn=True).knn # <B, N, K+1, 3>
    knn_pts = knn_pts[:, :, 1:, :].contiguous() # <B, N, K, 3>
    vectors = knn_pts - pcs.unsqueeze(-2) # <B, N, K, 3>
    vectors = _normalize(vectors)
    ori_curve = torch.abs((vectors * normals.unsqueeze(-2)).sum(-1)).mean(-1)  # <B, N>
    return ori_curve

def _get_std_curvature(pcs, normals, k=5):
    _, knn_idx, knn_pts = knn_points(pcs, pcs, K=k+1, return_nn=True) # <B, N, K+1, 3>
    knn_pts = knn_pts[:, :, 1:, :].contiguous() # <B, N, K, 3>
    vectors = knn_pts - pcs.unsqueeze(-2) # <B, N, K, 3>
    vectors = _normalize(vectors)

    ori_curve = torch.abs((vectors * normals.unsqueeze(-2)).sum(-1)).mean(-1)  # <B, N>
    knn_curve = knn_gather(ori_curve.unsqueeze(-1), knn_idx)  # <B, N, K+1, 1>
    knn_curve = knn_curve.squeeze(-1)[:, :, 1:].contiguous() # <B, N, K>
    std_curve = torch.std(knn_curve, dim=-1) # <B, N>
    return std_curve


def curvature_diff(ori_pcs, adv_pcs, ori_curve, adv_curve):
    nn_idx  = knn_points(adv_pcs, ori_pcs, K=1).idx # <B, N, 1>
    nn_idx  = nn_idx.squeeze(-1) # <B, N>
    nn_curv = torch.gather(ori_curve, dim=-1, index=nn_idx) # <B, N>
    return (adv_curve - nn_curv).square().mean(-1) # <B,>


def curvature_std_distance(ori_pcs, adv_pcs, ori_normals, k=5):
    # ref: https://github.com/TRLou/HiT-ADV/blob/master/util/dist_utils.py#L464-L495
    pdist = torch.nn.PairwiseDistance(p=2)
    ori_curve_std = _get_std_curvature(ori_pcs, ori_normals, k=k)
    adv_curve_std = _get_std_curvature(adv_pcs, ori_normals, k=k)
    curve_std_dis = pdist(ori_curve_std, adv_curve_std) # <B>
    return curve_std_dis


def curvature_distance(ori_pcs, adv_pcs, ori_normals, k=2):
    # ref: https://github.com/TRLou/HiT-ADV/blob/master/util/dist_utils.py#L498-L561
    def get_adv_curve(ori_pcs, adv_pcs, ori_normal, k=2):
        knn_idx = knn_points(adv_pcs, ori_pcs, K=1).idx # <B, N, 1, 3>
        normals = knn_gather(ori_normal, knn_idx).squeeze(-2) # <B, N, 3>

        return _get_curvature(adv_pcs, normals, k=k)

    ori_curve = _get_curvature(ori_pcs, ori_normals, k=k)
    adv_curve = get_adv_curve(ori_pcs, adv_pcs, ori_normals, k=k)
    return curvature_diff(ori_pcs, adv_pcs, ori_curve, adv_curve)



def knn_mean_distance(pcs, k=5):
    # ref: https://github.com/Gorilla-Lab-SCUT/GeoA3/blob/master/Lib/loss_utils.py#L125-L133
    knn_dis, knn_idx, _ = knn_points(pcs, pcs, K=k+1)
    knn_dis = knn_dis.sqrt()

    knn_dis_mean = knn_dis[:, :, 1:].contiguous().mean(-1)  # <B, N>
    knn_dis_mean = knn_dis_mean.unsqueeze(-1) # <B, N, 1>

    knn_idx = knn_idx[:, :, 1:].contiguous()
    knn_knn_dis_mean = knn_gather(knn_dis_mean, knn_idx) # <B, N, K, 1>
    knn_knn_dis_mean = knn_knn_dis_mean.squeeze(-1) # <B, N, K>

    kmean_loss = torch.abs(knn_dis_mean - knn_knn_dis_mean).mean(-1)  # <B, N>
    return kmean_loss.mean(-1)
    


def knn_outlier_distance(pcs, k=5, alpha=1.05):
    # ref: https://github.com/jinyier/ai_pointnet_attack/blob/master/attack.py#L206-L227
    # ref: https://github.com/Gorilla-Lab-SCUT/GeoA3/blob/master/Lib/loss_utils.py#L135-L149
    knn_dis = knn_points(pcs, pcs, K=k+1).dists  # [dists: <B, N, K+1>, idx: <B, N, K+1>]
    knn_dis = knn_dis[:, :, 1:].contiguous().mean(-1)  # <B, N>

    knn_dis_mean = knn_dis.mean(-1)  # <B>
    knn_dis_std  = knn_dis.std(-1)  # <B>
    threshold = knn_dis_mean + alpha * knn_dis_std  # <B>

    condition = torch.greater_equal(knn_dis, threshold.unsqueeze(1))  # <B, N>
    penalty = torch.where(condition, knn_dis, 0.) # <B, N>
    # NOTE: In kNN, they use penalty.sum(-1), while in GeoA3, they use penalty.mean(-1)
    penalty = penalty.mean(-1) # <B>
    return penalty


def displacement_diff(pcs1, pcs2, k=16):
    knn_idx = knn_points(pcs1, pcs2, K=k+1).idx # <B, N, K+1>
    knn_idx = knn_idx[:, :, 1:].contiguous() # <B, N, K>

    displacement = F.pairwise_distance(pcs1, pcs2, p=2)  # <B, N>
    knn_idx = rearrange(knn_idx, 'B N K -> B (N K)') # <B, (N * K)>
    knn_displacement = torch.gather(displacement, dim=1, index=knn_idx) # <B, (N * K)>
    knn_displacement = rearrange(knn_displacement, 'B (N K) -> B N K', K=k) # <B, N, K>

    return l2_norm_distance(displacement.unsqueeze(-1), knn_displacement) # <B,>


def repulsion_index(pcs, k, h):
    knn_dists = knn_points(pcs, pcs, K=k+1).dists # <B, N, K+1>
    knn_dists = torch.sqrt(knn_dists[:, :, 1:].contiguous()) # <B, N, K>

    h2 = h * h
    eta_fn = lambda r: -r
    w_fn   = lambda r: torch.exp(-(r * r) / h2)
    inner_term = eta_fn(knn_dists) * w_fn(knn_dists) # <B, N, K>
    return inner_term.mean(axis=(-1, -2))


def uniform_index(pcs, percentages, radius, shape):
    device   = pcs.device
    B, N, _  = pcs.size()
    seed_num = int(N * 0.05) 

    loss = 0.0
    for p in percentages:
        exp_num = N * p
        r = np.sqrt(p * radius * radius)

        if shape == 'square':
            expect_dis = (np.pi * (radius ** 2)) / exp_num * p
        elif shape == 'hexagon':
            expect_dis = (2 * np.pi * (radius ** 2)) / (np.sqrt(3) * exp_num) * p
        else:
            raise ValueError('Unsupported shape: {}'.format(shape))
        expect_dis = torch.sqrt(torch.Tensor([expect_dis])).to(device)

        sub_pcs = get_pcs_by_idx(pcs, farthest_point_sample(pcs, seed_num)[1]) # <B, seed_num, 3>
        grouped_pcs = ball_query(sub_pcs, pcs, r, int(2 * exp_num))[0] # <B, seed_num, 2 * exp_num, 3>
        grouped_pcs = rearrange(grouped_pcs, 'B seed_num exp_num C -> (B seed_num) exp_num C')

        grouped_nn_dis = knn_points(grouped_pcs, grouped_pcs, K=2).dists # <B * seed_num, 2 * exp_num, 2>
        grouped_nn_dis = grouped_nn_dis[:, :, 1:].contiguous() # <B * seed_num, 2 * exp_num, 1>
        grouped_nn_dis = torch.sqrt(grouped_nn_dis).squeeze(-1) # <B * seed_num, 2 * exp_num>

        true_nn_mask = (grouped_nn_dis > 1e-12) # <B * seed_num, 2 * exp_num>
        grouped_num  = torch.sum(true_nn_mask, dim=1) # <B * seed_num>

        uniform_clutter = torch.square(grouped_nn_dis - expect_dis) / (expect_dis + 1e-12) # <B * seed_num, 2 * exp_num>
        uniform_clutter = torch.sum(uniform_clutter * true_nn_mask) / grouped_num # <B * seed_num>

        uniform_imbalance = torch.square(grouped_num - exp_num) / exp_num # <B * seed_num>

        uniform_loss = uniform_clutter * uniform_imbalance # <B * seed_num>
        uniform_loss = rearrange(uniform_loss, '(B seed_num) -> B seed_num', B=B) # <B, seed_num>
        uniform_loss = torch.mean(uniform_loss, dim=-1) # <B,>

        loss += uniform_loss

    return loss / len(percentages)
