from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from ATK.utils.ops import (
    farthest_point_sample,
    knn_points,
    knn_gather,
)


# NOTE: add contiguous after rearrange to avoid waring when using DDP
# https://github.com/pytorch/pytorch/issues/47163#issuecomment-1766472400
def rearrange(tensor, pattern):
    return einops.rearrange(tensor, pattern).contiguous()

SGReturn = namedtuple('SGReturn', ['fps_xyz', 'grouped_xyz', 'grouped_features', 'grouped_density'])
def sample_and_group(npoint, nsample, xyz, features = None, density_scale = None):
    """
    在数据中执行采样和分组操作。（使用 knn 来确定邻域)

    Args:
        npoint (int): 采样点的数量。
        nsample (int): 每个采样点的近邻数量。
        xyz (torch.FloatTensor<B, N, C>): 输入点的坐标数据。
        features (torch.FloatTensor<B, N, D>): 输入点的特征数据。默认为 None。
        density_scale (torch.FloatTensor<B, N, 1>): 输入点的密度尺度。默认为 None。

    Returns:
        SGReturn: 一个包含以下内容的命名元组:
            - fps_xyz (torch.FloatTensor<B, S, C>): 采样点的坐标数据。
            - grouped_xyz (torch.FloatTensor<B, S, K, C>): 采样点的近邻点的局部坐标数据。
            - grouped_features (torch.FloatTensor<B, S, K, C+D>): 采样点的近邻点的特征数据。
            - grouped_density (torch.FloatTensor<B, S, K, 1>): 采样点的近邻点的密度。

    Note:
        fps_idx, grouped_idx 在 return_idx 为 False 时返回为 None。
        S = npoints, K = nsample

    Ref:
        https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py#L120-L148
    """
    fps_xyz, _ = farthest_point_sample(xyz, npoint) # [<B, S, C>, <B, S>] # S = npoint
    grouped_xyz, idx = knn_points(fps_xyz, xyz, k=nsample) # [<B, S, K, C>, <B, S, K>] # K = nsample
    grouped_xyz = grouped_xyz - fps_xyz.unsqueeze(-2) # <B, S, K, C>

    if features is not None:
        grouped_features = knn_gather(features, idx) # <B, S, K, D>
        grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1) # <B, S, K, C+D>
    else:
        grouped_features = grouped_xyz # <B, S, K, C>

    if density_scale is not None:
        grouped_density = knn_gather(density_scale, idx) # <B, S, K, 1>
        return SGReturn(fps_xyz, grouped_xyz, grouped_features, grouped_density)
    else:
        return SGReturn(fps_xyz, grouped_xyz, grouped_features, None)


SGAllReturn = namedtuple('SGAllReturn', ['center', 'grouped_xyz', 'grouped_features', 'grouped_density'])
def sample_and_group_all(xyz, features = None, density_scale = None):
    center = xyz.mean(dim = 1, keepdim = True) # <B, 1, C>
    grouped_xyz = (xyz - center).unsqueeze(1) # <B, 1, N, C>

    if features is not None:
        grouped_features = torch.cat([grouped_xyz, features.unsqueeze(1)], dim=-1) # <B, 1, N, C>
    else:
        grouped_features = grouped_xyz # <B, 1, N, C>

    if density_scale is None:
        return SGAllReturn(center, grouped_xyz, grouped_features, None)
    else:
        grouped_density = density_scale.unsqueeze(1) # <B, 1, N, 1>
        return SGAllReturn(center, grouped_xyz, grouped_features, grouped_density)


# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================


class DensityNet(nn.Module):
    def __init__(self, hidden_unit = [16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale =  bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)

        return density_scale

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))

    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights

class PointConvSetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, group_all):
        super(PointConvSetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.group_all = group_all

    def forward(self, xyz, features=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) # <B, N, C>
        if features is not None:
            features = features.permute(0, 2, 1) # <B, N, D>

        if self.group_all:
            # [<B, S, C>, <B, S, K, C>, <B, S, K, C+D>, None], S=1, K=N
            new_xyz, grouped_xyz, new_features, _ = sample_and_group_all(xyz, features)
        else:
            # [<B, S, C>, <B, S, K, C>, <B, S, K, C+D>, None], S=npoint, K=nsample
            new_xyz, grouped_xyz, new_features, _ = sample_and_group(self.npoint, self.nsample, xyz, features)

        new_features = rearrange(new_features, 'B S K CD -> B CD S K') # <B, C+D, S, K>
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features =  F.relu(bn(conv(new_features))) # <B, D', S, K>

        grouped_xyz = rearrange(grouped_xyz, 'B S K C -> B C S K') # <B, C, S, K>
        weights = self.weightnet(grouped_xyz) # <B, C', S, K>, C=3, C'=16

        new_features = torch.einsum('B D S K, B C S K -> B S D C', new_features, weights) # <B, S, D', C'>
        new_features = rearrange(new_features, 'B S D C -> B S (D C)') # <B, S, 16*D'>
        new_features = self.linear(new_features) # <B, S, D'>
        new_features = self.bn_linear(new_features.permute(0, 2, 1)) # <B, D', S>
        new_features = F.relu(new_features) # <B, D', S>
        new_xyz = new_xyz.permute(0, 2, 1) # <B, C, S>

        return new_xyz, new_features


class PointConvDensitySetAbstraction(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConvDensitySetAbstraction, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz, features=None):
        """
        Input:
            xyz: input features position data, [B, C, N]
            features: input features data, [B, D, N]
        Return:
            new_xyz: sampled features position data, [B, C, S]
            new_features_concat: sample features feature data, [B, D', S]
        """
        xyz_density = self.compute_density(xyz, self.bandwidth) # <B, N>
        inverse_density = 1.0 / xyz_density # <B, N>
        inverse_density = rearrange(inverse_density, 'B N -> B N 1') # <B, N, 1>

        if self.group_all:
            # [<B, S, C>, <B, S, K, C>, <B, S, K, C+D>, <B, S, K, 1>], S=1, K=N
            new_xyz, grouped_xyz, new_features, grouped_density = sample_and_group_all(xyz, features, inverse_density)
        else:
            # [<B, S, C>, <B, S, K, C>, <B, S, K, C+D>, <B, S, K, 1>], S=npoint, K=nsample
            new_xyz, grouped_xyz, new_features, grouped_density = sample_and_group(self.npoint, self.nsample, xyz, features, inverse_density)

        new_features = rearrange(new_features, 'B S K CD -> B CD S K') # <B, C+D, S, K>
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features =  F.relu(bn(conv(new_features))) # <B, D', S, K>

        inverse_max_density = grouped_density.max(dim=2, keepdim=True)[0] # <B, S, 1, 1>
        density_scale = grouped_density / inverse_max_density # <B, S, K, 1>
        density_scale = rearrange(density_scale, 'B S K 1 -> B 1 S K') # <B, 1, S, K>
        density_scale = self.densitynet(density_scale) # <B, 1, S, K>
        new_features = new_features * density_scale # <B, D', S, K>

        grouped_xyz = rearrange(grouped_xyz, 'B S K C -> B C S K') # <B, C, S, K>
        weights = self.weightnet(grouped_xyz) # <B, C', S, K>, C=3, C'=16

        new_features = torch.einsum('B D S K, B C S K -> B S D C', new_features, weights) # <B, S, D', C'>
        new_features = rearrange(new_features, 'B S D C -> B S (D C)') # <B, S, 16*D'>
        new_features = self.linear(new_features) # <B, S, D'>
        new_features = self.bn_linear(new_features.permute(0, 2, 1)) # <B, D', S>
        new_features = F.relu(new_features) # <B, D', S>
        new_features = new_features.permute(0, 2, 1) # <B, S, D'>

        return new_xyz, new_features

    @staticmethod
    def compute_density(xyz, bandwidth):
        '''
        xyz: input points position data, [B, N, C]
        '''
        sqrdists = torch.cdist(xyz, xyz)
        gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
        xyz_density = gaussion_density.mean(dim = -1)
        return xyz_density

