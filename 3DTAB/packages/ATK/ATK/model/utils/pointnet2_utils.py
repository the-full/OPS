from typing import Optional, Tuple
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from ATK.utils.ops import (
    farthest_point_sample,
    knn_points,
    knn_gather,
    ball_query,
)


# NOTE: add contiguous after rearrange to avoid waring when using DDP
# https://github.com/pytorch/pytorch/issues/47163#issuecomment-1766472400
def rearrange(tensor, pattern):
    return einops.rearrange(tensor, pattern).contiguous()


SGReturn = namedtuple('SGReturn', ['fps_xyz', 'grouped_features', 'fps_idx', 'grouped_idx'])
def sample_and_group(
    npoint: int,
    nsample: int,
    radius: float,
    xyz: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    return_idx: bool = False
):
    fps_xyz, fps_idx = farthest_point_sample(xyz, npoint) # [<B, S, C>, <B, S>], S = npoint
    grouped_xyz, idx = ball_query(
        query_pcs=fps_xyz,
        source_pcs=xyz,
        radius=radius,
        k=nsample,
        use_nearest=True
    ) # [<B, S, K, C>, <B, S, K>, K = nsample
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


# ================================================================================================================================================
# ================================================================================================================================================
# ================================================================================================================================================


class PointNetSetAbstraction(nn.Module):
    def __init__(
        self, 
        npoint, 
        radius, 
        nsample, 
        in_channel, 
        mlp, group_all
    ):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.group_all = group_all

    def forward(
        self,
        xyz: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.group_all:
            # [<B, S, C>, <B, S, K, C+D>], S=1, K=N
            new_xyz, new_features = sample_and_group_all(xyz, features)
        else:
            # [<B, S, C>, <B, S, K, C+D>], S=npoint, K=nsample
            new_xyz, new_features, _, _ = sample_and_group(
                npoint=self.npoint,
                nsample=self.nsample,
                radius=self.radius,
                xyz=xyz,
                features=features
            )
        new_features = rearrange(new_features, 'B S K CD -> B CD S K') # <B, C+D, S, K>
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_features =  F.relu(bn(conv(new_features))) # <B, D', S, K>

        new_features = torch.max(new_features, dim=-1)[0] # <B, D', S>
        new_features = new_features.permute(0, 2, 1) # <B, S, D'>
        return new_xyz, new_features


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(
        self, 
        npoint, 
        radius_list, 
        nsample_list, 
        in_channel, 
        mlp_list
    ):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks_list = nn.ModuleList()
        self.bn_blocks_list = nn.ModuleList()

        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks_list.append(convs)
            self.bn_blocks_list.append(bns)

    def forward(self, xyz: torch.Tensor, features: Optional[torch.Tensor] = None):
        new_xyz, _ = farthest_point_sample(xyz, self.npoint) # <B, S, C>, S=npoint
        new_features_list = []
        for radius, nsample, convs, bns in zip(
            self.radius_list,
            self.nsample_list,
            self.conv_blocks_list,
            self.bn_blocks_list
        ):
            grouped_xyz, grouped_idx = ball_query(
                query_pcs=new_xyz,
                source_pcs=xyz,
                radius=radius,
                k=nsample,
                use_nearest=True
            ) # [<B, S, K, C>, <B, S, K>], K=nsample
            grouped_xyz -= new_xyz.unsqueeze(-2) # <B, S, K, C>
            if features is not None:
                grouped_features = knn_gather(features, grouped_idx) # <B, S, K, D>
                grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1) # <B, S, K, C+D>
            else:
                grouped_features = grouped_xyz # <B, S, K, D>

            grouped_features = rearrange(grouped_features, 'B S K D -> B D S K') # <B, D, S, K> or <B, D+C, S, K>

            assert isinstance(convs, nn.ModuleList) and isinstance(bns, nn.ModuleList)
            for conv, bn in zip(convs, bns):
                grouped_features =  F.relu(bn(conv(grouped_features))) # <B, D', S, K>
            new_features = torch.max(grouped_features, dim=-1)[0]  # <B, D', S>
            new_features_list.append(new_features)

        new_features = torch.cat(new_features_list, dim=1) # <B, D', S>
        new_features = new_features.permute(0, 2, 1) # <B, S, D'>
        return new_xyz, new_features

