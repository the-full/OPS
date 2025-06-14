"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch.nn.functional as F

from .utils.pointconv_utils import PointConvDensitySetAbstraction
from .basic_model import BasicModel


class PointConvDensitySSG(BasicModel):
    def __init__(
        self, 
        num_classes,
        feat_in=0
    ):
        super(PointConvDensitySSG, self).__init__()
        self.feat_in = feat_in
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feat_in + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.7)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, data_dict):
        xyz = data_dict['xyz']
        feat = data_dict['feat'][:, :, 3:]

        B, _, _ = xyz.shape

        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _,      l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
