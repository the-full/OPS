from torch import wait
import torch.nn as nn
import torch.nn.functional as F

from .utils.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
from .basic_model import BasicModel


class PointNetPPMSG(BasicModel):
    def __init__(
        self, 
        num_classes, 
        feat_in=0
    ):
        super(PointNetPPMSG, self).__init__()
        self.num_classes = num_classes
        self.feat_in = feat_in

        in_channel = feat_in
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64],  [64,  64,  128], [64,  96,  128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,       [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, data_dict):
        xyz  = data_dict['xyz']
        feat = None if self.feat_in == 0 else data_dict['feat'][:, :, 3:] # NOTE: don't need xyz feat

        B, _, _ = xyz.shape

        l1_xyz, l1_feat = self.sa1(xyz, feat)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        _,      l3_feat = self.sa3(l2_xyz, l2_feat)
        x = l3_feat.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x


class PointNetPPSSG(BasicModel):
    def __init__(
        self, 
        num_classes,
        feat_in=0
    ):
        super(PointNetPPSSG, self).__init__()
        self.feat_in = feat_in
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=feat_in +3, mlp=[64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, data_dict): # type: ignore
        xyz = data_dict['xyz']
        feat = None if self.feat_in == 0 else data_dict['feat'][:, :, 3:] # NOTE: don't need xyz feat
        
        B, _, _ = xyz.shape

        l1_xyz, l1_feat = self.sa1(xyz, feat)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        _,      l3_feat = self.sa3(l2_xyz, l2_feat)
        x = l3_feat.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x
