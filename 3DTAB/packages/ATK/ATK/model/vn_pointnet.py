import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from .utils.vn_pointnet_utils import *
from .basic_model import BasicModel


class VNPointNet(BasicModel):
    def __init__(
        self, 
        pooling='max', 
        num_class=40, 
        n_knn = 20,
        normal_channel=False,
    ):
        args = DictConfig(
            dict(
                pooling = pooling,
                n_knn = n_knn,
            )
        )
        super(VNPointNet, self).__init__()
        self.args = args
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024//3*6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, data_dict):
        x = data_dict['xyz'].permute(0, 2, 1)

        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)

        return x

    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=1e-2,
            momentum=0.9,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=20,
            gamma=0.7,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
