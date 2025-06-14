"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig

from .utils.repsurf_utils.modules.repsurface_utils import (
        SurfaceAbstractionCD, 
        UmbrellaSurfaceConstructor
)
from .basic_model import BasicModel


class Model(BasicModel):
    def __init__(
        self, 
        return_polar = True,
        return_center = True,
        return_dist = True,
        num_point = 1024,
        group_size = 8,
        umb_pool = 'sum',
        cuda_ops = True,
        num_class = 40,
    ):
        args = DictConfig(
            dict(
                return_center = return_center,
                return_polar = return_polar,
                num_point = num_point,
                return_dist = return_dist,
                group_size = group_size,
                umb_pool = umb_pool,
                cuda_ops = cuda_ops,
                num_class = num_class,
            )
        )
        super(Model, self).__init__()
        center_channel = 0 if not args.return_center else (6 if args.return_polar else 3)
        repsurf_channel = 10

        self.init_nsample = args.num_point
        self.return_dist = args.return_dist
        self.surface_constructor = UmbrellaSurfaceConstructor(args.group_size + 1, repsurf_channel,
                                                              return_dist=args.return_dist, aggr_type=args.umb_pool,
                                                              cuda=args.cuda_ops)
        self.sa1 = SurfaceAbstractionCD(npoint=512, radius=0.1, nsample=24, feat_channel=repsurf_channel,
                                        pos_channel=center_channel, mlp=[128, 128, 256], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa2 = SurfaceAbstractionCD(npoint=128, radius=0.2, nsample=24, feat_channel=256 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[256, 256, 512], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa3 = SurfaceAbstractionCD(npoint=32, radius=0.4, nsample=24, feat_channel=512 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[512, 512, 1024], group_all=False,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        self.sa4 = SurfaceAbstractionCD(npoint=None, radius=None, nsample=None, feat_channel=1024 + repsurf_channel,
                                        pos_channel=center_channel, mlp=[1024, 1024, 2048], group_all=True,
                                        return_polar=args.return_polar, cuda=args.cuda_ops)
        # modelnet40
        self.classfier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(256, args.num_class))

    def forward(self, data_dict):
        # init
        center = data_dict['xyz'].permute(0, 2, 1)

        normal = self.surface_constructor(center)

        center, normal, feature = self.sa1(center, normal, None)
        center, normal, feature = self.sa2(center, normal, feature)
        center, normal, feature = self.sa3(center, normal, feature)
        center, normal, feature = self.sa4(center, normal, feature)

        feature = feature.view(-1, 2048)
        feature = self.classfier(feature)
        feature = F.log_softmax(feature, -1)

        return feature

    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=1e-3, 
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=20,
            gamma=0.7,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
