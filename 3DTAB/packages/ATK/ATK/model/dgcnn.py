"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ATK.utils.ops import get_graph_feature

from .basic_model import BasicModel


class LabelSmoothedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, epsilon=0.2):
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.epsilon = epsilon

    def __call__(self, pred, gold):
        gold = gold.contiguous().view(-1)

        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.epsilon) + (1 - one_hot) * self.epsilon / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss


class DGCNN(BasicModel):
    def __init__(
        self, 
        num_classes, 
        k = 20, 
        emb_dims = 1024, 
        prob_dropout=0.5
    ):
        super(DGCNN, self).__init__()
        self.k = k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=1, bias=False),
            self.bn1,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
            self.bn2,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
            self.bn3,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
            self.bn4,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
            self.bn5,
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.linear1 = nn.Linear(emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=prob_dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=prob_dropout)
        self.linear3 = nn.Linear(256, num_classes)

        self.loss_function = LabelSmoothedCrossEntropyLoss()

    def forward(self, data_dict):
        # NOTE: official implement only use xyz
        x = data_dict['xyz'].permute(0, 2, 1) # <B, C, N>

        B = x.size(0)
        x = get_graph_feature(x, k=self.k) # <B, 2*C, N, k>
        x = self.conv1(x) # <B, f1, N, k>
        x1 = x.max(dim=-1, keepdim=False)[0] # <B, f1, N>

        x = get_graph_feature(x1, k=self.k) # <B, 2*f1, N, k>
        x = self.conv2(x) # <B, f2, N, k>
        x2 = x.max(dim=-1, keepdim=False)[0] # <B, f2, N>

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x
