import torch
import torch.nn as nn

from .utils.pointcnn_utils import AbbPointCNN, Dense
from .basic_model import BasicModel

class Classifier(BasicModel):

    def __init__(self, num_classes):
        super(Classifier, self).__init__()

        self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(32, 64, 8, 2, -1),
            AbbPointCNN(64, 96, 8, 4, -1),
            AbbPointCNN(96, 128, 12, 4, 120),
            AbbPointCNN(128, 160, 12, 6, 120)
        )

        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, num_classes, with_bn=False, activation=None)
        )

    def forward(self, data_dict):
        xyz = data_dict['xyz']
        feat = data_dict['feat']
        x = (xyz, feat)

        x = self.pcnn1(x)
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean

