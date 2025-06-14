import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.simpleview_utils import *
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


class MVModel(BasicModel):
    def __init__(
        self, 
        task='cls', 
        num_classes=40, 
        backbone='resnet18',
        feat_size=3
    ):
        super().__init__()
        self.loss_function = LabelSmoothedCrossEntropyLoss()
        assert task == 'cls'
        self.task = task
        self.num_class = num_classes
        self.dropout_p = 0.5
        self.feat_size = feat_size

        pc_views = PCViews()
        self.num_views = pc_views.num_views
        self._get_img = pc_views.get_img

        img_layers, in_features = self.get_img_layers(
            backbone, feat_size=feat_size)
        self.img_model = nn.Sequential(*img_layers)

        self.final_fc = MVFC(
            num_views=self.num_views,
            in_features=in_features,
            out_features=self.num_class,
            dropout_p=self.dropout_p)

    def forward(self, data_dict):
        """
        :param pc:
        :return:
        """
        pc = data_dict['xyz']

        img = self.get_img(pc)
        feat = self.img_model(img)
        logit = self.final_fc(feat)
        return logit


    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=1e-3, 
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor = 0.5,
            patience = 10,
            verbose = False,
            min_lr = 0.00001,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'loss'}


    def get_img(self, pc):
        img = self._get_img(pc).clone().detach()
        # img = torch.tensor(img).float()
        img = img.to(next(self.parameters()).device)
        assert len(img.shape) == 3
        img = img.unsqueeze(3)
        # [num_pc * num_views, 1, RESOLUTION, RESOLUTION]
        img = img.permute(0, 3, 1, 2)

        return img

    @staticmethod
    def get_img_layers(backbone, feat_size):
        """
        Return layers for the image model
        """

        assert backbone == 'resnet18'
        layers = [2, 2, 2, 2]
        block = BasicBlock
        backbone_mod = _resnet(
            arch=None,
            block=block,
            layers=layers,
            pretrained=False,
            progress=False,
            feature_size=feat_size,
            zero_init_residual=True)

        all_layers = [x for x in backbone_mod.children()]
        in_features = all_layers[-1].in_features

        # all layers except the final fc layer and the initial conv layers
        # WARNING: this is checked only for resnet models
        main_layers = all_layers[4:-1]
        img_layers = [
            nn.Conv2d(1, feat_size, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1), bias=False),
            nn.BatchNorm2d(feat_size, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            *main_layers,
            Squeeze()
        ]

        return img_layers, in_features


class MVFC(nn.Module):
    """
    Final FC layers for the MV model
    """

    def __init__(self, num_views, in_features, out_features, dropout_p):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.model = nn.Sequential(
                BatchNormPoint(in_features),
                # dropout before concatenation so that each view drops features independently
                nn.Dropout(dropout_p),
                nn.Flatten(),
                nn.Linear(in_features=in_features * self.num_views,
                          out_features=in_features),
                nn.BatchNorm1d(in_features),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(in_features=in_features, out_features=out_features,
                          bias=True))

    def forward(self, feat):
        feat = feat.view((-1, self.num_views, self.in_features))
        out = self.model(feat)
        return out
