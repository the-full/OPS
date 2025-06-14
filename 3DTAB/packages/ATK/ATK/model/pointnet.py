import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from .utils.pointnet_utils import PointNetEncoder
from .basic_model import BasicModel


class PointNetLoss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(PointNetLoss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = nn.CrossEntropyLoss()(pred, target)
        mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
    
    @staticmethod
    def feature_transform_reguliarzer(trans):
        device = trans.device
        d = trans.size()[1]
        I = torch.eye(d, device=device)[None, :, :]
        loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
        return loss


class PointNet(BasicModel):
    def __init__(self, num_classes=40, feat_in=0):
        super(PointNet, self).__init__()
        self.feat = PointNetEncoder(
            global_feat=True, 
            feature_transform=True, 
            channel=feat_in + 3
        )
        self.num_classes = num_classes
        self.fc1     = nn.Linear(1024, 512)
        self.fc2     = nn.Linear(512, 256)
        self.fc3     = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)
        self.relu    = nn.ReLU()

        self.loss_function = PointNetLoss()

    def forward(self, data_dict):
        x = data_dict['xyz'].permute(0, 2, 1)
        x, _, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans_feat

    def training_step(self, batch, batch_idx): # type: ignore
        labels = batch['category'].view(-1)

        logits, trans_feat = self(batch)
        loss = self.loss_function(logits, labels, trans_feat=trans_feat)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx): # type: ignore
        labels = batch['category'].view(-1)

        logits, trans_feat = self(batch)
        preds = logits.argmax(dim=-1)
        loss = self.loss_function(logits, labels, trans_feat=trans_feat)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.validation_step_outputs.append((preds, labels))
        return loss
