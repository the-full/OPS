import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from omegaconf import DictConfig

from .utils.vn_dgcnn_utils import *
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


class VNDGCNN(BasicModel):
    def __init__(
        self, 
        pooling='max', 
        n_knn = 20,
        num_class=40, 
    ):
        super(VNDGCNN, self).__init__()
        args = DictConfig(
            dict(
                pooling = pooling,
                n_knn = n_knn,
            )
        )
        self.args = args
        self.n_knn = args.n_knn
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

        self.loss_function = LabelSmoothedCrossEntropyLoss()

    def forward(self, data_dict):
        x = data_dict['xyz'].permute(0, 2, 1)

        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.n_knn)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, _ = self.std_feature(x)
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x


    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=0.1,
            momentum=0.9,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,  # type: ignore
            eta_min=1e-3,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
