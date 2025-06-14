import torch
import torch.nn as nn

from omegaconf import DictConfig

from .utils.vn_transformer_utils import layers as vn
from .basic_model import BasicModel


class InvariantClassifier(BasicModel):
    # Described in Figure 2
    def __init__(
        self,
        num_classes = 40,
        in_features = 1,
        hidden_features = 256,
        num_heads = 64,
        latent_size = 64,
        bias_eps = 1e-6,
        leaky=0.0,
    ):
        super().__init__()
        args = DictConfig(dict(
            leaky = leaky,
        ))
        self.vn_mlp = nn.Sequential(
            vn.Linear(in_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.LeakyReLU(hidden_features, args.leaky),
            vn.Linear(hidden_features, hidden_features, bias_eps),
            vn.BatchNorm(hidden_features),
            vn.LeakyReLU(hidden_features, args.leaky),
        )

        if latent_size is not None:
            self.query_proj = vn.MeanProject(latent_size, hidden_features, hidden_features)
        else:
            self.query_proj = nn.Identity()

        self.vn_transformer = vn.TransformerBlock(f_dim=hidden_features, num_heads=num_heads, bias_eps=bias_eps, leaky=args.leaky)

        self.vn_mlp_inv = nn.Sequential(
            vn.Linear(hidden_features, 3, bias_eps),
            vn.LeakyReLU(3, args.leaky),
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_features*3, hidden_features),
            nn.ReLU(True),
            nn.Linear(hidden_features, num_classes)
        )

    def forward(self, data_dict):
        '''
        x: tensor of shape [B, num_features, 3, num_points]
        return: tensor of shape [B, num_classes]
        '''
        x = data_dict['xyz'].permute(0, 2, 1).unsqueeze(1)

        x = self.vn_mlp(x)

        queries = self.query_proj(x)
        x = self.vn_transformer(x, queries)

        x = vn.invariant(x, self.vn_mlp_inv(x))

        x = torch.flatten(vn.mean_pool(x), start_dim=1)

        x = self.mlp(x)

        return x

    def configure_optimizers(self): # type: ignore
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,  # type: ignore
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
