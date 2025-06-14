import torch
import torch.nn as nn
import torch.nn.functional as F


class KerLoss(nn.Module):
    def __init__(self, a):
        super(KerLoss, self).__init__()
        self.a = a

    def forward(self, delta: torch.FloatTensor, sigma: torch.FloatTensor) -> torch.FloatTensor:
        num_GCP = delta.shape[1]
        norm_delta = delta.norm(dim=(1, 2)) # <B,>
        norm_sigma = (self.a - sigma).norm(dim=-1) # <B,>
        return (norm_delta + norm_sigma) / num_GCP


class HideLoss(nn.Module):
    def __init__(self, max_sigma, min_sigma):
        super(HideLoss, self).__init__()
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    def forward(self, sigma, curve_std):
        sigma  = (sigma - self.min_sigma) / (self.max_sigma - self.min_sigma + 1e-7)
        curve_std = (curve_std - torch.min(curve_std)) / (torch.max(curve_std) - torch.min(curve_std) + 1e-7)
        return F.cosine_similarity(sigma, curve_std, dim=-1)

