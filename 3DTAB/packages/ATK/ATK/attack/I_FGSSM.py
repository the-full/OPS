import torch
import numpy as np

from .attack_template import IterAttack


class IFGSSM(IterAttack):
    ''' Iterative Fast Gradient Staircase Sign Method (I-FGSSM).

    Ref: 
        - Staircase Sign Method for Boosting Adversarial Attacks
        - https://arxiv.org/abs/2104.09722
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/ifgssm.py
        - https://github.com/qilong-zhang/Staircase-sign-method
    '''

    def __init__(self, model, tao=1.5625, **kwargs):
        super().__init__(model, **kwargs)
        self.tao = tao

    def ssign(self, noise):
        noise = noise.permute(0, 2, 1)
        noise_staircase = torch.zeros_like(noise)
        B, C, N = noise.shape
        percentiles = []
        sign = torch.sign(noise)
        temp_noise = noise
        abs_noise = abs(noise)
        base = self.tao / 100

        for i in np.arange(self.tao, 100.1, self.tao):
            percentile = torch.quantile(
                abs_noise.reshape(-1, N), 
                q = float(i/100), 
                dim = 1, 
                keepdim = True, 
                interpolation='lower'
            ).reshape(B, C, 1)
            percentiles.append(percentile)

        for k, percentile in enumerate(percentiles):
            update = sign * (abs(temp_noise) <= percentile).float() * ((2*k+1) * base)
            noise_staircase += update
            temp_noise += update * 1e5 # NOTE: avoid repeat update.

        return noise_staircase.permute(0, 2, 1)

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        for _ in range(self.num_iter):
            logits = self.get_logits(ori_pcs + delta)
            loss   = self.get_loss(logits, labels, target)
            grad   = self.get_grad(loss, delta)
            delta  = self.update_delta(delta, self.ssign(grad))
            delta  = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()
