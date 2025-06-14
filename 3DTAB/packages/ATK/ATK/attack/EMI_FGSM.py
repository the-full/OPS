import torch
import numpy as np

from .attack_template import IterAttack


class EMIFGSM(IterAttack):
    ''' Enhanced Momentum Iteration FGSM (EMI-FGSM)

    Ref:
        - Boosting Adversarial Transferability through Enhanced Momentum (BMVC 2021)
        - https://arxiv.org/abs/2103.10609
        - https://github.com/JHL-HUST/EMI/blob/main/emi_fgsm.py
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/emifgsm.py
    '''
    def __init__(
        self, 
        model, 
        radius = 7,
        num_sample = 11,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.radius = radius
        self.num_sample = num_sample


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta    = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()

        grad_bar = torch.zeros_like(delta)
        for _ in range(self.num_iter):
            grad_bar = self.get_grad_bar(ori_pcs, delta, labels, grad_bar, target)
            momentum = self.update_momentum(momentum, grad_bar)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()

    def get_grad_bar(self, ori_pcs, delta, labels, pre_grad, target=None):
        grad_sum = torch.zeros_like(delta)
        factors = np.linspace(-self.radius, self.radius, num=self.num_sample)
        for factor in factors:
            logits = self.get_logits(
                ori_pcs + delta + factor * self.alpha * pre_grad
            )
            loss      = self.get_loss(logits, labels, target)
            grad      = self.get_grad(loss, delta)
            grad_sum += grad
        return self.norm_grad(grad_sum / self.num_sample)

