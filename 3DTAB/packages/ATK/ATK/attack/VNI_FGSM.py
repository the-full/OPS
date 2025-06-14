import torch

from ..utils import *
from .attack_template import SampleAttack

class VNIFGSM(SampleAttack):
    ''' Variance tuning Nesterov's accelerated Iteration FGSM (VNI-FGSM).

    Ref:
        - Enhancing the Transferability of Adversarial Attacks through Variance Tuning (CVPR 2021)
        - https://arxiv.org/abs/2103.15571.
        - https://github.com/JHL-HUST/VT/blob/main/vni_fgsm.py
    '''
    def __init__(
        self, 
        model, 
        beta = 2.0,
        num_sample = 20,
        **kwargs
    ):
        super().__init__(
            model, 
            beta = beta,
            num_sample = num_sample,
            **kwargs
        )

    def look_ahead(self, x, momentum):
        return x + self.alpha * self.mu * momentum

    def get_sampled_grad(self, ori_pcs, delta, momentum, labels, target=None): # type: ignore
        grad_sum = torch.zeros_like(ori_pcs)
        for _ in range(self.num_sample):
            perturb   = self.get_perturb(delta, self.beta)
            nes_del   = self.look_ahead(delta, momentum)
            logits    = self.get_logits(ori_pcs + nes_del + perturb)
            loss      = self.get_loss(logits, labels, target)
            grad_sum += self.get_grad(loss, delta)
        return grad_sum / self.num_sample

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()
        variance = torch.zeros_like(ori_pcs).detach()

        for _ in range(self.num_iter):
            nes_del   = self.look_ahead(delta, momentum)
            logits    = self.get_logits(ori_pcs + nes_del)
            loss      = self.get_loss(logits, labels, target)
            grad      = self.get_grad(loss, delta)
            momentum  = self.update_momentum(momentum, grad + variance)
            variance  = self.get_sampled_grad(ori_pcs, delta, momentum, labels, target) - grad
            delta     = self.update_delta(delta, momentum.sign())
            delta     = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()
