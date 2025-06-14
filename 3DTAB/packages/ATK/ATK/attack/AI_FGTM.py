import math

import torch

from .attack_template import IterAttack

class AIFGTM(IterAttack):
    ''' Adam Iterative Fast Gradient Tanh Method (AI-FGTM). 

    Ref: 
        - Making Adversarial Examples More Transferable and Indistinguishable (AAAI 2022)
        - https://arxiv.org/abs/2007.03838
        - https://github.com/278287847/AI-FGTM
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/aifgtm.py
    '''
    def __init__(
        self, 
        model, 
        mu_1     = 1.5,
        mu_2     = 1.9,
        beta_1   = 0.9,
        beta_2   = 0.99,
        lambda_  = 1.3,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.mu_1     = mu_1
        self.mu_2     = mu_2
        self.beta_1   = beta_1
        self.beta_2   = beta_2
        self.lambda_  = lambda_


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        m     = torch.zeros_like(delta).detach()
        v     = torch.zeros_like(delta).detach()
        alpha_list = self.pre_compute_alpha()

        for t in range(self.num_iter):
            logits = self.get_logits(ori_pcs + delta)
            loss   = self.get_loss(logits, labels, target)
            grad   = self.get_grad(loss, delta)

            # Adam update
            m     = m + self.mu_1 * grad
            v     = v + self.mu_2 * grad * grad
            grad  = self.lambda_ * m / (torch.sqrt(v) + 1e-20)
            delta = self._update_delta(delta, alpha_list[t], grad.tanh())
            delta = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()

    def pre_compute_alpha(self):
        alpha_list = [0. for _ in range(self.num_iter)]
        for t in range(self.num_iter):
            alpha_list[t] = (1-self.beta_1**(t+1))/math.sqrt(1-self.beta_2**(t+1))
        res = sum(alpha_list)
        for t in range(self.num_iter):
            alpha_list[t] = self.budget / res * alpha_list[t]
        return alpha_list

