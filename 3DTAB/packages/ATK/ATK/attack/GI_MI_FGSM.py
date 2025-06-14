import torch

from .attack_template import IterAttack

class GIMIFGSM(IterAttack):
    ''' Global momentum Initialization MI-FGSM (GI-MI-FGSM).

    Ref: 
        - Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization
        - https://arxiv.org/abs/2211.11236

    '''
    def __init__(self, model, pre_iter=5, s=10, **kwargs):
        super().__init__(model, **kwargs)
        self.pre_iter = pre_iter
        self.s = s

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta    = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)

        for _ in range(self.pre_iter):
            logits   = self.get_logits(ori_pcs + delta)
            loss     = self.get_loss(logits, labels, target)
            grad     = self.get_grad(loss, delta)
            momentum = self.update_momentum(momentum, grad)
            delta    = self.update_delta(delta * self.s, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        delta = self.init_delta(ori_pcs)

        for _ in range(self.num_iter):
            logits   = self.get_logits(ori_pcs + delta)
            loss     = self.get_loss(logits, labels, target)
            grad     = self.get_grad(loss, delta)
            momentum = self.update_momentum(momentum, grad)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
