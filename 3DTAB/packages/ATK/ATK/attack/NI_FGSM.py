import torch

from .attack_template import IterAttack

class NIFGSM(IterAttack):
    ''' Nesterov's accelerated Iterative FGSM (NI-FGSM). 

    Ref: 
        - Nesterov Accelerated Gradient and Scale Invariance for Adversarial Attacks (ICLR2020)
        - https://arxiv.org/abs/1908.06281
        - https://github.com/JHL-HUST/SI-NI-FGSM/blob/master/ni_fgsm.py
    '''
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def look_ahead(self, x, direction):
        return x + self.alpha * self.mu * direction

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta    = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)

        for _ in range(self.num_iter):
            nes_del  = self.look_ahead(delta, momentum)
            logits   = self.get_logits(ori_pcs + nes_del)
            loss     = self.get_loss(logits, labels, target)
            grad     = self.get_grad(loss, delta)
            momentum = self.update_momentum(momentum, grad)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
