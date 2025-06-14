import torch

from .attack_template import IterAttack

class SIM(IterAttack):
    ''' Scale Invariant Momentum (SIM) attack.'''
    def __init__(self, model, num_scale=5, **kwargs):
        super().__init__(model, **kwargs)
        self.num_scale = num_scale

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()

        for _ in range(self.num_iter):
            grad_sum = torch.zeros_like(ori_pcs)
            for i in range(self.num_scale):
                logits    = self.get_logits((ori_pcs + delta) / (2 ** i))
                loss      = self.get_loss(logits, labels, target)
                grad_sum += self.get_grad(loss, delta)
            grad = grad_sum / self.num_scale
            momentum = self.update_momentum(momentum, grad)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
