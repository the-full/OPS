import torch
from .iter_attack import IterAttack


class SampleAttack(IterAttack):
    def __init__(
        self, 
        model, 
        beta = 0.5,
        num_sample = 20,
        **kwargs
    ):
        super().__init__(model, **kwargs)
        self.beta = beta
        self.num_sample = num_sample

    @property
    def radius(self):
        return self.beta * self.budget

    def get_sampled_grad(self, ori_pcs, delta, labels, target=None):
        grad_sum = torch.zeros_like(ori_pcs)
        for _ in range(self.num_sample):
            perturb   = self.get_perturb(delta)
            logits    = self.get_logits(ori_pcs + delta + perturb)
            loss      = self.get_loss(logits, labels, target)
            grad_sum += self.get_grad(loss, delta)
        return grad_sum / self.num_sample

    def get_perturb(self, data, radius=None):
        radius = self.radius if radius is None else radius
        return torch.zeros_like(data).uniform_(-radius, radius)

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()

        for _ in range(self.num_iter):
            grad      = self.get_sampled_grad(ori_pcs, delta, labels, target=target)
            momentum  = self.update_momentum(momentum, grad)
            delta     = self.update_delta(delta, momentum.sign())
            delta     = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()
