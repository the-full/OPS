import torch

from .attack_template import SampleAttack

class PGN(SampleAttack):
    ''' Penalizing Gradient Norm (PGN).

    Ref: 
        - Boosting Adversarial Transferability by Achieving Flat Local Maxima (NeurIPS 2023)
        - https://arxiv.org/abs/2306.05225
        - https://github.com/Trustworthy-AI-Group/PGN
    '''
    def __init__(
        self, 
        model, 
        beta       = 2.0,
        lambda_    = 0.5,
        num_sample = 20
    ):
        super().__init__(
            model,
            beta = beta,
            num_sample = num_sample
        )
        self.lambda_ = lambda_

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)

        for _ in range(self.num_iter):
            grad_bar = self.get_grad_bar(ori_pcs, delta, labels, target)
            momentum = self.update_momentum(momentum, grad_bar)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()

    def get_grad_bar(self, ori_pcs, delta, labels, target=None):
        alpha    = self.alpha
        grad_bar = torch.zeros_like(ori_pcs)
        for _ in range(self.num_sample):
            perturb    = self.get_perturb(delta)
            logits     = self.get_logits(ori_pcs + delta + perturb)
            loss       = self.get_loss(logits, labels, target)
            grad_near  = self.get_grad(loss, delta)

            direction  = -self.norm_grad(grad_near)
            logits     = self.get_logits(ori_pcs + delta + alpha * direction)
            loss       = self.get_loss(logits, labels, target)
            grad_star  = self.get_grad(loss, delta)
            grad_bar  += (1 - self.lambda_) * grad_near + self.lambda_ * grad_star
        return grad_bar / self.num_sample
