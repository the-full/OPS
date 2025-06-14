import torch
import torch.nn as nn

from .basic_attack import BasicAttack


class IterAttack(BasicAttack):
    def __init__(
        self, 
        model, 
        num_iter = 10,
        momentum_decay = 1.0,
        alpha = None,
        loss_fn = nn.CrossEntropyLoss(),
        loss_type = 'neg',
        **kwargs,
    ):
        kwargs.setdefault('budget', 0.045)
        kwargs.setdefault('budget_type', 'linfty')
        super().__init__(model, **kwargs)

        # NOTE: attack setting
        self.loss_fn   = loss_fn
        self.loss_type = loss_type
        self.num_iter  = num_iter
        if alpha is None:
            self.alpha = self.budget / num_iter
            self.update_alpha = True
        else:
            self.alpha = alpha
            self.update_alpha = False
        self.mu = momentum_decay


    def set_budget(
        self, 
        budget: float = 0.045,
        budget_type: str = 'linfty',
    ):
        super().set_budget(budget, budget_type)
        if self.update_alpha:
            self.alpha = budget / self.num_iter


    def get_loss(self, logits, labels, target=None): 
        if self.loss_type == 'neg':
            if target is None:
                return  self.loss_fn(logits, labels)
            else:
                target = torch.tensor(target).long().to(self.device)
                target = target.expand_as(labels)
                return -self.loss_fn(logits, target)
        elif self.loss_type == 'logits':
            b_idx = torch.arange(logits.shape[0])
            return -logits[b_idx, labels].mean()
        else:
            raise NotImplementedError


    def update_momentum(self, momentum, grad, **kwargs):
        return momentum * self.mu + self.norm_grad(grad)


    def update_delta(self, delta, direction, **kwargs):
        return self._update_delta(delta, self.alpha, direction)


    @staticmethod
    def _update_delta(delta, step_size, direction):
        return delta + step_size * direction

    def get_perturb(self, data, radius):
        return torch.rand_like(data).uniform_(-radius, radius)
