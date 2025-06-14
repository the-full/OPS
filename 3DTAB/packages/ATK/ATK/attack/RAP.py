import torch

from .attack_template import IterAttack


class RAP(IterAttack):
    ''' Reverse Adversarial Perturbation (RAP). 

    Ref: 
        - Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation (NeurIPS 2022)
        - https://arxiv.org/abs/2210.05968
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/rap.py
    '''

    def __init__(
        self, 
        model, 
        num_iter=400,
        num_late_start=100,
        adv_steps=8,
        alpha_n=None,
        budget_n=None,
    ):
        super().__init__(model, num_iter=num_iter)
        self.model     = model
        self.num_LS    = num_late_start
        self.adv_steps = adv_steps
        self.budget_n  = self.budget if budget_n is None else budget_n
        self.alpha_n   = self.budget_n / adv_steps if alpha_n is None else alpha_n

    def set_budget(
        self, 
        budget: float = 0.45,
        budget_type: str = 'linfty',
        update_alpha: bool = True,
        alpha_n = None,
        budget_n = None,
    ):
        super().set_budget(budget, budget_type, update_alpha)
        self.budget_n = self.budget if budget_n is None else budget_n
        self.alpha_n  = self.budget_n / self.adv_steps if alpha_n is None else alpha_n

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta    = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)
        n_rap    = torch.zeros_like(ori_pcs)
        
        for iter in range(self.num_iter):
            if iter >= self.num_LS:
                n_rap = self.get_n_rap(ori_pcs, delta, labels)

            logits   = self.get_logits(ori_pcs + delta + n_rap)
            loss     = self.get_loss(logits, labels, target)
            grad     = self.get_grad(loss, delta)
            momentum = self.update_momentum(momentum, grad)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()

    def init_n_rap(self, pcs):
        delta = torch.zeros_like(pcs).detach().to(self.device)
        delta.uniform_(-self.budget, self.budget)
        delta.requires_grad = True
        return delta

    def get_n_rap(self, ori_pcs, labels, target=None, **kwargs):
        n_rap = self.init_n_rap(ori_pcs)

        for _ in range(self.adv_steps):
            logits = self.get_logits(ori_pcs + n_rap)
            loss   = self.get_loss(logits, labels, target)
            grad   = self.get_grad(loss, n_rap)

            n_rap = self._update_delta(n_rap, self.alpha_n, -grad.sign())

            temp = self.budget
            self.budget = self.budget_n
            n_rap = self.proj_delta(ori_pcs, n_rap, **kwargs)
            self.budget = temp

        return n_rap.detach()
