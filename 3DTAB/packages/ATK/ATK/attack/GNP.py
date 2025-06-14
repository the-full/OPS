import torch

from .attack_template import IterAttack

class GNP(IterAttack):
    ''' Gradient Norm Penalty (GNP). 


    Ref: 
        - GNP Attack: Transferable Adversarial Examples via Gradient Norm Penalty (ICIP 2023)
        - https://ieeexplore.ieee.org/abstract/document/10223158
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/gnp.py

    Note:
        - The version implemented here is MI-FGSM + GNP.
    '''
    def __init__(
        self, 
        model, 
        device = 'cuda',
        lambda_ = 0.8,
        r = 0.01,
    ):
        super().__init__(
            model,
            device = device,
        )
        self.lambda_ = lambda_
        self.r = r

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)

        for _ in range(self.num_iter):
            logits    = self.get_logits(ori_pcs + delta)
            loss      = self.get_loss(logits, labels, target)
            grad1     = self.get_grad(loss, delta)

            direction = self.norm_grad(grad1)
            logits    = self.get_logits(ori_pcs + delta + self.r * direction)
            loss      = self.get_loss(logits, labels, target)
            grad2     = self.get_grad(loss, delta)

            grad      = (1 + self.lambda_) * grad1 - self.lambda_ * grad2
            momentum  = self.update_momentum(momentum, grad)
            delta     = self.update_delta(delta, momentum.sign())
            delta     = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()

