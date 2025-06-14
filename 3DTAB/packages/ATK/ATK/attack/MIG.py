import torch
import torch.nn.functional as F

from .attack_template import IterAttack

class MIG(IterAttack):
    ''' Momentum Integrated Gradients (GIM).

    Ref: 
        - Transferable Adversarial Attack for Both Vision Transformers and Convolutional Networks via Momentum Integrated Gradients (ICCV 2023)
        - https://openaccess.thecvf.com/content/ICCV2023/papers/Ma_Transferable_Adversarial_Attack_for_Both_Vision_Transformers_and_Convolutional_Networks_ICCV_2023_paper.pdf
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/mig.py

    '''
    def __init__(
            self, 
            model, 
            alpha = 0.18,
            num_iter = 25,
            s_factor = 20, 
            **kwargs
        ):
        super().__init__(
            model, 
            alpha = alpha,
            num_iter = num_iter, 
            **kwargs
        )
        self.s = s_factor


    def set_budget(
        self, 
        budget: float = 0.45,
        budget_type: str = 'linfty',
        update_alpha: bool = True,
    ):
        super().set_budget(budget, budget_type, False)


    def get_loss(self, logits, labels, target=None):
        probs = F.softmax(logits, dim=-1)
        if target is None:
            loss = torch.mean(probs.gather(1, labels.view(-1, 1)))
        else:
            loss = -torch.mean(probs[:, target])
        return loss


    def get_grad_GI(self, ori_pcs, delta, labels, target=None):
        # NOTE: See paper Section 3.1
        # b = torch.zeros_like(delta)
        grad_sum = torch.zeros_like(delta)
        for k in range(self.s):
            logits    = self.get_logits(k / self.s * (ori_pcs + delta))
            loss      = self.get_loss(logits, labels, target)
            grad_sum += self.get_grad(loss, delta)
        return (ori_pcs + delta) * grad_sum / self.s


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta    = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)

        for _ in range(self.num_iter):
            i_grad   = self.get_grad_GI(ori_pcs, delta, labels, target)
            momentum = self.update_momentum(momentum, i_grad)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
