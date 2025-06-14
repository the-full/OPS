import torch

from .attack_template import IterAttack


class PCIFGSM(IterAttack):
    ''' Prediction-Correction I-FGSM (PC-I-FGSM).

    Ref: 
        - Adversarial Attack Based on Prediction-Correction
        - https://arxiv.org/abs/2306.01809
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/pcifgsm.py
    '''
    def __init__(self, model, K=1, **kwargs):
        super().__init__(model, **kwargs)
        self.K = K

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        G = torch.zeros_like(ori_pcs)

        for _ in range(self.num_iter):
            delta_pre = self.init_delta(ori_pcs)

            logits = self.get_logits(ori_pcs + delta)
            loss   = self.get_loss(logits, labels, target)
            grad   = self.get_grad(loss, delta, norm=1)
            g_pre  = grad.clone()

            for _ in range(self.K):
                delta_pre = self.update_delta(delta_pre, grad.sign())
                delta_pre = self.proj_delta(ori_pcs, delta_pre, **kwargs)

                logits = self.get_logits(ori_pcs + delta + delta_pre)
                loss   = self.get_loss(logits, labels, target)
                grad   = self.get_grad(loss, delta_pre, norm=1)
                g_pre  = self.mu * g_pre + grad / self.K

            G = grad + g_pre
            delta = self.update_delta(delta, G.sign())
            delta = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()
