import torch

from .attack_template import SampleAttack

class VMIFGSM(SampleAttack):
    ''' Variance tuning Momentum Iteration FGSM (VMI-FGSM).

    Ref:
        - Enhancing the Transferability of Adversarial Attacks through Variance Tuning (CVPR 2021)
        - https://arxiv.org/abs/2103.15571.
        - https://github.com/thu-ml/ares/blob/main/ares/attack/vmi_fgsm.py
        - https://github.com/JHL-HUST/VT/blob/main/vmi_fgsm.py
    '''
    def __init__(
        self, 
        model, 
        beta = 2.0,
        num_sample = 20,
        **kwargs,
    ):
        super().__init__(
            model, 
            beta = beta,
            num_sample = num_sample,
            **kwargs
        )

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()
        variance = torch.zeros_like(ori_pcs).detach()

        for _ in range(self.num_iter):
            logits    = self.get_logits(ori_pcs + delta)
            loss      = self.get_loss(logits, labels, target)
            grad      = self.get_grad(loss, delta)
            momentum  = self.update_momentum(momentum, grad + variance)
            variance  = self.get_sampled_grad(ori_pcs, delta, labels, target) - grad
            delta     = self.update_delta(delta, momentum.sign())
            delta     = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()
