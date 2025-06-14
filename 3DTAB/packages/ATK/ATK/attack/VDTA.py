# TODO:
import torch

from .attack_template import SampleAttack


class VDTA(SampleAttack):
    ''' Variance tuning-based Direction Tuning Attack (VDTA).

    Ref: 
        - Improving the Transferability of Adversarial Examples via Direction Tuning (INS 2023).
        - https://arxiv.org/abs/2303.15109
        - https://github.com/HaloMoto/DirectionTuningAttack
    '''
    def __init__(
            self, 
            model, 
            K = 10, 
            mu1 = 1.0,
            mu2 = 0.8,
            beta = 1.5,
            num_sample = 20,
            **kwargs
        ):
        super().__init__(
            model, 
            beta=beta, 
            num_sample=num_sample, 
            **kwargs
        )
        self.K = K
        self.mu1 = mu1
        self.mu2 = mu2

    def look_ahead(self, x, direction):
        return x + self.alpha * self.mu1 * direction 

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs)
        variance = torch.zeros_like(ori_pcs).detach()

        for _ in range(self.num_iter):
            m_tk = momentum.clone().detach()
            v_tk = variance.clone().detach()
            delta_tk = delta.clone().detach()
            delta_tk.requires_grad_(True)

            grad_sum = torch.zeros_like(ori_pcs)
            for _ in range(self.K):
                delta_nes = self.look_ahead(delta_tk, m_tk)
                logits    = self.get_logits(ori_pcs + delta_nes)
                loss      = self.get_loss(logits, labels, target)
                grad      = self.get_grad(loss, delta_tk)
                m_tk      = self.mu2 * m_tk + self.norm_grad(grad + v_tk)
                v_tk      = self.get_sampled_grad(ori_pcs, delta_tk, labels, target) - grad
                delta_tk  = self._update_delta(delta_tk, self.alpha / self.K, m_tk.sign())
                delta_tk  = self.proj_delta(ori_pcs, delta_tk, **kwargs)
                grad_sum += m_tk

            grad = grad_sum / self.K
            momentum = self.mu1 * momentum + grad
            delta = self.update_delta(delta, momentum.sign())
            delta = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()
