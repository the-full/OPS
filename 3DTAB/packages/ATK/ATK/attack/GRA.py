import torch

from .attack_template import SampleAttack


class GRA(SampleAttack):
    ''' Gradient Relevance Attack (GRA).

    Ref: 
        - Boosting Adversarial Transferability via Gradient Relevance Attack (ICCV 2023)
        - https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Boosting_Adversarial_Transferability_via_Gradient_Relevance_Attack_ICCV_2023_paper.pdf
        - https://github.com/RYC-98/GRA
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/gra.py
    '''
    def __init__(
        self, 
        model, 
        beta = 1.0, 
        eta = 0.94, 
        num_sample = 20,
        **kwargs
    ):
        super().__init__(
            model, 
            beta = beta,
            num_sample = num_sample,
            **kwargs)
        self.eta  = eta

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta    = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()
        M        = torch.full_like(delta, 1.0 / self.eta)

        for _ in range(self.num_iter):
            logits   = self.get_logits(ori_pcs + delta)
            loss     = self.get_loss(logits, labels, target)
            cur_grad = self.get_grad(loss, delta)
            sam_grad = self.get_sampled_grad(ori_pcs, delta, labels, target)
            s        = self.get_cosine_similarity(cur_grad, sam_grad)
            grad     = s * cur_grad + (1 - s) * sam_grad

            last_sign = momentum.sign()
            momentum  = self.update_momentum(momentum, grad)
            curr_sign = momentum.sign()
            M = self.update_decay_indicator(M, curr_sign, last_sign)
            delta = self.update_delta(delta, M * curr_sign)
            delta = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()

    def get_cosine_similarity(self, current_grad, sampled_grad):
        B = current_grad.shape[0]
        current_grad = current_grad.reshape(B, -1)
        sampled_grad = sampled_grad.reshape(B, -1)
        sim = torch.cosine_similarity(current_grad, sampled_grad)
        sim = sim.unsqueeze(-1).unsqueeze(-1)
        return sim

    def update_decay_indicator(self, M, curr, last):
        decay_mat = torch.where(last == curr, 1.0, self.eta)
        M = M * decay_mat
        return M
