import torch

from .attack_template import SampleAttack


class SMIFGRM(SampleAttack):
    ''' Sampling-based Fast Gradient Rescaling Method (VMI-FGRM).

    Ref:
        - Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks
        - https://arxiv.org/abs/2307.02828
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/smifgrm.py
    '''
    def __init__(
        self, 
        model, 
        beta = 1.5,
        num_sample = 12,
        rescale_factor = 2,
        **kwargs,
    ):
        super().__init__(
            model, 
            beta = beta,
            num_sample = num_sample,
            **kwargs
        )
        self.rescale_factor = rescale_factor


    def rescale(self, grad):
        log_abs_grad = grad.abs().log2()
        grad_mean = torch.mean(log_abs_grad, dim=(1,2,3), keepdim=True)
        grad_std  = torch.std(log_abs_grad, dim=(1,2,3), keepdim=True)
        norm_grad = ((log_abs_grad - grad_mean) / grad_std + 1e-10)
        return self.rescale_factor * grad.sign() * torch.sigmoid(norm_grad)


    def get_sampled_grad(self, ori_pcs, delta, labels, target=None):
        sampled_pcs = ori_pcs
        grad_sum = torch.zeros_like(ori_pcs)
        for _ in range(self.num_sample):
            logits       = self.get_logits(sampled_pcs + delta)
            loss         = self.get_loss(logits, labels, target)
            grad_sum    += self.get_grad(loss, delta)
            perturb      = self.get_perturb(delta)
            sampled_pcs += perturb

        return grad_sum / self.num_sample


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()

        for _ in range(self.num_iter):
            grad     = self.get_sampled_grad(ori_pcs, delta, labels, target)
            momentum = self.update_momentum(momentum, grad)
            momentum = self.rescale(momentum)
            delta    = self.update_delta(delta, momentum.sign())
            delta    = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()
