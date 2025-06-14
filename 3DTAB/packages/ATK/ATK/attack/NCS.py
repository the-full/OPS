import torch 

from .attack_template import SampleAttack


class NCS(SampleAttack):
    ''' Neighborhood Conditional Sampling (NCS). 

    Ref:
        - Enhancing Adversarial Transferability Through Neighborhood Conditional Sampling
        - https://arxiv.org/abs/2405.16181
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/gradient/ncs.py
    '''
    def __init__(
        self, 
        model, 
        beta = 1.0,
        num_sample = 20,
        gamma = 0.15,
        lambda_ = 0.16/255,
        **kwargs,
    ):
        super().__init__(
            model, 
            beta = beta,
            num_sample = num_sample,
            **kwargs
        )
        self.gamma = gamma
        self.lambda_ = lambda_

    def get_conditional_sampled_points(self, delta, grad_pgia):
        sample_delta = delta + torch.zeros_like(grad_pgia).uniform_(-self.radius, self.radius)
        sample_delta = sample_delta + self.gamma * grad_pgia
        return sample_delta

    def get_points_gradient(self, ori_pcs, delta, label, target=None, **kwargs):
        B, N, C = ori_pcs.shape
        loss_list = torch.zeros([self.num_sample, B]).to(self.device)
        grad_list = torch.zeros([self.num_sample, B, N, C]).to(self.device)
        for i in range(self.num_sample):
            x_near = ori_pcs + delta[i]
            logits = self.get_logits(x_near)
            loss_list[i] = self.get_loss(logits, label, target)
            grad_list[i] = self.get_grad(loss_list[i].mean(), x_near)

        grad = (1/self.num_sample)*grad_list - (self.lambda_)*(2*(self.num_sample-1)/(self.num_sample**2))*(loss_list - loss_list.mean(0).view(1,B)).view(self.num_sample,B,1,1)*grad_list
        return grad

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta     = self.init_delta(ori_pcs)
        momentum  = torch.zeros_like(ori_pcs).detach()
        grad_pgia = torch.zeros([self.num_sample, *ori_pcs.shape]).to(self.device)

        for _ in range(self.num_iter):
            sample_delta = self.get_conditional_sampled_points(delta, grad_pgia)
            gradient  = self.get_points_gradient(ori_pcs, sample_delta, labels, target)
            grad_pgia = -grad_pgia + gradient / torch.mean(torch.abs(gradient), (2, 3), keepdim=True)
            momentum  = self.update_momentum(momentum, gradient.sum(0))
            delta     = self.update_delta(delta, momentum.sign())
            delta     = self.proj_delta(ori_pcs, delta, **kwargs)
        return delta.detach()


