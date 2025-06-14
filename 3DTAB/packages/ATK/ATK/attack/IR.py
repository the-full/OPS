import torch
import torch.nn.functional as F

from ATK.utils import ops

from .attack_template import IterAttack

class IR(IterAttack):
    """ Interaction-Reduced (IR) Attack.

    Ref:
        - Interpreting and Boosting Dropout from a Game-Theoretic View (ICLR 2021) [1]
        - https://arxiv.org/abs/2009.11729
        - A Unified Approach to Interpreting and Boosting Adversarial Transferability (ICLR 2021) [2]
        - https://arxiv.org/abs/2010.04055
        - https://zhuanlan.zhihu.com/p/369883667
        - https://github.com/xherdan76/A-Unified-Approach-to-Interpreting-and-Boosting-Adversarial-Transferability
        - https://github.com/Trustworthy-AI-Group/TransferAttack/blob/main/transferattack/advanced_objective/ir.py

    """

    def __init__(
        self, 
        model, 
        region_num = 64,
        sample_num = 16,
        times      = 16,
        lambda_    = 1.0,
        num_iter   = 20,
        **kwargs
    ):
        super().__init__(model, num_iter=num_iter, **kwargs)
        self.region_num = region_num
        self.sample_num = sample_num
        self.times      = times
        self.lambda_    = lambda_

    def setup(self, model):
        self.model = model

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        delta = self.init_delta(ori_pcs)
        momentum = torch.zeros_like(ori_pcs).detach()

        for _ in range(self.num_iter):
            logits    = self.get_logits(ori_pcs + delta)
            loss      = self.get_loss(logits, labels, target)
            grad1     = self.get_grad(loss, delta)
            grad2     = self.get_interaction_grad(ori_pcs, delta, labels, transfer_target=target)
            grad      = self.norm_grad(grad1 - self.lambda_ * grad2)
            momentum  = self.update_momentum(momentum, grad)
            delta     = self.update_delta(delta, momentum.sign())
            delta     = self.proj_delta(ori_pcs, delta, **kwargs)

        return delta.detach()

    def get_interaction_grad(self, ori_pcs, delta, labels, transfer_target=None):
        device = ori_pcs.device
        B, _, C = ori_pcs.shape
        adv_pcs = ori_pcs + delta
        def get_mask_generator(pcs):
            region_centers, _ = ops.farthest_point_sample(pcs, self.region_num) # <B, K, C>
            _, region_idx = ops.knn_points(pcs, region_centers, k=1) # <B, N, 1>
            region_idx = region_idx.squeeze() # <B, N>

            while True:
                mask = torch.zeros_like(region_idx) # <B, N>
                picked_regions = torch.randint(self.region_num, size=(B, self.sample_num), device=device) # <B, S>
                for j in range(self.sample_num):
                    picked_region = picked_regions[:, j].unsqueeze(-1) # <B, 1>
                    mask = torch.where(region_idx == picked_region, 1, mask) # <B, N>
                mask = mask.unsqueeze(-1).repeat(1, 1, C) # <B, N, C>
                yield mask

        b_idx  = torch.arange(B, device=ori_pcs.device)

        full_unit_logits = self.get_logits(adv_pcs)
        if transfer_target is None:
            labels_one_hot = F.one_hot(labels, num_classes=full_unit_logits.shape[1])
            transfer_target  = torch.argmax(full_unit_logits - 1e10 * labels_one_hot, dim=-1)
        full_unit_loss   = full_unit_logits[b_idx, transfer_target] - full_unit_logits[b_idx, labels]
        full_unit_grad   = self.get_grad(full_unit_loss.sum(), adv_pcs)

        final_grad1 = full_unit_grad 

        final_grad2 = torch.zeros_like(final_grad1)
        mask_generator = get_mask_generator(ori_pcs)
        for _ in range(self.times):
            picked_unit_mask = next(mask_generator) # <B, N, C>
            others_unit_mask = 1 - picked_unit_mask
            
            picked_unit_logits = self.get_logits(ori_pcs + picked_unit_mask * delta)
            picked_unit_loss   = picked_unit_logits[b_idx, transfer_target] - picked_unit_logits[b_idx, labels]
            picked_unit_grad   = self.get_grad(picked_unit_loss.sum(), delta)

            others_unit_logits = self.get_logits(ori_pcs + others_unit_mask * delta)
            others_unit_loss   = others_unit_logits[b_idx, transfer_target] - others_unit_logits[b_idx, labels]
            others_unit_grad   = self.get_grad(others_unit_loss.sum(), delta)
            
            final_grad2 += picked_unit_grad + others_unit_grad

        final_grad = final_grad1 - (final_grad2 / self.times)
        return final_grad
