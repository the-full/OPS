import torch
import torch.nn as nn
from einops import rearrange
from pytorch3d.ops import (
    knn_points, 
    knn_gather,
)

from ATK.utils import ops
from ATK.utils.advloss import (
    CWAdvLoss,
    ChamferLoss,
)

from ..attack_template import CWAttack
from .loss import KerLoss, HideLoss


class HiTADV(CWAttack):
    ''' Hide-in-Thicket Attack (HiTADV)

    - Ref:
        - Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations 
          on 3D Point Clouds (CVPR 2024)
        - https://github.com/TRLou/HiT-ADV.git
        - https://arxiv.org/abs/2403.05247
    '''

    _attack_name = 'HiT-ADV'

    def __init__(
        self, 
        model, 
        kappa               = 30.,
        num_LCP             = 256,
        num_GCP             = 192,
        knn_k_for_SI        = 16,
        knn_k_for_LCP       = 16,
        alpha_for_saliency  = 1.0,
        weight_for_SI       = 1e-3,
        ker_loss_weight     = 1.0,
        a_for_ker_loss      = 1.0,
        hide_loss_weight    = 1.0,
        cha_loss_weight     = 1e-4,
        init_delta          = 0.55,
        min_sigma           = 0.1,
        max_sigma           = 1.2,
        **kwargs,
    ):
        kwargs.setdefault('binary_search_step', 10)
        kwargs.setdefault('inner_loop_max_iter', 500)
        kwargs.setdefault('initial_weight', 10.)
        kwargs.setdefault('upper_bound_weight', 80.)

        super().__init__(model, **kwargs)
        self.adv_loss_fn.add_objective(
            'adv_loss',
            CWAdvLoss(kappa),
            weight=1.0,
        )

        self.res_loss_fn.add_objective(
            'ker_loss',
            KerLoss(a=a_for_ker_loss),
            ker_loss_weight,
        )
        self.res_loss_fn.add_objective(
            'hide_loss',
            HideLoss(max_sigma, min_sigma),
            hide_loss_weight,
        )
        self.res_loss_fn.add_objective(
            'cha_loss',
            ChamferLoss(single=False, reduce='max'),
            cha_loss_weight,
        )

        assert num_LCP >= num_GCP
        self.num_LCP = num_LCP
        self.num_GCP = num_GCP
        self.knn_k_for_SI  = knn_k_for_SI
        self.knn_k_for_LCP = knn_k_for_LCP
        self.weight_for_SI = weight_for_SI
        self.alpha_for_saliency = alpha_for_saliency
        self.init_scale = init_delta
        self.min_sigma  = min_sigma
        self.max_sigma  = max_sigma


    def init_global_val(self, ori_pcs, ori_normals, labels):
        _g = self.global_val

        SI_score, _, S2_score = self.compute_SI_score(ori_pcs, ori_normals, labels) # <B, N>
        LCP_idx = self.choose_LCP(ori_pcs, SI_score) # <B, K(LCP)>
        GCP_idx = self.choose_GCP(LCP_idx, SI_score) # <B, K(GCP)>
        weight  = torch.gather(S2_score, dim=-1, index=GCP_idx) # <B, num_atk>

        _g.LCP_idx = LCP_idx
        _g.GCP_idx = GCP_idx
        _g.weight  = weight


    def set_pbar_info_inner(
        self, 
        adv_info, 
        res_info, 
        **kwargs
    ):
        super().set_pbar_info_inner(adv_info, res_info)
        if res_info is not None:
            self.pbar.set_infos({
                'ker_loss':  f'{res_info.ker_loss.mean().item():.4f}',
                'hide_loss': f'{res_info.hide_loss.mean().item():.4f}',
                'cha_loss':  f'{res_info.cha_loss.mean().item():.4f}',
            })


    def inner_loop(self, ori_pcs, labels, target=None, **kwargs):
        c_for_each_pc = self.coef_bound.c_for_each_pc
        device = self.device
        B  = ori_pcs.shape[0]
        _g = self.global_val

        delta = torch.randn(B, self.num_GCP, 3).mul_(self.init_scale).to(device)
        delta.requires_grad_(True)
        sigma = torch.ones(B, self.num_GCP).mul_(self.min_sigma).to(device)
        sigma = sigma + torch.rand((B, self.num_GCP)).to(device) * (self.max_sigma - self.min_sigma)
        sigma.requires_grad_(True)

        optimizer = torch.optim.Adam([
            {'params': delta, 'lr': self.lr * 5},
            {'params': sigma, 'lr': self.lr * 3},
        ], weight_decay=0.)

        for _ in self.pbar:
            adv_pcs = self.NW_kernel_reg(ori_pcs, _g.GCP_idx, delta, sigma) # <B, N, 3>
            centers = adv_pcs.mean(dim=1, keepdim=True) # <B, 1, 3>
            adv_pcs = adv_pcs - centers

            logits   = self.get_logits(adv_pcs)
            res_info = self.res_loss_fn(ker_loss = (delta, sigma))

            self.update_record_items(
                adv_pcs, res_info.ker_loss, labels, target=target, logits=logits
            )

            adv_info = self.adv_loss_fn(
                adv_loss = (logits, labels, target),
            ) # <B,>
            res_info = self.res_loss_fn(
                loss_info = res_info,
                hide_loss = (sigma, _g.weight),
                cha_loss  = (ori_pcs, adv_pcs),
            ) # <B,>

            adv_loss, res_loss = adv_info.loss, res_info.loss
            loss = torch.mean(adv_loss + c_for_each_pc * res_loss) # <1,>

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.set_pbar_info_inner(adv_info, res_info)


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        ori_normals = kwargs.get('normal', None)
        if ori_normals is None:
            ori_normals = self.estimate_pcs_normals(ori_pcs)
        self.init_global_val(ori_pcs, ori_normals, labels)

        delta = super().attack(ori_pcs, labels, target=target, **kwargs)
        return delta


    def compute_saliency_score(self, pcs, labels):
        pcs_copy = pcs.clone().requires_grad_(True)
        logits = self.get_logits(pcs_copy)
        loss = nn.CrossEntropyLoss()(logits, labels)
        grad = self.get_grad(loss, pcs_copy) # <B, N, 3>

        sphere_core = pcs_copy.median(dim=1, keepdims=True)[0] # <B, 1, 3>
        sphere_diff = pcs_copy - sphere_core # <B, N, 3>
        sphere_r    = torch.norm(sphere_diff, p=2, dim=-1) # <B, N>
        sphere_map  = -torch.einsum(
            'B N C, B N C, B N -> B N',
            grad, sphere_diff, torch.pow(sphere_r, self.alpha_for_saliency)
        ) # <B, N>
        return sphere_map


    def compute_imperceptiblity_score(self, pcs, normals):
        k = self.knn_k_for_SI
        _, knn_idx, neighbors = knn_points(pcs, pcs, K=k+1, return_nn=True) # <B, N, K+1, 3>
        neighbors = neighbors[:, :, 1:, :].contiguous() # <B, N, K, 3>

        offset = neighbors - pcs.unsqueeze(2) # <B, N, K, 3>
        offset = offset / torch.norm(offset, dim=-1, keepdim=True).clamp(min=1e-12) # <B, N, K, 1>

        normals = normals.unsqueeze(2) # <B, N, 1, 3>
        dot_product = torch.abs(torch.einsum("B N k C, B N k C -> B N k", offset, normals)) # <B, N, K>
        curvature   = dot_product.mean(-1, keepdim=True) # <B, N, 1>

        neighbors_curvature = knn_gather(curvature, knn_idx) # <B, N, K+1, 1>
        neighbors_curvature = neighbors_curvature[:, :, 1:, :].contiguous() # <B, N, K, 1>

        imperceptiblity_score = neighbors_curvature.squeeze(-1).std(-1) # <B, N>

        return imperceptiblity_score


    def compute_SI_score(self, pcs, normals, labels):
        S1_score = self.compute_saliency_score(pcs, labels) # <B, N>
        S2_score = self.compute_imperceptiblity_score(pcs, normals) # <B, N>

        S1_score = (S1_score - S1_score.min()) / (S1_score.max() - S1_score.min() + 1e-7) # <B, N>
        S2_score = (S2_score - S2_score.min()) / (S2_score.max() - S2_score.min() + 1e-7) # <B, N>

        SI_score = self.weight_for_SI * S1_score + S2_score # <B, N>
        return SI_score, S1_score, S2_score


    def choose_LCP(self, pcs, SI_score):
        fps_pcs, _ = ops.farthest_point_sample(pcs, k=self.num_LCP) # <B, L, 3>
        knn_idx = knn_points(fps_pcs, pcs, K=self.knn_k_for_LCP + 1).idx # <B, L, K>
        knn_idx = rearrange(knn_idx, 'B L K -> B (L K)') # <B, L * K>,
        SI_score = torch.gather(SI_score, dim=-1, index=knn_idx) # <B, L * K>
        SI_score = rearrange(SI_score, 'B (L K) -> B L K', L=self.num_LCP) # <B, L, K>
        LCP_idx  = torch.argmax(SI_score, dim=-1) # <B, L>
        return LCP_idx


    def choose_GCP(self, LCP_idx, SI_score):
        SI_score = torch.gather(SI_score, dim=-1, index=LCP_idx) # <B, K>
        _, GCP_idx = torch.topk(SI_score, k=self.num_GCP, dim=-1, largest=True, sorted=True) # <B, G>
        return GCP_idx


    def NW_kernel_reg(self, ori_pcs, GCP_idx, delta, sigma):
        gcp_pcs = ops.get_pcs_by_idx(ori_pcs, GCP_idx) # <B, num_atk, 3>

        kernel_val = self.gauss_kernel_reg(gcp_pcs, ori_pcs, sigma) # <B, N, num_atk>
        kernel_val = kernel_val.unsqueeze(-1) # B N num_GCP -> B N num_atk 1
        delta   = delta.unsqueeze(1)   # B num_GCP 3 -> B 1 num_atk 3
        perturb = torch.einsum('B N G C, B N G C -> B N C', kernel_val, delta) # <B, N, 3>

        new_pc = ori_pcs + perturb
        return new_pc


    @staticmethod
    def gauss_kernel_reg(gcp_pcs, pcs, sigma):
        sigma = sigma.unsqueeze(1) # B K -> B 1 K

        norm_between_pcs = torch.cdist(pcs, gcp_pcs) # <B, K, k>
        return torch.softmax(-norm_between_pcs / (2 * torch.square(sigma)), dim=-1)
