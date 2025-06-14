import torch

from ATK.utils.ops import offset_proj
from ATK.utils.advloss import (
    ChamferLoss,
    NegtiveAdvLoss, 
)
from .loss import PFLoss
from ..attack_template import CWAttack 


class PFAttack(CWAttack):
    ''' Perturbation Factorization(PF) Attack

    Ref:
        - Generating Transferable 3D Adversarial Point Cloud via Random Perturbation Factorization (AAAI 2023)
        - https://github.com/HeBangYan/PF-Attack
        - https://ojs.aaai.org/index.php/AAAI/article/view/25154
    '''

    _attack_name = 'PF-Attack'

    def __init__(
        self,
        model,
        pf_loss_weight = 0.5,
        decomp_num     = 2,
        **kwargs,
    ):
        kwargs.setdefault('attack_lr', 0.01)
        kwargs.setdefault('initial_weight', 10)
        kwargs.setdefault('binary_search_step', 1)
        kwargs.setdefault('inner_loop_max_iter', 200)
        super().__init__(model, **kwargs)

        self.adv_loss_fn.add_objective(
            'adv_loss',
            NegtiveAdvLoss(),
            weight=1.0,
        )

        self.res_loss_fn.add_objective(
            'cha_loss',
            ChamferLoss(single=False, reduce='sum'),
            weight=1.0,
        )
        self.res_loss_fn.add_objective(
            'pf_loss',
            PFLoss(model, decomp_num=decomp_num),
            pf_loss_weight,
        )


    def set_task(
        self,
        model,
        device: str = 'cuda',
    ):
        super().set_task(model, device)
        self.res_loss_fn.objectives['pf_loss'].model = model


    def set_pbar_info_inner(
        self, 
        adv_info, 
        res_info, 
        **kwargs
    ):
        super().set_pbar_info_inner(adv_info, res_info)
        if res_info is not None:
            self.pbar.set_infos({
                'cha_loss': f'{res_info.cha_loss.mean().item():.4f}',
                'pf_loss':  f'{res_info.pf_loss.mean().item():.4f}',
            })


    def binary_search_coef(self):
        _c = self.coef_bound
        attack_success = self.outer_attack_achieve()

        B = _c.c_for_each_pc.shape[0]
        for k in range(B):
            if attack_success[k]:
                _c.c_lower_bound[k] = max(_c.c_for_each_pc[k], _c.c_lower_bound[k])
                if _c.c_upper_bound[k] < 1e9:
                    _c.c_for_each_pc[k] = (_c.c_lower_bound[k] + _c.c_upper_bound[k]) * 0.5
                else:
                    _c.c_for_each_pc[k] *= 2
            else:
                _c.c_upper_bound[k] = min(_c.c_for_each_pc[k], _c.c_upper_bound[k])
                if _c.c_upper_bound[k] < 1e9:
                    _c.c_for_each_pc[k] = (_c.c_lower_bound[k] + _c.c_upper_bound[k]) * 0.5


    def inner_loop(self, ori_pcs, labels, target=None, **kwargs):
        c_for_each_pc = self.coef_bound.c_for_each_pc
        device = self.device

        offset = torch.zeros(ori_pcs.shape).to(device)
        offset.normal_(mean=0, std=1e-3).requires_grad_(True)

        optimizer = torch.optim.Adam([offset], lr=self.lr)

        for _ in self.pbar:
            adv_pcs  = ori_pcs + offset # <B, N, C>
            logits   = self.get_logits(adv_pcs)
            adv_info = self.adv_loss_fn(
                adv_loss = (logits, labels, target),
            ) # <B,>
            res_info = self.res_loss_fn(
                cha_loss = (ori_pcs, adv_pcs),
                pf_loss  = (ori_pcs, adv_pcs, labels),
            ) # <B,>

            adv_loss, res_loss = adv_info.loss, res_info.loss
            loss = torch.mean(adv_loss + c_for_each_pc * res_loss) # <1,>

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.set_pbar_info_inner(adv_info, res_info)
            self.update_record_items(adv_pcs, res_info.loss, labels, target=target)

            ori_normals = kwargs.get('normal', None)
            offset.data = self.proj_offset(
                ori_pcs, adv_pcs, offset.data, ori_normals, **kwargs
            )


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        ori_normals = kwargs.get('normal', None)
        if ori_normals is None:
            ori_normals = self.estimate_pcs_normals(ori_pcs)
            kwargs['normal'] = ori_normals

        delta = super().attack(ori_pcs, labels, target=target, **kwargs)
        return delta


    def proj_offset(
        self, 
        ori_pcs, 
        adv_pcs, 
        offset, 
        ori_normals=None, 
        **kwargs
    ):
        with torch.no_grad():
            assert ori_normals is not None
            offset = offset_proj(offset, ori_pcs, ori_normals)
        return self.proj_delta(ori_pcs, offset, **kwargs)
