import torch
from pytorch3d.ops import knn_points

from ATK.utils.common import check_option
from ATK.utils.advloss import (
    NegtiveAdvLoss, 
    CWAdvLoss,
)
from ATK.utils.metric import l2_norm_distance

from ..attack_template import CWAttack 


class AOF(CWAttack):
    _attack_name = 'AOF'

    def __init__(
        self,
        model,
        adv_loss_type = 'CW',
        kappa = 30.,
        gamma = 0.25,
        low_pass = 100,
        **kwargs,
    ):
        kwargs.setdefault('attack_lr', 0.01)
        kwargs.setdefault('binary_search_step', 2)
        kwargs.setdefault('inner_loop_max_iter', 200)
        super().__init__(model, **kwargs)

        check_option(adv_loss_type, ['CE', 'CW'])

        self.adv_loss_fn.add_objective(
            'adv_loss',
            self.parse_adv_loss_fn(adv_loss_type, kappa=kappa),
            weight = 1.0 - gamma,
        )
        self.adv_loss_fn.add_objective(
            'lfc_loss',
            self.parse_adv_loss_fn(adv_loss_type, kappa=kappa),
            weight = gamma,
        )

        self.low_pass = low_pass
        self.gamma = gamma


    @staticmethod
    def parse_adv_loss_fn(adv_loss_type, **kwargs):
        adv_loss_map = {
            'CE': NegtiveAdvLoss,
            'CW': CWAdvLoss,
        }
        Loss = adv_loss_map[adv_loss_type]
        if adv_loss_type == 'CW':
            kappa = kwargs.get('kappa', 0.)
            return Loss(kappa)
        else:
            return Loss()

    def set_pbar_info_outer(self, binary_step, **kwargs):
        self.pbar = self.get_cw_pbar(self._attack_name)
        _r = self.record_items
        if hasattr(_r, 'final_success'):
            num_suc = _r.final_success.sum().item()
            num_all = len(_r.final_success)
            asr = num_suc / num_all
            self.pbar.set_infos({
                'ASR:':   f'{num_suc}/{num_all} = {asr*100:.2f}%',
            })

    def set_pbar_info_inner(
        self, 
        adv_info, 
        res_info, 
        **kwargs
    ):
        super().set_pbar_info_inner(adv_info, res_info)
        self.pbar.set_infos({
            'adv_loss': f'{adv_info.adv_loss.mean().item():.4f}',
            'lfc_loss': f'{adv_info.lfc_loss.mean().item():.4f}',
        })


    @torch.no_grad()
    def update_record_items(
        self, 
        adv_pcs, 
        dist_val, 
        labels,
        target=None,
        **kwargs
    ):
        _r = self.record_items

        lfc_logits = kwargs.get('lfc_logits') # <B, num_class>
        adv_logits = kwargs.get('adv_logits') # <B, num_class>
        assert lfc_logits is not None and adv_logits is not None

        adv_attack_success = self.attack_achieved(adv_logits, labels, target) # <B,>
        lfc_attack_success = self.attack_achieved(lfc_logits, labels, target=None) # <B,>
        attack_success = adv_attack_success & (lfc_attack_success | (self.gamma < 0.001))

        update_mask = attack_success & (dist_val < _r.best_dist_val)
        _r.best_dist_val = torch.where(update_mask, dist_val, _r.best_dist_val)
        _r.final_success = torch.where(update_mask, True, _r.final_success)
        _r.best_attack[update_mask] = adv_pcs[update_mask].detach().cpu()


    def inner_loop(self, ori_pcs, labels, target=None, **kwargs):
        adv_pcs = ori_pcs.clone().detach() + torch.randn_like(ori_pcs) * 1e-7
        _, V = self.get_Laplace_from_pcs(adv_pcs) # <B, N, N>
        lfc, hfc = self.get_lfc_and_hfc(adv_pcs, V=V)
        lfc.requires_grad_(True)

        # optimizer
        optimizer = torch.optim.Adam([lfc], lr=self.lr, weight_decay=0.)

        for _ in self.pbar:
            adv_pcs = lfc + hfc

            adv_logits = self.get_logits(adv_pcs)
            lfc_logits = self.get_logits(lfc)
            adv_info = self.adv_loss_fn(
                adv_loss = (adv_logits, labels, target),
                lfc_loss = (lfc_logits, labels, target),
            ) # <B,>

            loss = torch.mean(adv_info.loss) # <1,>

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.set_pbar_info_inner(adv_info, self.res_loss_fn())

            with torch.no_grad():
                adv_pcs = lfc + hfc
                delta = adv_pcs - ori_pcs
                delta = self.proj_delta(ori_pcs, delta, **kwargs)
                adv_pcs = ori_pcs + delta
                lfc.data, hfc = self.get_lfc_and_hfc(adv_pcs, V=V)

            with torch.no_grad():
                adv_logits = self.get_logits(adv_pcs)
                lfc_logits = self.get_logits(lfc)

            dist_val = l2_norm_distance(ori_pcs, adv_pcs)

            self.update_record_items(
                adv_pcs, dist_val, labels, 
                target=target, adv_logits=adv_logits, lfc_logits=lfc_logits,
            )

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        self.init_record_items(ori_pcs)

        for bs in range(self.binary_search_step):
            self.set_pbar_info_outer(bs)
            self.inner_loop(ori_pcs, labels, target=target, **kwargs)

        delta = self.record_items.best_attack.to(self.device) - ori_pcs
        return delta.detach()


    @staticmethod
    @torch.no_grad()
    def get_Laplace_from_pcs(pcs, k=30):
        """
        pcs:(B, N, 3)
        """
        knn_idx = knn_points(pcs, pcs, K=k).idx # <B, N, k>
        dists   = torch.cdist(pcs, pcs, p=2).square() # <B, N, N>

        A = torch.exp(-dists) # <B, N, N>
        mask = torch.zeros_like(A) # <B, N, N>
        mask.scatter_(2, knn_idx, 1) # <B, N, N>
        mask = mask + mask.transpose(2, 1)
        mask[mask>1] = 1
        A = A * mask

        D = torch.diag_embed(torch.sum(A, dim=2)) # <B, N, N>
        L = D - A
        e, v = torch.linalg.eigh(L, UPLO="U")
        return e.to(pcs), v.to(pcs) # <B, N, N>


    @torch.no_grad()
    def get_lfc_and_hfc(self, pcs, V=None):
        """
        pcs:(B, N, 3)
        """
        # 
        if V is None:
            _, V = self.get_Laplace_from_pcs(pcs) # <B, N, N>
        
        projs = torch.bmm(V.transpose(2, 1), pcs) # <B, N, 3>
        hfc = torch.bmm(V[..., self.low_pass:], projs[:, self.low_pass:, :]) # <B, N, N>
        lfc = torch.bmm(V[..., :self.low_pass], projs[:, :self.low_pass, :]) # <B, N, N>
        return lfc, hfc
