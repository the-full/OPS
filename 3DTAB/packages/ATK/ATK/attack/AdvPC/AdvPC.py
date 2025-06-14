import os.path as osp

import torch

from ATK.utils.common import check_option, load_weights_to_model
from ATK.utils.advloss import (
    NegtiveAdvLoss, 
    CWAdvLoss,
)
from ATK.utils.metric import l2_norm_distance

from ..attack_template import CWAttack 


_cur_dir = osp.dirname(osp.abspath(__file__))
_ae_ckpt = osp.join(_cur_dir, 'pretain/mn40/AE/2021-12-31 15:15:52_1024/BEST_model9800_CD_0.0038.pth')

class AdvPC(CWAttack):
    _attack_name = 'AdvPC'

    def __init__(
        self,
        model,
        ae_model = _ae_ckpt,
        adv_loss_type = 'CW',
        kappa = 30.,
        gamma = 0.25,
        **kwargs,
    ):
        kwargs.setdefault('binary_search_step', 2)
        kwargs.setdefault('inner_loop_max_iter', 200)
        kwargs.setdefault('attack_lr', 0.01)
        super().__init__(model, **kwargs)

        check_option(adv_loss_type, ['CE', 'CW'])

        self.adv_loss_fn.add_objective(
            'adv_loss',
            self.parse_adv_loss_fn(adv_loss_type, kappa=kappa),
            weight = 1.0 - gamma,
        )
        self.adv_loss_fn.add_objective(
            'rec_loss',
            self.parse_adv_loss_fn(adv_loss_type, kappa=kappa),
            weight = gamma,
        )
        self.ae_model = self.parse_ae_model(ae_model).to(self.device).eval()
        self.gamma = gamma


    @staticmethod
    def parse_ae_model(ae_model): 
        if isinstance(ae_model, str):
            ckpt_path = ae_model
            from . import encoders_decoders
            model = encoders_decoders.AutoEncoder(3)
            load_weights_to_model(model, ckpt_path)
            return model
        elif isinstance(ae_model, torch.nn.Module):
            return ae_model
        else:
            raise TypeError('ae_model must be path or a torch.nn.Module object.')

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
            'rec_loss': f'{adv_info.rec_loss.mean().item():.4f}',
        })
        if 'dist_val' in kwargs.keys():
            dist_val = kwargs['dist_val']
            self.pbar.set_infos({
                'dist_val': f'{dist_val.mean().item():.4f}',
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

        rec_logits = kwargs.get('rec_logits') # <B, num_class>
        adv_logits = kwargs.get('adv_logits') # <B, num_class>
        assert rec_logits is not None and adv_logits is not None

        adv_attack_success = self.attack_achieved(adv_logits, labels, target) # <B,>
        rec_attack_success = self.attack_achieved(rec_logits, labels, target=None) # <B,>
        attack_success = adv_attack_success & (rec_attack_success)

        update_mask = attack_success & (dist_val < _r.best_dist_val)
        _r.best_dist_val = torch.where(update_mask, dist_val, _r.best_dist_val)
        _r.final_success = torch.where(update_mask, True, _r.final_success)
        _r.best_attack[update_mask] = adv_pcs[update_mask].detach().cpu()


    def inner_loop(self, ori_pcs, labels, target=None, **kwargs):
        adv_pcs = ori_pcs.clone().detach() + torch.randn_like(ori_pcs) * 1e-7
        adv_pcs.requires_grad_(True)

        # optimizer
        optimizer = torch.optim.Adam([adv_pcs], lr=self.lr, weight_decay=0.)

        for _ in self.pbar:
            adv_logits = self.get_logits(adv_pcs)
            adv_info = self.adv_loss_fn(
                adv_loss = (adv_logits, labels, target),
            ) # <B,>
            optimizer.zero_grad()
            adv_info.loss.mean().backward()

            rec_pcs = self.ae_model(adv_pcs.permute(0, 2, 1))
            rec_pcs = rec_pcs.permute(0, 2, 1)
            rec_logits = self.get_logits(rec_pcs)

            adv_info.loss = adv_info.loss.detach()
            adv_info = self.adv_loss_fn(
                loss_info = adv_info,
                rec_loss = (rec_logits, labels, target),
            ) # <B,>
            adv_info.loss.mean().backward()
            optimizer.step()

            with torch.no_grad():
                delta = adv_pcs - ori_pcs
                delta = self.proj_delta(ori_pcs, delta, **kwargs)
                adv_pcs.data = ori_pcs + delta

            with torch.no_grad():
                rec_pcs    = self.ae_model(adv_pcs.permute(0, 2, 1))
                rec_pcs    = rec_pcs.permute(0, 2, 1)
                adv_logits = self.get_logits(adv_pcs)
                rec_logits = self.get_logits(rec_pcs)

            dist_val = l2_norm_distance(ori_pcs, adv_pcs)
            self.set_pbar_info_inner(
                adv_info, self.res_loss_fn(), dist_val=dist_val
            )

            self.update_record_items(
                adv_pcs, dist_val, labels, 
                target=target, adv_logits=adv_logits, rec_logits=rec_logits,
            )

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        self.init_record_items(ori_pcs)

        for bs in range(self.binary_search_step):
            self.set_pbar_info_outer(bs)
            self.inner_loop(ori_pcs, labels, target=target, **kwargs)

        delta = self.record_items.best_attack.to(self.device) - ori_pcs
        return delta.detach()
