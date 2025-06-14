import types

import torch
from pytorch3d.ops import estimate_pointcloud_normals

from ATK.utils.advloss import CombinedLoss
from ATK.utils.common import check_option

from .basic_attack import BasicAttack


class CWAttack(BasicAttack):
    _attack_name = 'CW-Attack'

    def __init__(
        self, 
        model, 
        binary_search_step = 10,
        binary_update_cond = 'record update',
        inner_loop_max_iter = 500,
        initial_weight = 1.,
        lower_bound_weight = 0.,
        upper_bound_weight = 1e10,
        attack_lr = 0.001,
        **kwargs,
    ):
        super().__init__(model, **kwargs)

        # attack setting
        check_option(
            binary_update_cond, ['attack success', 'record update', 'success at last']
        )
        self.binary_search_step  = binary_search_step
        self.binary_update_cond  = binary_update_cond
        self.inner_loop_max_iter = inner_loop_max_iter
        self.initial_weight      = initial_weight
        self.upper_bound_weight  = upper_bound_weight
        self.lower_bound_weight  = lower_bound_weight
        self.lr                  = attack_lr

        self.adv_loss_fn = CombinedLoss()    # adversarial loss
        self.res_loss_fn = CombinedLoss()    # restaint/residual loss

        self.record_items = types.SimpleNamespace()
        self.coef_bound = types.SimpleNamespace()
        self.global_val = types.SimpleNamespace()

    def init_record_items(self, ori_pcs):
        _r = self.record_items
        B = ori_pcs.shape[0]

        _r.best_attack   = ori_pcs.clone().cpu()
        _r.best_dist_val = torch.full((B,), 1e10).to(self.device)
        _r.final_success = torch.full((B,), False).to(self.device)


    def inner_init_record_items(self, ori_pcs):
        _r = self.record_items
        B = ori_pcs.shape[0]

        _r.iter_best_dist_val = torch.full((B,), 1e10).to(self.device)
        _r.iter_final_success = torch.full((B,), False).to(self.device)

        if self.binary_update_cond == 'success at last':
            _r.iter_attack_success = torch.full((B,), False).to(self.device)

    def init_coef_bound(self, batch_size):
        _c = self.coef_bound
        device = self.device

        _c.c_for_each_pc = torch.full((batch_size,), self.initial_weight).float().to(device)
        _c.c_lower_bound = torch.full((batch_size,), self.lower_bound_weight).float().to(device)
        _c.c_upper_bound = torch.full((batch_size,), self.upper_bound_weight).float().to(device)

    def init_attack(self, ori_pcs, **kwargs):
        self.init_coef_bound(ori_pcs.shape[0])
        self.init_record_items(ori_pcs)

    def inner_init_attack(self, binary_step, ori_pcs, **kwargs):
        self.set_pbar_info_outer(binary_step)
        self.inner_init_record_items(ori_pcs)

    def clean_attack(self):
        self.record_items = types.SimpleNamespace()
        self.coef_bound = types.SimpleNamespace()
        self.global_val = types.SimpleNamespace()
        self.pbar.close()

    def outer_attack_achieve(self):
        _r = self.record_items
        if self.binary_update_cond == 'attack success':
            return _r.iter_final_success
        elif self.binary_update_cond == 'record update':
            return _r.iter_final_success & (_r.iter_best_dist_val <= _r.best_dist_val)
        elif self.binary_update_cond == 'success at last':
            return _r.iter_attack_success
        else:
            raise NotImplementedError

    def binary_search_coef(self):
        _c = self.coef_bound
        atk_success_mask = self.outer_attack_achieve()
        _c.c_lower_bound = torch.where( atk_success_mask, _c.c_for_each_pc, _c.c_lower_bound)
        _c.c_upper_bound = torch.where(~atk_success_mask, _c.c_for_each_pc, _c.c_upper_bound)
        _c.c_for_each_pc = (_c.c_lower_bound + _c.c_upper_bound) * 0.5

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

        attack_success = kwargs.get('attack_success', None)
        if attack_success is None:
            predict_logits = kwargs.get('loigts', None) # <B, num_class>
            if predict_logits is None:
                predict_logits = self.get_logits(adv_pcs)
            attack_success = self.attack_achieved(predict_logits, labels, target) # <B,>

        if self.binary_update_cond == 'success at last':
            _r.iter_attack_success = attack_success

        iter_update_mask = attack_success & (dist_val < _r.iter_best_dist_val)
        _r.iter_best_dist_val = torch.where(iter_update_mask, dist_val, _r.iter_best_dist_val)
        _r.iter_final_success = torch.where(iter_update_mask, True, _r.iter_final_success)

        update_mask = attack_success & (dist_val < _r.best_dist_val)
        _r.best_dist_val = torch.where(update_mask, dist_val, _r.best_dist_val)
        _r.final_success = torch.where(update_mask, True, _r.final_success)
        _r.best_attack[update_mask] = adv_pcs[update_mask].detach().cpu()


    def get_cw_pbar(self, attack_name=None):
        pbar = self.get_pbar(
            range(self.inner_loop_max_iter),
            attack_name = attack_name,
        )
        return pbar

    def inner_loop(
        self, 
        ori_pcs, 
        labels, 
        target=None, 
        **kwargs
    ):
        c_for_each_pc = self.coef_bound.c_for_each_pc

        delta = self.init_delta(ori_pcs).requires_grad_(True)
        torch.nn.init.trunc_normal_(delta, std=0.01)
        optimizer = torch.optim.Adam([delta], lr=self.lr)

        for _ in self.pbar:
            adv_pcs  = ori_pcs + delta # <B, N, C>
            logits   = self.get_logits(adv_pcs)
            adv_info = self.adv_loss_fn(logits, labels, target) # <B,>
            res_info = self.res_loss_fn(adv_pcs, ori_pcs) # <B,>

            adv_loss, res_loss = adv_info.loss, res_info.loss
            loss = torch.mean(adv_loss + c_for_each_pc * res_loss) # <1,>

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.set_pbar_info_inner(adv_info, res_info)
            self.update_record_items(
                adv_pcs, res_loss.loss, labels, target=target, logits=logits
            )

    def set_pbar_info_outer(self, binary_step, **kwargs):
        _c = self.coef_bound
        self.pbar = self.get_cw_pbar(self._attack_name)
        self.pbar.set_infos({
            'b_step':     f'{binary_step}',
            'coef_lower': f'{_c.c_lower_bound.mean().item():.2f}',
            'coef':       f'{_c.c_for_each_pc.mean().item():.2f}',
            'coef_upper': f'{_c.c_upper_bound.mean().item():.2f}',
        })
        _r = self.record_items
        if hasattr(_r, 'final_success'):
            num_suc = _r.final_success.sum().item()
            num_all = len(_r.final_success)
            asr = num_suc / num_all
            self.pbar.set_infos({
                'ASR:':   f'{num_suc}/{num_all} = {asr*100:.2f}%',
            })

    def set_pbar_info_inner(self, adv_info, res_info, **kwargs):
        self.pbar.set_infos({
            'Adv_loss': f'{adv_info.loss.mean().item():.4f}',
            'Res_loss': f'{res_info.loss.mean().item():.4f}',
        })

    def attack(self, ori_pcs, labels, target=None, **kwargs):
        self.init_attack(ori_pcs)

        for binary_step in range(self.binary_search_step):
            self.inner_init_attack(binary_step, ori_pcs)
            self.inner_loop(ori_pcs, labels, target=target, **kwargs)
            self.binary_search_coef()

        delta = self.record_items.best_attack.to(self.device) - ori_pcs
        return delta.detach()

    def __call__(self, data_dict, target=None):
        data_dict = super().__call__(data_dict, target)
        self.clean_attack()
        return data_dict

    @staticmethod
    def attack_achieved(logits, labels, target=None):
        if target is None:
            return torch.argmax(logits, dim=-1) != labels
        else:
            return torch.argmax(logits, dim=-1) == target

    @staticmethod
    def estimate_pcs_normals(pcs, k=50):
        return -estimate_pointcloud_normals(pcs, k)
