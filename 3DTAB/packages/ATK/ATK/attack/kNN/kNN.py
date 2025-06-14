from functools import partial

import torch
import torch.optim as optim

from ATK.utils.common import check_option
from ATK.utils.advloss import (
    NegtiveAdvLoss, 
    CWAdvLoss,
    MLCAdvLoss,
    ChamferLoss,
    KNNSmoothingLoss,
)

from ..attack_template import CWAttack 
from .utils import ProjectInnerPoints


class KNN(CWAttack):
    _attack_name = 'kNN'

    def __init__(
        self,
        model,
        adv_loss_type = 'CW',
        kappa = 15,
        optim_type = 'adam',
        cha_loss_weight = 5.0,
        knn_loss_weight = 3.0,
        k_for_knn_loss = 5,
        alpha_for_knn_loss = 1.05,
        **kwargs,
    ):
        kwargs.setdefault('attack_lr', 0.001)
        kwargs.setdefault('inner_loop_max_iter', 2500)
        super().__init__(model, **kwargs)

        check_option(adv_loss_type, ['CE', 'CW', 'MLC'])
        check_option(
            optim_type, 
            ['adam', 'adadelta', 'adagrad', 'graddesc', 'momentum', 'rmsprop']
        )

        self.adv_loss_fn.add_objective(
            'adv_loss',
            self.parse_adv_loss_fn(adv_loss_type, kappa=kappa),
            weight=1.0,
        )

        self.res_loss_fn.add_objective(
            'cha_loss',
            ChamferLoss(single=True),
            cha_loss_weight,
        )
        self.res_loss_fn.add_objective(
            'knn_loss',
            KNNSmoothingLoss(k=k_for_knn_loss, alpha=alpha_for_knn_loss),
            knn_loss_weight,
        )

        self.optim_type = optim_type


    @staticmethod
    def parse_adv_loss_fn(adv_loss_type, **kwargs):
        adv_loss_map = {
            'CE':  NegtiveAdvLoss,
            'CW':  CWAdvLoss,
            'MLC': MLCAdvLoss,
        }
        Loss = adv_loss_map[adv_loss_type]
        if adv_loss_type == 'CW' or adv_loss_type == 'MLC':
            kappa = kwargs.get('kappa', 0.)
            return Loss(kappa)
        else:
            return Loss()

    def get_optimizer(self):
        optimizer_classes = {
            'adadelta': optim.Adadelta,
            'adagrad':  optim.Adagrad,
            'adam':     optim.Adam,
            'graddesc': optim.SGD,  
            'momentum': partial(optim.SGD, momentum=0.9),  
            'rmsprop':  optim.RMSprop,
        }
        return partial(optimizer_classes[self.optim_type], lr=self.lr)

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

        if res_info is not None:
            self.pbar.set_infos({
                'cha_loss': f'{res_info.cha_loss.mean().item():.4f}',
                'knn_loss': f'{res_info.knn_loss.mean().item():.4f}',
            })

    def init_record_items(self, ori_pcs):
        _r = self.record_items
        B = ori_pcs.shape[0]

        _r.best_attack   = torch.zeros_like(ori_pcs).cpu()
        _r.final_success = torch.full((B,), False).to(self.device)

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

        predict_logits = kwargs.get('loigts', None) # <B, num_class>
        if predict_logits is None:
            predict_logits = self.get_logits(adv_pcs)
        attack_success = self.attack_achieved(predict_logits, labels, target) # <B,>

        _r.final_success = torch.where(attack_success, True, _r.final_success)

        _r.best_attack = adv_pcs.detach().cpu()
        return attack_success 


    def inner_loop(self, ori_pcs, labels, target=None, **kwargs):
        N = ori_pcs.size(1)
        adv_pcs = ori_pcs.clone().detach() + torch.randn_like(ori_pcs) * 1e-7
        adv_pcs.requires_grad_(True)

        optimizer = self.get_optimizer()
        optimizer = optimizer(params=[adv_pcs])

        for _ in self.pbar:
            logits   = self.get_logits(adv_pcs)
            adv_info = self.adv_loss_fn(
                adv_loss = (logits, labels)
            ) # <B,>
            res_info = self.res_loss_fn(
                cha_loss = (adv_pcs, ori_pcs),
                knn_loss  = (adv_pcs,), 
            ) # <B,>
            # NOTE: in the official tensorflow code, they use sum instead of mean for dist_loss
            #       so we multiply num_points as sum
            res_info.loss *= N
            res_info.cha_loss *= N
            res_info.knn_loss *= N

            adv_loss, res_loss = adv_info.loss, res_info.loss
            loss = torch.mean(adv_loss + res_loss) # <1,>

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.set_pbar_info_inner(adv_info, res_info)

            with torch.no_grad():
                adv_pcs.data = self.proj_func(adv_pcs.data, ori_pcs, kwargs['normal'])
                delta = adv_pcs - ori_pcs
                delta = self.proj_delta(ori_pcs, delta, **kwargs)
                adv_pcs.data = ori_pcs + delta

            self.update_record_items(
                adv_pcs, None, labels, target=target, logits=logits,
            )


    def attack(self, ori_pcs, labels, target=None, **kwargs):
        ori_normals = kwargs.get('normal', None)
        if ori_normals is None:
            ori_normals = self.estimate_pcs_normals(ori_pcs)
            kwargs['normal'] = ori_normals

        self.proj_func = ProjectInnerPoints()
        self.set_pbar_info_outer(0)
        self.init_record_items(ori_pcs)
        self.inner_loop(ori_pcs, labels, target=target, **kwargs)

        delta = self.record_items.best_attack.to(self.device) - ori_pcs
        return delta.detach()
