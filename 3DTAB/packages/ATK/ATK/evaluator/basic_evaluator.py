from typing import Optional

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from ATK.utils.common import get_log
from .utils import metrics_printer, MetricRecorder


class BasicEvaluator(object):
    def __init__(
        self,
        meta_info,
        model,
        attacker,
        defenser,
        dataset,
        budget = 0.06,
        budget_type = 'linfty',
        update_alpha = True,
        batch_size = 50,
        device = 'cuda',
        logger = None,
        renorm_result = True,
        renorm_type = 'to_one',
    ):
        assert all(
            key in meta_info 
            for key in ['model_name', 'attacker_name', 'defenser_name', 'dataset_name']
        )
        self.meta_info = meta_info
        self.model     = model
        self.attacker  = attacker
        self.defenser  = defenser
        self.dataset   = dataset
        self.logger    = logger if logger is not None else get_log()

        self.device = device
        self.target = None
        self.budget = budget
        self.budget_type  = budget_type
        self.update_alpha = update_alpha
        self.renorm_result = renorm_result
        self.renorm_type   = renorm_type

        self.hooks    = []
        self.metrics  = {}

        self.batch_size = batch_size 
        self.metrics_printer = metrics_printer

    def setup(self, attack_type='untarget'):
        if isinstance(attack_type, int):
            self.target = attack_type
        elif isinstance(attack_type, str):
            if attack_type == 'untarget':
                self.target = None
            elif attack_type.isdigit():
                self.target = int(attack_type)
            else:
                raise ValueError('can not parse attack_type')
        else:
            raise ValueError('can not attack_type')

    def call_hook(self, fn_name, *args, **kwargs):
        for hook in self.hooks:
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def register_hook(self, hook):
        self.hooks.append(hook)

    def register_metric(self, metric_name, reduce_method='mean'):
        if metric_name in self.metrics:
            pass
        else:
            self.metrics.update(
                {metric_name: MetricRecorder(metric_name, reduce_method)}
            )
        
    def update_metrics(self, metric_list):
        for metric_dict in metric_list:
            metric_name   = metric_dict['name']
            metric_value  = metric_dict['value']
            reduce_method = metric_dict.get('reduce', 'mean')

            if metric_name not in self.metrics:
                self.register_metric(metric_name, reduce_method)

            self.metrics[metric_name].update(metric_value)

    def reduce_metrics(self):
        for _, metric in self.metrics.items():
            metric.reduce()
            self.metrics_printer(self.logger, metric)

    @staticmethod
    def metrics_helper(name, value, reduce_methods):
        metric_list = [
            {
                'name': f'{name}_{reduce_method}', 
                'value': value, 
                'reduce': reduce_method
            } 
            for reduce_method in reduce_methods
        ]
        return metric_list

    def start_test(self):
        self.call_hook('on_test_begin')

        self.attacker.set_task(self.model, self.device)
        self.attacker.set_budget(self.budget, self.budget_type)
        self.attacker.set_renorm(self.renorm_result, self.renorm_type)

        self.call_hook('on_dataloader_loop_begin')

        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            drop_last=False
        )
        for data_dict in tqdm(dataloader):

            self.call_hook('before_attack', data_dict)
            data_dict = self.attacker(data_dict, self.target)
            self.call_hook('after_attack', data_dict)

            self.call_hook('before_defense', data_dict)
            data_dict = self.defenser(data_dict)
            self.call_hook('after_defense', data_dict)

            self.record_atk_result(data_dict)

        self.call_hook('on_dataloader_loop_end')

        self.reduce_metrics()

        self.call_hook('on_test_end')

    def record_atk_result(self, data_dict):
        labels = data_dict['category'].view(-1)
        with torch.no_grad():
            logits  = self.model.__predict__(data_dict)
        atk_success = self.get_atk_success(logits, labels, self.target)
        metric_list = self.metrics_helper('ASR', atk_success, ['mean'])
        self.update_metrics(metric_list)

    @staticmethod
    def get_atk_success(
        adv_logits: torch.Tensor,
        labels: torch.Tensor,
        target: Optional[int] = None
    ) -> torch.Tensor:
        preds = torch.argmax(adv_logits, dim=-1)
        if target is None:
            return preds != labels
        else:
            return preds == target
