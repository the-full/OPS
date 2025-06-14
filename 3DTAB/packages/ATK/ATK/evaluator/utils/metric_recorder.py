import warnings
from typing import Sequence

import torch


class MetricRecorder:
    def __init__(self, metric_name, reduce_method):
        self.init_value_map = {
            'mean': [],
            'sum': 0.0,
            'max': -1e10,
            'min': 1e10,
        }
        self.reduced = False
        self.metric_name = metric_name
        self.reduce_method = reduce_method
        self.metric_buffer = self.init_value_map[reduce_method]
        self.metric_result = 0.

    @property
    def result(self):
        if not self.reduced:
            warnings.warn('metric not reduced yet, will be auto reduce now, that may be not expected.')
            self.reduce()
        return self.metric_result

    def update(self, metric_value):
        if isinstance(metric_value, torch.Tensor):
            metric_value = list(metric_value.flatten().cpu().detach().numpy())
        elif isinstance(metric_value, Sequence):
            metric_value = list(metric_value)
        else:
            try:
                metric_value = list(metric_value)
            except:
                raise RuntimeError(f'the value of {self.metric_name} can not be convert to list.')

        if metric_value == []:
            pass
        else:
            self.metric_reduced = False
            
        if self.reduce_method == 'mean':
            assert isinstance(self.metric_buffer, list)
            self.metric_buffer.extend(metric_value)
        elif self.reduce_method == 'sum':
            self.metric_buffer += sum(metric_value) if isinstance(metric_value, Sequence) else metric_value
        elif self.reduce_method == 'max':
            self.metric_buffer = max(max(metric_value), self.metric_buffer)
        elif self.reduce_method == 'min':
            self.metric_buffer = min(min(metric_value), self.metric_buffer)
        else:
            raise NotImplemented

    def reduce(self):
        if not self.reduced and self.reduce_method == 'mean':
            assert isinstance(self.metric_buffer, list)
            if self.metric_buffer == []:
                warnings.warn(f'{self.metric_name} record is empty and will be recorded as zero.')
                self.metric_result = 0.
            else:
                self.metric_result = sum(self.metric_buffer) / len(self.metric_buffer)
        else:
            raise NotImplemented
        self.reduced = True

    def reset(self):
        self.metric_result = 0.
        self.metric_buffer = self.init_value_map[self.reduce_method]
        self.reduced = False
