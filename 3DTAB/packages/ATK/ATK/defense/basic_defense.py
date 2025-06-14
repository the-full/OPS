from collections.abc import Sequence
from copy import copy

import torch

class BasicDefense(object):
    def __init__(self, model=None, device='cuda'):
        self.device = device
        self.sel_mask = None

    def defense(self, adv_pcs, labels, **kwargs):
        return adv_pcs

    def __call__(self, data_dict):
        data_dict = copy(data_dict)
        adv_pcs = data_dict.pop('xyz')
        labels = data_dict.pop('category').view(-1)
        def_pcs = self.defense(adv_pcs, labels, **data_dict)
        data_dict['xyz'] = def_pcs

        if 'feat' in data_dict.keys():
            feat = data_dict.get('feat')
            if self.sel_mask is not None:
                x = data_dict['feat']
                B = x.shape[0]
                feat = self.padding([x[i][self.sel_mask[i]] for i in range(B)])
            try:
                feat[:, :, :3] = def_pcs
            except:
                feat = def_pcs
        else:
            feat = def_pcs

        offset = torch.tensor([len(pc) for pc in def_pcs])
        offset = offset.long().to(def_pcs).unsqueeze(-1)

        data_dict['xyz']      = def_pcs
        data_dict['feat']     = feat
        data_dict['offset']   = offset
        data_dict['category'] = labels

        return data_dict

    def padding(self, pc_list):
        max_len = max(pc.size(0) for pc in pc_list)
        return self._padding(pc_list, max_len)


    @staticmethod
    def _padding(pc_list, num_points):
        padded_pcs = []
        if isinstance(num_points, Sequence):
            assert len(pc_list) == len(num_points)
            for pc, num in zip(pc_list, num_points):
                num_padding = num - pc.size(0)
                padding = torch.zeros(num_padding, 3, dtype=pc.dtype, device=pc.device)
                padded_pc = torch.cat((pc, padding), dim=0)
                padded_pcs.append(padded_pc)
        else:
            for pc in pc_list:
                num_padding = num_points - pc.size(0)
                padding = torch.zeros(num_padding, 3, dtype=pc.dtype, device=pc.device)
                padded_pc = torch.cat((pc, padding), dim=0)
                padded_pcs.append(padded_pc)
        
        pcs = torch.stack(padded_pcs, dim=0)
        
        return pcs
