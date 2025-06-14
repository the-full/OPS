"""SRS defense proposed by https://arxiv.org/pdf/1902.10899.pdf"""
import numpy as np

import torch

from ..basic_defense import BasicDefense

from ATK.utils.common import return_first_item


class SRSDefense(BasicDefense):
    """Random dropping points as defense."""

    def __init__(self, drop_num=128):
        """SRS defense method.

        Args:
            drop_num (int, optional): number of points to drop.
                                        Defaults to 500.
        """
        super(SRSDefense, self).__init__()

        self.drop_num = drop_num

    def _random_drop(self, pc):
        """Random drop self.drop_num points in each pc.

        Args:
            pc (torch.FloatTensor): batch input pc, [B, K, 3]
        """
        B, K = pc.shape[:2]
        idx = [np.random.choice(K, K - self.drop_num, replace=False) for _ in range(B)]
        pc = torch.stack([pc[i][torch.from_numpy(
            idx[i]).long().to(pc.device)] for i in range(B)])
        return pc, idx

    def random_drop(self, x):
        return return_first_item(self._random_drop(x))

    def defense(self, adv_pcs, labels, **kwargs):
        x = adv_pcs
        with torch.no_grad():
            x, idx = self._random_drop(x)
            self.sel_mask = idx
        return self.padding(x)
