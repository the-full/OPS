from typing import Mapping, Sequence
import os
import os.path as osp

import torch
from torch.utils.data import DataLoader, Dataset

from ATK.utils.common import get_log

from .basic_hook import BasicHook


_logger = get_log()
_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_adv_data_dir = osp.join(_asset_dir, 'adv_data')


class AdvDataSet(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.all_data = os.listdir(data_root)
        self.all_data.sort()

    def __getitem__(self, idx):
        data_path = osp.join(self.data_root, self.all_data[idx])
        data_dict = torch.load(data_path, map_location='cpu')
        data_dict.pop('target')
        return data_dict

    def __len__(self):
        return len(self.all_data)

class SaveHook(BasicHook):
    def __init__(
        self, 
        exp_name = 'Testing',
        atk_name = '',
        def_name = '',
        save_atk = True,
        save_def = True,
        save_root = _adv_data_dir,
        use_saved_atk_result = True,
        use_saved_def_result = True,
    ):
        super().__init__()

        self.exp_name = exp_name
        self.atk_name = atk_name
        self.def_name = def_name
        self.save_atk = save_atk
        self.save_def = save_def

        self.save_root = save_root

        self.use_saved_atk_result = use_saved_atk_result
        self.use_saved_def_result = use_saved_def_result

        self.device  = None
        self.atk_cnt = 0
        self.def_cnt = 0


    def hijack_attacker(self, evaluator, saved_dict):
        self.attacker = evaluator.attacker
        evaluator.attacker = lambda data_dict, target: saved_dict 

    def hijack_defenser(self, evaluator, saved_dict):
        self.defenser = evaluator.defenser
        evaluator.defenser = lambda data_dict: saved_dict

    def give_attacker_back(self, evaluator):
        evaluator.attacker = self.attacker

    def give_defenser_back(self, evaluator):
        evaluator.defenser = self.defenser

    def parse_atk_def_save_path(self, evaluator):
        meta_info = evaluator.meta_info
        dataset_name  = meta_info['dataset_name']
        model_name    = meta_info['model_name']

        if self.atk_name == '':
            self.atk_name = meta_info['attacker_name']

        if self.def_name == '':
            self.def_name = meta_info['defenser_name']

        atk_save_name = f'{model_name}@{self.atk_name}'
        def_save_name = f'{model_name}@{self.atk_name}@{self.def_name}'
        data_dir = osp.join(self.save_root, dataset_name, self.exp_name)

        self.atk_save_path = osp.join(data_dir, atk_save_name)
        self.def_save_path = osp.join(data_dir, def_save_name)

    def on_test_begin(self, evaluator):
        self.device = evaluator.device
        self.parse_atk_def_save_path(evaluator)

        exp_data_len = len(evaluator.dataset)
        atk_data_dir = self.atk_save_path
        self.atk_data_dir = atk_data_dir

        if not osp.exists(atk_data_dir):
            _logger.info(f"The attack data directory does not exist, Creating...")
            os.makedirs(atk_data_dir)

        atk_data_len = len(os.listdir(atk_data_dir))

        if self.use_saved_atk_result and atk_data_len != exp_data_len:
            _logger.warning(
                f"Attention: The attack data directory '{atk_data_dir}' contains {atk_data_len} files, "
                f"which does not match the length of the dataset ({exp_data_len}), ",
            )
            if self.save_atk:
                _logger.info(f"Found `save_atk` is True，evaluator will save attack data at this execution. ")
                self.use_saved_atk_result = False
            else:
                raise RuntimeError("Unable to ensure the evaluator execution safety, terminating execution.")

        elif self.use_saved_atk_result and atk_data_len == exp_data_len:
            _logger.info(f"Found `use_saved_atk_result` is True，evaluator will use saved attack data at this execution. ")
            self.save_atk = False

        if self.save_atk and atk_data_len != 0:
            _logger.warning(
                f"Attention: The attack data directory '{atk_data_dir}' already contains {atk_data_len} files, "
                f"which will be overwritten during the test. ",
            )


        def_data_dir = self.def_save_path
        self.def_data_dir = def_data_dir

        if not osp.exists(def_data_dir):
            _logger.info(f"The defense data directory does not exist, Creating...")
            os.makedirs(def_data_dir)

        def_data_len = len(os.listdir(def_data_dir))

        if self.use_saved_def_result and def_data_len != exp_data_len:
            _logger.warning(
                f"Attention: The defense data directory '{def_data_dir}' contains {def_data_len} files, "
                f"which does not match the length of the dataset ({exp_data_len}), ",
            )
            if self.save_def:
                _logger.info(f"Found `save_def` is True，evaluator will save defense data at this execution. ",)
                self.use_saved_def_result = False
            else:
                raise RuntimeError("Unable to ensure the evaluator execution safety, terminating execution.")
        elif self.use_saved_def_result and def_data_len == exp_data_len:
            _logger.info(f"Found `use_saved_def_result` is True，evaluator will use saved defense data at this execution. ")
            self.save_def = False

        if self.save_def and def_data_len != 0:
            _logger.warning(
                f"Attention: The defense data directory '{def_data_dir}' already contains {def_data_len} files, "
                f"which will be overwritten during the test. ",
            )

        batch_size = evaluator.batch_size
        if self.use_saved_atk_result:
            self.atk_dataloader = iter(DataLoader(
                AdvDataSet(data_root=self.atk_data_dir),
                batch_size=batch_size,
                shuffle=False, 
                drop_last=False
            ))

        if self.use_saved_def_result:
            self.def_dataloader = iter(DataLoader(
                AdvDataSet(data_root=self.def_data_dir),
                batch_size=batch_size,
                shuffle=False, 
                drop_last=False
            ))


    def cast_data(self, data, ):
        """Copying data to the target device.

        Args:
            data (dict): Data returned by ``DataLoader``.

        Returns:
            CollatedResult: Inputs and data sample at target device.
        """
        if isinstance(data, Mapping):
            return {key: self.cast_data(data[key]) for key in data}
        elif isinstance(data, (str, bytes)) or data is None:
            return data
        elif isinstance(data, tuple) and hasattr(data, '_fields'):
            return type(data)(*(self.cast_data(sample) for sample in data))  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, Sequence):
            return type(data)(self.cast_data(sample) for sample in data)  # type: ignore  # noqa: E501  # yapf:disable
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return data


    def before_attack(self, evaluator, data_dict):
        if self.use_saved_atk_result:
            saved_dict = next(self.atk_dataloader)
            saved_dict = self.cast_data(saved_dict)
            self.hijack_attacker(evaluator, saved_dict)

    def before_defense(self, evaluator, data_dict):
        if self.use_saved_def_result:
            saved_dict = next(self.def_dataloader)
            saved_dict = self.cast_data(saved_dict)
            self.hijack_defenser(evaluator, saved_dict)

    def after_attack(self, evaluator, data_dict):
        if self.use_saved_atk_result:
            self.give_attacker_back(evaluator)

        if self.save_atk:
            atk_data_dir = self.atk_save_path
            batch_size = data_dict['xyz'].shape[0]
            for i in range(batch_size):
                save_name = "{:06d}.pth".format(self.atk_cnt)
                save_path = osp.join(atk_data_dir, save_name)
                save_data = dict(
                    xyz = data_dict['xyz'][i],
                    feat = data_dict['feat'][i],
                    offset = data_dict['offset'][i],
                    delta = data_dict['delta'][i],
                    category = data_dict['category'][i],
                    target = evaluator.target,
                )
                if 'normal' in data_dict.keys():
                    save_data['normal'] = data_dict['normal'][i]
                torch.save(save_data, save_path)
                self.atk_cnt += 1

    def after_defense(self, evaluator, data_dict):
        if self.use_saved_def_result:
            self.give_defenser_back(evaluator)

        if self.save_def:
            def_data_dir = self.def_save_path
            batch_size = data_dict['xyz'].shape[0]
            for i in range(batch_size):
                save_name = "{:06d}.pth".format(self.def_cnt)
                save_path = osp.join(def_data_dir, save_name)
                save_data = dict(
                    xyz = data_dict['xyz'][i],
                    feat = data_dict['feat'][i],
                    offset = data_dict['offset'][i],
                    category = data_dict['category'][i],
                    target = evaluator.target,
                )
                torch.save(save_data, save_path)
                self.def_cnt += 1
