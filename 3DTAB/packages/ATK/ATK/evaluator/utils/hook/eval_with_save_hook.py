
import torch

from .basic_hook import BasicHook


_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_adv_data_dir = osp.join(_asset_dir, 'adv_data')


class AdvDataSet(object):
    def __init__(self, data_root):
        self.data_root = data_root
        self.all_data = os.listdir(data_root)
        self.all_data.sort()

    def __getitem__(self, idx):
        data_path = osp.join(self.data_root, self.all_data[idx])
        data_dict = torch.load(data_path, map_location='cpu')
        return data_dict

    def __len__(self):
        return len(self.all_data)

class EvalWithSaveHook(BasicHook):
    def __init__(
        self, 
        exp_name = '',
        save_dir = _adv_data_dir,
        save_atk = False,
        save_def = False,
        use_saved_atk_result = False,
        use_saved_def_result = False,
    ):
        super().__init__()
        self.exp_name = exp_name
        self.save_dir = save_dir

        self.use_saved_atk_result = use_saved_atk_result
        self.use_saved_def_result = use_saved_def_result

        self.save_atk = False if use_saved_atk_result else save_atk
        self.save_def = False if use_saved_def_result else save_def

        if self.use_saved_def_result:
            self.use_saved_atk_result = False
            if self.save_atk:
                warnings.warn(
                    "Set `save_atk` with `use_saved_def_result` are conflicting, "
                    "It will be treat as `save_atk=False`. "
                    "`save_atk` depend on the original dataset, "
                    "thus the `evaluator.dataset` must be preserved when saving. "
                    "However, `use_saved_atk_result` requires threfting `evaluator.dataset`.",
                    UserWarning
                )
                self.save_atk = False

        self.atk_cnt = 0
        self.def_cnt = 0

    def theft_attacker(self, evaluator):
        self.attacker = evaluator.attacker
        evaluator.attacker = lambda data_dict, target: data_dict

    def theft_defenser(self, evaluator):
        self.defenser = evaluator.defenser
        evaluator.defenser = lambda data_dict: data_dict

    def give_attacker_back(self, evaluator):
        evaluator.attacker = self.attacker

    def give_defenser_back(self, evaluator):
        evaluator.defenser = self.defenser

    def get_evaluator_data_dir(self, evaluator):
        meta_info = evaluator.meta_info
        dataset_name  = meta_info['dataset_name']
        attacker_name = meta_info['attacker_name']
        model_name    = meta_info['model_name']
        defenser_name = meta_info['defenser_name']
        dir_name = f'{attacker_name}@{model_name}@{defenser_name}'
        if self.exp_name != '':
            dir_name += f'@{self.exp_name}'
        data_dir = osp.join(self.save_dir, dataset_name, dir_name)
        return data_dir

    def get_atk_data_dir(self, evaluator):
        data_dir = self.get_evaluator_data_dir(evaluator)
        return osp.join(data_dir, 'atk_data')

    def get_def_data_dir(self, evaluator):
        data_dir = self.get_evaluator_data_dir(evaluator)
        return osp.join(data_dir, 'def_data')

    def on_dataloader_loop_begin(self, evaluator):
        if self.use_saved_atk_result:
            atk_data_dir = self.get_atk_data_dir(evaluator)
            if osp.exists(atk_data_dir):
                self.theft_attacker(evaluator)
                evaluator.dataset = AdvDataSet(data_root=atk_data_dir)
            else:
                warnings.warn(
                    "The attack data directory does not exist. "
                    "Please ensure that you have saved the attack results before attempting to use them. "
                    "Now, set 'save_atk' to True.",
                    UserWarning
                )
                self.save_atk = True
        elif self.use_saved_def_result:
            def_data_dir = self.get_def_data_dir(evaluator)
            if osp.exists(def_data_dir):
                self.theft_attacker(evaluator)
                self.theft_defenser(evaluator)
                evaluator.dataset = AdvDataSet(data_root=def_data_dir)
            else:
                warnings.warn(
                    "The defense data directory does not exist. "
                    "Please ensure that you have saved the defense results before attempting to use them. "
                    "Now, set 'save_def' to True.",
                    UserWarning
                )
        else:
            pass

    def on_dataloader_loop_end(self, evaluator):
        if self.use_saved_atk_result:
            self.give_attacker_back(evaluator)
        elif self.use_saved_def_result:
            self.give_attacker_back(evaluator)
            self.give_defenser_back(evaluator)
        else:
            pass

    def after_attack(self, evaluator, data_dict):
        if self.save_atk:
            atk_data_dir = self.get_atk_data_dir(evaluator)
            batch_size = data_dict['xyz'].shape[0]
            for i in range(batch_size):
                save_name = "{:06d}".format(self.atk_cnt)
                save_path = osp.join(atk_data_dir, save_name)
                save_data = dict(
                    xyz = data_dict['xyz'][i],
                    delta = data_dict['delta'][i],
                    category = data_dict['category'][i],
                    target = evaluator.target,
                )
                torch.save(save_data, save_path)
        else:
            pass

    def after_defense(self, evaluator, data_dict):
        if self.save_def:
            def_data_dir = self.get_def_data_dir(evaluator)
            batch_size = data_dict['xyz'].shape[0]
            for i in range(batch_size):
                save_name = "{:06d}".format(self.def_cnt)
                save_path = osp.join(def_data_dir, save_name)
                save_data = dict(
                    xyz = data_dict['xyz'][i],
                    category = data_dict['category'][i],
                    target = evaluator.target,
                )
                torch.save(save_data, save_path)
        else:
            pass
