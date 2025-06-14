import re
import os 
import os.path as osp

import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig

from ATK import register_table

from .basickit import *


_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_ckpt_dir  = osp.join(_asset_dir, 'model_ckpt') 


class AdvDataSet(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.all_data = os.listdir(data_root)

    def __getitem__(self, idx):
        data_path = osp.join(self.data_root, self.all_data[idx])
        data_dict = torch.load(data_path, map_location='cpu')
        return data_dict

    def __len__(self):
        return len(self.all_data)


def get_model_cfg(model_name: str, cfg: DictConfig):
    if model_name in cfg.models.keys():
        return cfg.models[model_name]
    else:
        for _, model_cfg in cfg.models.items():
            if model_name == model_cfg.type:
                return model_cfg
        raise KeyError


def get_model_ckpt(dataset_type, model_type):
    ckpt_dir = osp.join(_ckpt_dir, dataset_type, model_type)
    if not os.path.exists(ckpt_dir):
        raise ValueError(f"The checkpoint directory {ckpt_dir} does not exist.")

    files = os.listdir(ckpt_dir)
    max_acc = -1
    max_acc_ckpt = None
    pattern = re.compile(r'val_acc=(\d+\.\d+)\.ckpt')

    for file in files:
        if (match := pattern.search(file)):
            val_acc_str = match.group(1)
            val_acc = float(val_acc_str)

            if val_acc > max_acc:
                max_acc = val_acc
                max_acc_ckpt = file

    if max_acc_ckpt is None:
        raise ValueError("No valid checkpoint file found with the expected format.")

    return torch.load(osp.join(ckpt_dir, max_acc_ckpt))


def load_model(model_name: str, cfg:DictConfig):
    model_cfg = get_model_cfg(model_name, cfg)
    if model_cfg['type'] == 'Point-NN':
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        name  = model_cfg.pop('type') # type: ignore
        return register_table.models[name](**model_cfg)
    else:
        ckpt  = get_model_ckpt(cfg.dataset.type, model_cfg.type)
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        name  = model_cfg.pop('type') # type: ignore
        model = register_table.models[name](**model_cfg)
        model.load_state_dict(ckpt['state_dict'])
        return model


def build_attacker(cfg: DictConfig, **kwargs):
    attack_cfg = OmegaConf.to_container(cfg.attacker, resolve=True)
    name = attack_cfg.pop('type') # type: ignore
    attacker = register_table.attacks[name](**attack_cfg, **kwargs)
    return attacker

def build_defenser(cfg: DictConfig, **kwargs):
    defense_cfg = OmegaConf.to_container(cfg.defenser, resolve=True)
    name = defense_cfg.pop('type') # type: ignore
    defenser = register_table.defenses[name](**defense_cfg, **kwargs)
    return defenser

def build_evaluator(cfg: DictConfig, **kwargs):
    def build_hook(hook_cfg: DictConfig):
        name = hook_cfg.pop('type') # type: ignore
        hook = register_table.hooks[name](**hook_cfg)
        return hook

    evaluator_cfg = OmegaConf.to_container(cfg.evaluator, resolve=True)
    hooks_cfg = evaluator_cfg.pop('hooks', []) # type: ignore
    name = evaluator_cfg.pop('type') # type: ignore
    evaluator = register_table.evaluators[name](**evaluator_cfg, **kwargs)

    for hook_cfg in hooks_cfg:
        evaluator.register_hook(build_hook(hook_cfg))
    return evaluator


def transfer_attack(cfg: DictConfig, evaluate_dataset):
    # init
    init_experiment(cfg)

    # build
    surrogate_model = load_model(cfg.surrogate_model, cfg).eval().to(cfg.evaluator.device)
    attacker = build_attacker(cfg, model=None)
    defenser = build_defenser(cfg)

    victim_models_dict = {}
    default_bs = cfg.victim_models_batch_size.default
    for model_name in cfg.victim_models:
        if model_name in cfg.victim_models_batch_size:
            bs = cfg.victim_models_batch_size[model_name]
        else:
            bs = default_bs

        victim_model = load_model(model_name, cfg).eval().to(cfg.evaluator.device)
        for p in victim_model.parameters():
            p.requires_grad = False
        victim_models_dict.update(
            {model_name : (victim_model, bs)}
        )
    transfer_evaluator = build_evaluator(
        cfg, 
        surrogate_model = surrogate_model,
        attacker = attacker,
        defenser = defenser,
        evaluate_dataset = evaluate_dataset,
        victim_models_dict = victim_models_dict,
    )

    # eval
    transfer_evaluator.start_test()
