import os
import os.path as osp

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from .basickit import *


_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_ckpt_dir  = osp.join(_asset_dir, 'model_ckpt') 

def build_trainer(cfg: DictConfig):
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)
    return trainer


def train(cfg: DictConfig, ckpt_path=None):
    init_experiment(cfg)
    model = build_model(cfg)
    datamodule = build_datamodule(cfg)
    trainer: Trainer = build_trainer(cfg)
    if ckpt_path is not None:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

def test(cfg: DictConfig):
    init_experiment(cfg)
    ckpt_path = osp.join(_ckpt_dir, cfg.dataset.type, cfg.model.type)
    model = build_model(cfg)
    datamodule = build_datamodule(cfg)
    trainer: Trainer = build_trainer(cfg)
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


batch_size_map = {
    'ModelNetHdf5': {
        'PointNet': 600,
        'PointNet++_SSG': 100,
        'PointConv': 150,
        'PointCNN': 800,
        'CurveNet': 200,
        'PointMLP': 40,
        'PointMLP_Elite': 40,
        'VN-DGCNN': 40,
    },
}
def set_batch_size(dataset_type, model_type):
    if dataset_type in batch_size_map:
        if model_type in batch_size_map[dataset_type]:
            return batch_size_map[dataset_type][model_type] # type: ignore
    return 50


epochs_map = {
    'ModelNetHdf5': {
        'PointMLP': 300,
        'PointMLP_Elite': 300,
        'PointCAT': 250,
        'VN-DGCNN': 250,
    },
}
def set_epochs(dataset_type, model_type):
    if dataset_type in batch_size_map:
        if model_type in epochs_map[dataset_type]:
            return epochs_map[dataset_type][model_type] # type: ignore
    return 200

def set_ckpt_dirpath(dataset_type, model_type):
    return osp.join(_ckpt_dir, dataset_type, model_type)

