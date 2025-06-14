import torch
import pytorch_lightning as pl
import os
import torch

from omegaconf import OmegaConf, DictConfig

from ATK import register_table


def set_seed(seed):
    pl.seed_everything(seed)

def set_device(device_config):
    if device_config == 'cpu':
        pass
    else:
        from torch.backends import cudnn
        # set the global cuda device
        cudnn.enabled = True
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
        torch.cuda.set_device(device_config.cuda)


def set_processtitle(cfg):
    # set process title
    import setproctitle
    setproctitle.setproctitle(cfg.process_title)

def init_experiment(cfg, **kwargs):
    cfg = cfg

    verbose = cfg.get('verbose', True)
    if verbose:
        print("config:")
        for k, v in cfg.items():
            print(k, v)
        print("=" * 20)

        print("kwargs:")
        for k, v in kwargs.items():
            print(k, v)
        print("=" * 20)

    # set seed
    set_seed(cfg.seed)
    # set device
    set_device(cfg.device)

    # set process title
    set_processtitle(cfg)


def build_model(cfg: DictConfig, **kwargs):
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    name = model_cfg.pop('type') # type: ignore
    model = register_table.models[name](**model_cfg, **kwargs)
    return model

def build_dataset(cfg: DictConfig, **kwargs):
    dataset_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    name = dataset_cfg.pop('type') # type: ignore
    dataset = register_table.datasets[name](**dataset_cfg, **kwargs)
    return dataset

def build_datamodule(cfg: DictConfig, **kwargs):
    datamodule_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    name = datamodule_cfg.pop('type') # type: ignore
    datamodule = register_table.datamodules[name](**datamodule_cfg, **kwargs)
    return datamodule

batch_size_map = {
    'PointNet':       600,
    'PointNet++_MSG': 50,
    'PointNet++_SSG': 100,
    'PointConv':      150,
    'DGCNN':          100 ,
}
