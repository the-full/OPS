import os
import os.path as osp

import hydra
from omegaconf import DictConfig
from ATK.utils import scriptkit as kit


_asset_dir = os.getenv('ATK_ASSET_PATH', '')
_dataset_dir = osp.join(_asset_dir, 'dataset', 'modelnet40_ply_hdf5_2048_mini')

@hydra.main(
    config_path='configs', 
    config_name='attack_base.yaml', 
    version_base='1.2',
)
def main(cfg: DictConfig):
    data_dir = osp.join(_dataset_dir, cfg.split)
    evaluate_dataset = kit.AdvDataSet(data_dir)
    kit.transfer_attack(cfg, evaluate_dataset)

if __name__ == '__main__':
    main()

