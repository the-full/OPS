import hydra

from omegaconf import DictConfig, OmegaConf
from ATK.utils.scriptkit import trainkit as kit

OmegaConf.register_new_resolver('set_batch_size',   kit.set_batch_size,   replace=True)
OmegaConf.register_new_resolver('set_ckpt_dirpath', kit.set_ckpt_dirpath, replace=True)
OmegaConf.register_new_resolver('set_epochs', kit.set_epochs, replace=True)

@hydra.main(
    config_path='configs', 
    config_name='train_base',
    version_base='1.2',
)
def main(cfg: DictConfig):
    if cfg.mode == 'train':
        kit.train(cfg)
    elif cfg.model == 'test':
        kit.test(cfg)

if __name__ == '__main__':
    main()
