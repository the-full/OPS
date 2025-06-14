from .ModelNet import ModelNetHdf5, ModelNetResampled
from .modelnet_datamodule import ModelNetHdf5DataModule, ModelNetResampledDataModule

datasets = {
    'ModelNetHdf5': ModelNetHdf5,
    'ModelNetResampled': ModelNetResampled,
}

datamodules = {
    'ModelNetHdf5': ModelNetHdf5DataModule,
    'ModelNetResampled': ModelNetResampledDataModule
}
