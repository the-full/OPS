from ATK.utils import transform as T

from .basic_datamodule import BasicDataModule
from .ModelNet import ModelNetHdf5, ModelNetResampled


train_transform = [
    T.NormalizeCoord(),
    T.RandomScale(scale=[0.7, 1.5], anisotropic=True),
    T.RandomShift(shift=[(-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)]),
    T.ShufflePoint(),
    T.ToTensor(),
    T.Collect(keys=('xyz', 'category'), feat_keys=['xyz']),
]

val_transform = [
    T.NormalizeCoord(),
    T.ToTensor(),
    T.Collect(keys=('xyz', 'category'), feat_keys=['xyz']),
]

class ModelNetHdf5DataModule(BasicDataModule):
    def __init__(
        self, 
        batch_size=32, 
        num_workers=8,
        num_points=1024, 
        uniform_sampling=True,
    ):
        super().__init__(batch_size, num_workers)
        self.num_points       = num_points
        self.uniform_sampling = uniform_sampling

    @property
    def train_dataset(self):
        return ModelNetHdf5(
            split='train', 
            transform=train_transform,
            num_points=self.num_points, 
            uniform_sampling=self.uniform_sampling, 
        )

    @property
    def val_dataset(self):
        return ModelNetHdf5(
            split='test', 
            transform=val_transform,
            num_points=self.num_points, 
            uniform_sampling=self.uniform_sampling, 
        )

    @property
    def test_dataset(self):
        return self.val_dataset


class ModelNetResampledDataModule(BasicDataModule):
    def __init__(
        self, 
        batch_size=32, 
        num_workers=8,
        num_points=1024, 
        uniform_sampling=True,
    ):
        super().__init__(batch_size, num_workers)
        self.num_points       = num_points
        self.uniform_sampling = uniform_sampling

    @property
    def train_dataset(self):
        return ModelNetResampled(
            split='train', 
            transform=train_transform,
            num_points=self.num_points, 
            uniform_sampling=self.uniform_sampling, 
        )

    @property
    def val_dataset(self):
        return ModelNetResampled(
            split='test', 
            transform=val_transform,
            num_points=self.num_points, 
            uniform_sampling=self.uniform_sampling, 
        )

    @property
    def test_dataset(self):
        return self.val_dataset


