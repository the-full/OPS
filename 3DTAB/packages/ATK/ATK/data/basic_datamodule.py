import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class BasicDataModule(pl.LightningDataModule):

    def __init__(self, batch_size=32, num_workers=8):
        super().__init__()
        self.batch_size  = batch_size
        self.num_workers = num_workers


    def prepare_data(self):
        pass

    @property
    def train_dataset(self):
        raise NotImplementedError

    @property
    def val_dataset(self):
        raise NotImplementedError

    @property
    def test_dataset(self):
        raise NotImplementedError

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=True
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            shuffle=False)
