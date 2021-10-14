import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

import utils
from dataset import DivingWithMasksDataset


class DivingSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, config: dict):
        super().__init__()
        self.data_dir = data_dir
        self.config = config

    def setup(self, stage: str = None) -> None:
        # if stage == 'fit' or stage is None:
        if stage == 'fit':
            train_val_data = DivingWithMasksDataset(
                root=self.data_dir,
                train=True)
            self.train_data, self.val_data = random_split(
                train_val_data, [self.config['nb_train_samples'], self.config['nb_val_samples']],
                generator=torch.Generator().manual_seed(42))

        # if stage == 'test' or stage is None:
        if stage == 'test':
            self.test_data = DivingWithMasksDataset(
                root=self.data_dir,
                train=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data, batch_size=self.config['train_batch_size'], shuffle=True,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=True,
            collate_fn=utils.collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data, batch_size=self.config['train_batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False,
            collate_fn=utils.collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.config['test_batch_size'], shuffle=False,
            num_workers=self.config['num_workers'], pin_memory=True, drop_last=False,
            collate_fn=utils.collate_fn)
