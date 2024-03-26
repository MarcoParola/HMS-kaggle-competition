from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.data.dataset import HMSSignalClassificationDataset


class HMSSignalClassificationDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32, train_transform=None, val_transform=None,
                 test_transform=None, transform=None):
        super().__init__()

        self.train_dataset = HMSSignalClassificationDataset("train", data_dir, transform=transform)
        self.val_dataset = HMSSignalClassificationDataset("val", data_dir, transform=transform)
        self.test_dataset = HMSSignalClassificationDataset("test", data_dir, transform=transform)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


