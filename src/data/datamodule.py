from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.data.dataset import HMSSignalClassificationDataset
from src.utils import get_transformations
import hydra


class HMSSignalClassificationDataModule(LightningDataModule):
    def __init__(self, data_dir, mode, batch_size=32, train_transform=None, val_transform=None,
                 test_transform=None, transform=None):
        super().__init__()

        self.train_dataset = HMSSignalClassificationDataset("train", data_dir, mode, transform=transform)
        self.val_dataset = HMSSignalClassificationDataset("val", data_dir, mode, transform=transform)
        self.test_dataset = HMSSignalClassificationDataset("test", data_dir, mode, transform=transform)
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    transformations = get_transformations(cfg)

    data = HMSSignalClassificationDataModule(
        data_dir="../."+cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        transform=transformations,
    )

    train_dataloader = DataLoader(data.train_dataset, batch_size=8)
    val_dataloader = DataLoader(data.val_dataset, batch_size=8)
    test_dataloader = DataLoader(data.test_dataset, batch_size=8)

    print("Dimensione del train_dataloader:")
    print(len(train_dataloader.dataset))
    print("Dimensione del val_dataloader:")
    print(len(val_dataloader.dataset))
    print("Dimensione del test_dataloader:")
    print(len(test_dataloader.dataset))

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, L]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    for X, y in val_dataloader:
        print(f"Shape of X [N, C, L]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, L]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break


if __name__ == "__main__":
    main()
