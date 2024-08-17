import hydra
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.data.dataset import FeatureDataset, HMSSignalClassificationDataset
from src.utils import get_transformations


class HMSSignalClassificationDataModule(LightningDataModule):
    def __init__(self, data_dir, task, freeze, highcut, dataset_type, batch_size=32, transform=None):
        super().__init__()

        print(f"Using only {dataset_type} data")

        if task=="eegsspectr" and freeze:
            print("Using FeatureDataset")
            self.train_dataset = FeatureDataset("train", data_dir, transform, dataset_type)
            self.val_dataset = FeatureDataset("val", data_dir, transform, dataset_type)
            self.test_dataset = FeatureDataset("test", data_dir, transform, dataset_type)   
        else:
            print("Using HMSSignalClassificationDataset")
            self.train_dataset = HMSSignalClassificationDataset("train", data_dir, task, freeze, highcut, dataset_type, transform=transform)
            self.val_dataset = HMSSignalClassificationDataset("val", data_dir, task, freeze, highcut, dataset_type, transform=transform)
            self.test_dataset = HMSSignalClassificationDataset("test", data_dir, task, freeze, highcut, dataset_type, transform=transform)
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
        freeze=True,
        transform=transformations
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