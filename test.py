import os
import torch

from src.data.datamodule import HMSSignalClassificationDataModule
from src.utils import apply_bandpass_filter
from src.models.classification import HMSEEGClassifierModule, HMSEEGSpectrClassifierModule, HMSSpectrClassifierModule
from src.utils import get_transformations

from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from src.utils import *

import torch.utils.data
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
import numpy as np

import hydra
import pytorch_lightning as pl
import torch.nn.functional as F



class HMSSignalTestDataModule(LightningDataModule):
    def __init__(self, data_dir, task, freeze, highcut, norm_type, batch_size=32, transform=None):
        super().__init__()
        print("Using HMSSignalTestDataModule")
        self.test_dataset = HMSSignalTestDataset("test", data_dir, task, freeze, highcut, norm_type, transform=transform)
        self.batch_size = batch_size

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)



class HMSSignalTestDataset(Dataset):
    def __init__(self, stage, data_dir, task, freeze, highcut, dataset_type, augmentation, transform=None):
        print(f"Loading {stage} dataset in {task} mode")
        self.stage = stage
        self.data_dir = data_dir
        self.task = task
        self.freeze = freeze
        self.highcut = highcut
        self.dataset_type = dataset_type
        if dataset_type == 'full':
            csv_file = os.path.join(data_dir, f"{stage}_{task}.csv")
        if dataset_type == 'ge4':
            csv_file = os.path.join(data_dir, f"ge4_{stage}.csv")
        if dataset_type == 'hq':
            csv_file = os.path.join(data_dir, f"hq_{stage}.csv")
        data = pd.read_csv(csv_file)

        self.eeg_ids = data["eeg_id"]

        self.expert_consensus = data["expert_consensus"]
        self.seizure_vote = data["seizure_vote"]
        self.lpd_vote = data["lpd_vote"]
        self.gpd_vote = data["gpd_vote"]
        self.lrda_vote = data["lrda_vote"]
        self.grda_vote = data["grda_vote"]
        self.other_vote = data["other_vote"]

        self.class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

        self.transform = transform
        self.eeg_transform, self.spectr_transform, self.eeg_features_transform, self.spec_features_transform, self.eeg_augment, self.spectr_augment = transform 


    def __len__(self):
        return len(self.eeg_ids)
    

    def __getitem__(self, idx):

        eeg_id = self.eeg_ids[idx]


        if self.task == 'eegs':
            eeg_file = os.path.join(self.data_dir, f"filtered_eeg_windows_{self.highcut}Hz/{self.stage}/{eeg_id}.parquet")

            eeg_df = pd.read_parquet(eeg_file)
            eeg_tensor = torch.tensor(eeg_df.values).to('cuda')

            if self.eeg_transform:
                eeg = self.eeg_transform(eeg_tensor)

            eeg = eeg_tensor.T

            return eeg, eeg_id



@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.extend([get_early_stopping(cfg), get_checkpoint(cfg), get_lr_monitor(cfg)])
    # loggers = get_loggers(cfg)

    transformations = get_transformations(cfg)


    if cfg.task == 'eegs':
        ckpt_name = f"{cfg.train.save_path}eegs_{cfg.train.eegs_run_name}.ckpt"
        model = HMSEEGClassifierModule.load_from_checkpoint(ckpt_name)
    elif cfg.task == 'spectr':
        ckpt_name = f"{cfg.train.save_path}spectr_{cfg.train.spectr_run_name}.ckpt"
        model = HMSSpectrClassifierModule.load_from_checkpoint(ckpt_name)
    elif cfg.task == 'eegsspectr':
        ckpt_name = f"{cfg.train.save_path}eegsspectr_{cfg.train.eegsspectr_run_name}.ckpt"
        model = HMSEEGSpectrClassifierModule.load_from_checkpoint(ckpt_name)


    data = HMSSignalClassificationDataModule(
        data_dir=cfg.dataset.data_dir,
        # data_dir="./submission",
        task=cfg.task,
        freeze=cfg.train.freeze,
        highcut=cfg.dataset.highcut,
        batch_size=cfg.train.batch_size,
        transform=transformations,
        dataset_type=cfg.dataset.dataset_type,
        augmentation=cfg.dataset.augmentation
    )


    trainer = pl.Trainer(
        default_root_dir='logs/hms/',
        # logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.max_epochs,
        fast_dev_run=False,
        enable_progress_bar=True
    )

    # test model
    trainer.test(model, data.test_dataloader())

    # ----------------- SUBMISSION -----------------

    # test_df = pd.read_csv("./submission/test_eegs.csv")

    # for id in test_df['eeg_id'].astype(str).values:
    #     eeg = pd.read_parquet(f"./submission/{id}.parquet")
    #     eeg_window = eeg.iloc[:2000]
    #     filtered_eeg_window = apply_bandpass_filter(eeg_window.values)
    #     filtered_eeg_window_df = pd.DataFrame(filtered_eeg_window, columns=eeg_window.columns)
    #     filtered_eeg_window_df = filtered_eeg_window_df.astype(np.float32)
    #     if not os.path.exists("./submission/filtered_eeg_windows_40Hz"):
    #         os.makedirs("../submission/filtered_eeg_windows_40Hz")
    #     if not os.path.exists("./submission/filtered_eeg_windows_40Hz/test"):
    #         os.makedirs("./submission/filtered_eeg_windows_40Hz/test")
    #     filtered_eeg_window_df.to_parquet(f"./submission/filtered_eeg_windows_40Hz/test/{id}.parquet", index=False)
        
        
    # data = HMSSignalTestDataModule(
    #     data_dir="./submission",
    #     task="eegs",
    #     freeze=cfg.train.freeze,
    #     highcut=cfg.dataset.highcut,
    #     norm_type=cfg.dataset.norm_type,
    #     batch_size=cfg.train.batch_size,
    #     transform=transformations
    # )

    # model.eval()

    # label_names = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

    # # Esegui il modello sul dataset di test
    # all_probabilities = []
    # eeg_ids = []
    # with torch.no_grad():
    #     for batch in data.test_dataloader():
    #         signals, eeg_id_batch = batch
    #         eeg_id_batch = [int(eeg_id.item()) for eeg_id in eeg_id_batch]
    #         eeg_ids.extend(eeg_id_batch)

    #         if isinstance(signals, torch.Tensor):
    #             if len(signals.shape) == 2:  # Caso tensore 2D
    #                 signals = signals.unsqueeze(0)
    #         elif isinstance(signals, list):
    #             signals = torch.stack(signals)

    #         print(f"Shape of signals before modification: {signals.shape}")
    #         # Assumendo che `signals` sia di forma [batch_size, num_channels, num_features, sequence_length]
    #         # dobbiamo rimuovere `num_features` per ottenere [batch_size, num_channels, sequence_length]
    #         signals = signals.view(signals.size(0), signals.size(1), -1)  # Combina le ultime due dimensioni
    #         print(f"Shape of signals after modification: {signals.shape}")

    #         outputs = model(signals)
    #         probabilities = F.softmax(outputs, dim=1)
    #         all_probabilities.extend(probabilities.cpu().numpy())


    # probabilities_df = pd.DataFrame(all_probabilities, columns=label_names)
    # probabilities_df.insert(0, 'eeg_id', eeg_ids)
    # probabilities_df.to_csv('submission.csv', index=False)


    # print("Probabilità salvate con successo!")

if __name__ == "__main__":
    main()



