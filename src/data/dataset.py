import os
import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset

import hydra
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from PIL import Image

from src.utils import augment, apply_stft, apply_cwt_to_eeg


class HMSSignalClassificationDataset(Dataset):

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

        # ----------------- Augmentation -----------------
        self.augmentation = augmentation
        if augmentation != 'no' and dataset_type == 'hq' and stage == 'train':
            print("Augmenting data")
            data = augment(data)
            #salva data in un file csv
            data.to_csv(os.path.join(data_dir, f"{stage}_{task}_augmented.csv"), index=False)
        else:
            print("Data not augmented")
            data["augmented"] = "no"
        self.augmented = data["augmented"]


        self.eeg_ids = data["eeg_id"]
        self.eeg_sub_ids = data["eeg_sub_id"]
        self.eeg_label_offset_seconds = data["eeg_label_offset_seconds"]
        self.label_id = data["label_id"]
        self.expert_consensus = data["expert_consensus"]
        self.seizure_vote = data["seizure_vote"]
        self.lpd_vote = data["lpd_vote"]
        self.gpd_vote = data["gpd_vote"]
        self.lrda_vote = data["lrda_vote"]
        self.grda_vote = data["grda_vote"]
        self.other_vote = data["other_vote"]

        self.class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.expert_consensus)

        self.transform = transform
        # passo anche spec_features_transform e eeg_features_transform per la features extraction
        self.eeg_transform, self.spectr_transform, self.eeg_features_transform, self.spec_features_transform, self.eeg_augment, self.spectr_augment = transform 

        print(f"Dataloader size: {len(self.eeg_ids)}")

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):

        expert_consensus = self.expert_consensus[idx]
        label = self.label_encoder.transform([expert_consensus])[0]
        label = torch.tensor(label, dtype=torch.long).to('cuda')
        label_id = self.label_id[idx]
        augmented = self.augmented[idx]
        
        seizure_vote = self.seizure_vote[idx]
        lpd_vote = self.lpd_vote[idx]
        gpd_vote = self.gpd_vote[idx]
        lrda_vote = self.lrda_vote[idx]
        grda_vote = self.grda_vote[idx]
        other_vote = self.other_vote[idx]

        # create a tensor with the votes
        votes = torch.tensor([seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote], dtype=torch.float32).to('cuda')
        votes_prob = F.softmax(votes, dim=0)

        if self.task == 'eegs':
            # eeg_file = os.path.join(self.data_dir, f"{self.task}_filtered_eeg_windows_{self.highcut}Hz/{self.stage}/{label_id}.parquet")
            eeg_file = os.path.join(self.data_dir, f"filtered_eeg_windows_{self.highcut}Hz/{label_id}.csv")
            eeg_df = pd.read_csv(eeg_file)

            # eeg_values = eeg_df.values.astype('float').T #.
            eeg_tensor = torch.tensor(eeg_df.values).to('cuda')

            if augmented == 'yes' and self.eeg_augment and self.augmentation != 'spectr':
                eeg = self.eeg_augment(eeg_tensor)
            else:
                eeg = self.eeg_transform(eeg_tensor)
            # eeg = self.eeg_transform(eeg_tensor)        # scommenta per aug solo su spectr
            print(f"EEG shape: {eeg.shape}")

            # Applicazione della FFT
            # eeg = torch.abs(torch.fft.rfft(eeg, dim=0))
            # print("Dimensione dell'output della RFFT:", eeg.shape)
            # print("Numero di frequenze:", eeg.shape[0])

            return eeg, (label, votes_prob)

        elif self.task == 'eegs_cwt':
            eeg_file = os.path.join(self.data_dir, f"cwt/{label_id}.npy")
            cwt_data = np.load(eeg_file)
            cwt_tensor = torch.tensor(cwt_data, dtype=torch.float32).to('cuda')

            # applicazione della STFT
            # image = apply_stft(image)
            # applicazione della Wavelet
            # image = apply_cwt_to_eeg(eeg_df.T)

            #stampa shape per debug
            print(f"EEG shape after transform: {cwt_tensor.shape}") # (2000, 64, 20)

            return cwt_tensor, (label, votes_prob)


        elif self.task == 'spectr':
            spectr_file = os.path.join(self.data_dir, "spectr_windows", f"{label_id}.png")
            image = Image.open(spectr_file).convert('RGB')

            if augmented == 'yes' and self.spectr_augment and self.augmentation != 'eeg':
                image = self.spectr_augment(image)
            else:
                image = self.spectr_transform(image)

            return image, (label, votes_prob)
            # return image, votes_prob
    
        elif self.task == 'eegsspectr' and self.freeze==False:

            # eeg_file = os.path.join(self.data_dir, f"{self.stage}_{self.task}", f"{label_id}.csv")
            eeg_file = os.path.join(self.data_dir, f"filtered_eeg_windows_{self.highcut}Hz/{label_id}.csv")
            eeg_df = pd.read_csv(eeg_file)

            eeg_tensor = torch.tensor(eeg_df.values).to('cuda')

            if augmented == 'yes' and self.eeg_augment and self.augmentation != 'spectr':
                eeg = self.eeg_augment(eeg_tensor)
            else:
                eeg = self.eeg_transform(eeg_tensor)

            spectr_file = os.path.join(self.data_dir, "spectr_windows", f"{label_id}.png")
            image = Image.open(spectr_file).convert('RGB')

            if augmented == 'yes' and self.spectr_augment and self.augmentation != 'eeg':
                image = self.spectr_augment(image)
            else:
                image = self.spectr_transform(image)

            return (eeg, image), label
            # return (eeg, image), votes_prob


class FeatureDataset(Dataset):
    def __init__(self, stage, data_path, augmentation, transform=None):

        if augmentation == 'both':
            print("Augmented both data")
            data_path = os.path.join(data_path, "features_augmented", stage)
        elif augmentation == 'eeg':
            print("Augmented EEG data")
            data_path = os.path.join(data_path, "features_augmented_eeg", stage)
        elif augmentation == 'spectr':
            print("Augmented Spectrogram data")
            data_path = os.path.join(data_path, "features_augmented_spec", stage)
        else:
            print("Data not augmented")
            data_path = os.path.join(data_path, "features", stage)

        self.data_path = data_path
        self.stage = stage
        self.eeg_transform, self.spectr_transform, self.eeg_features_transform, self.spec_features_transform, self.eeg_augment, self.spectr_augment = transform 

        eeg_files = sorted([f for f in os.listdir(data_path) if 'features_eeg' in f], key=lambda x: int(x.rstrip(".pt").split("_")[-1]))
        spec_files = sorted([f for f in os.listdir(data_path) if 'features_spec' in f], key=lambda x: int(x.rstrip(".pt").split("_")[-1]))
        label_files = sorted([f for f in os.listdir(data_path) if 'label' in f], key=lambda x: int(x.rstrip(".pt").split("_")[-1]))

        assert len(eeg_files) == len(spec_files) == len(label_files), "The number of EEG, spectral features, and label files must be equal."

        self.eeg_features = torch.stack([torch.load(os.path.join(data_path, f)) for f in eeg_files])
        self.spec_features = torch.stack([torch.load(os.path.join(data_path, f)) for f in spec_files])
        self.labels = torch.stack([torch.load(os.path.join(data_path, f)) for f in label_files])

        # Apply transformations if they are provided and compatible with PyTorch tensors
        if self.eeg_features_transform:
            self.eeg_features = torch.stack([self.eeg_features_transform(f) for f in self.eeg_features])
        if self.spec_features_transform:
            self.spec_features = torch.stack([self.spec_features_transform(f) for f in self.spec_features])

        # Print tensors dimensions for debugging
        print(f"EEG features tensor shape: {self.eeg_features.shape}")
        print(f"Spectrogram features tensor shape: {self.spec_features.shape}")
        print(f"Labels tensor shape: {self.labels.shape}")

    def __getitem__(self, index):
        eeg = self.eeg_features[index]
        spec = self.spec_features[index]
        label = self.labels[index]
        return (eeg, spec), label

    def __len__(self):
        return len(self.labels)


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    dataset = HMSSignalClassificationDataset(
        "test", "../." + cfg.dataset.data_dir, cfg.task,
        transform=transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    )

    data, label = dataset.__getitem__(0)
    print("Dimensioni di data:", data.size())
    print("Dimensioni di label:", label)

    print("Tipo di data:", type(data))
    print("Tipo di label:", type(label))

    print("Esempio di data:", data[0])
    print("Esempio di etichetta:", label)

    if (cfg.task) == 'eegs':
        df = pd.read_csv(f'../.{cfg.dataset.data_dir}train_{cfg.task}/338.csv')
        features = df.columns
        print(f'There are {len(features)} raw {cfg.task} features')
        print(list(features))
    elif cfg.task == 'spectr':
        image = Image.open(f'../.{cfg.dataset.data_dir}train_{cfg.task}/29652.png').convert('RGB')

    data, label = dataset.__getitem__(1)
    print("Dimensioni di data:", data.size())
    print("Dimensioni di label:", label)
    data, label = dataset.__getitem__(2)
    print("Dimensioni di data:", data.size())
    print("Dimensioni di label:", label)
    data, label = dataset.__getitem__(3)
    print("Dimensioni di data:", data.size())
    print("Dimensioni di label:", label)


if __name__ == "__main__":
    main()