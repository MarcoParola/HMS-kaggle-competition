import torch
import torch.utils.data
import torchvision.transforms as transforms
import os
import pandas as pd
import hydra
from sklearn.preprocessing import LabelEncoder
from PIL import Image


class HMSSignalClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, stage, data_dir, mode, transform=None):
        self.stage = stage
        self.data_dir = data_dir
        self.mode = mode
        csv_file = os.path.join(data_dir, f"{stage}_{mode}.csv")
        data = pd.read_csv(csv_file)

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

    def __len__(self):
        return len(self.eeg_ids)
    
    def __getitem__(self, idx):

        expert_consensus = self.expert_consensus[idx]

        label = self.label_encoder.transform([expert_consensus])[0]   #etichetta
        label_id = self.label_id[idx]

        if self.mode == 'eegs':
            eeg_file = os.path.join(self.data_dir, f"{self.stage}_eegs", f"{label_id}.csv")
            eeg = pd.read_csv(eeg_file)
            eeg_values = eeg.values.astype('float32').T
            eeg_tensor = torch.tensor(eeg_values) 
 
            if self.transform:
                image = self.transform(image)

            return eeg_tensor, label

        elif self.mode == 'spectr':
            eeg_file = os.path.join(self.data_dir, f"{self.stage}_spectr", f"{label_id}.png")
            image = Image.open(eeg_file).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, label


class HMSSignalClassificationSpectrDataset(torch.utils.data.Dataset):
    def __init__(self, stage, data_dir, mode, transform=None):
        self.stage = stage
        self.data_dir = data_dir
        csv_file = os.path.join(data_dir, f"{stage}_{mode}.csv")
        data = pd.read_csv(csv_file)

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

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):

        expert_consensus = self.expert_consensus[idx]
        label = self.label_encoder.transform([expert_consensus])[0]   # Etichetta
        label_id = self.label_id[idx]
        eeg_file = os.path.join(self.data_dir, f"{self.stage}_spectr", f"{label_id}.png")
        image = Image.open(eeg_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label



@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    dataset = HMSSignalClassificationDataset(
        "test", "../." + cfg.dataset.data_dir,
        transform=transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.ToTensor(),
        ])
    )

    eeg, label = dataset.__getitem__(0)
    print("Dimensioni di eeg:", eeg.size())
    print("Dimensioni di label:", label)

    print("Tipo di eeg:", type(eeg))
    print("Tipo di label:", type(label))

    print("Esempio di EEG:", eeg[0])
    print("Esempio di etichetta:", label)

    df = pd.read_parquet('../.' + cfg.dataset.data_dir + '/train_eegs/568657.parquet')
    features = df.columns
    print(f'There are {len(features)} raw eeg features')
    print(list(features))

    eeg, label = dataset.__getitem__(1)
    print("Dimensioni di eeg:", eeg.size())
    print("Dimensioni di label:", label)
    eeg, label = dataset.__getitem__(2)
    print("Dimensioni di eeg:", eeg.size())
    print("Dimensioni di label:", label)
    eeg, label = dataset.__getitem__(3)
    print("Dimensioni di eeg:", eeg.size())
    print("Dimensioni di label:", label)


if __name__ == "__main__":
    main()
