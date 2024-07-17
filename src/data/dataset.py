import torch
import torch.utils.data
import torchvision.transforms as transforms
import os
import pandas as pd
import hydra
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# from src.utils import apply_bandpass_filter



class HMSSignalClassificationDataset(Dataset):
    def __init__(self, stage, data_dir, mode, freeze, transform=None):
        print(f"Loading {stage} dataset in {mode} mode")
        self.stage = stage
        self.data_dir = data_dir
        self.mode = mode
        self.freeze = freeze
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
        self.eeg_transform, self.spectr_transform = transform
        # print(f"eeg_transform: {self.eeg_transform}")
        # print(f"spectr_transform: {self.spectr_transform}")

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):

        expert_consensus = self.expert_consensus[idx]

        label = self.label_encoder.transform([expert_consensus])[0]  # etichetta
        label = torch.tensor(label, dtype=torch.long)
        label_id = self.label_id[idx]

        if self.mode == 'eegs':
            eeg_file = os.path.join(self.data_dir, f"filtered_eeg_windows_40Hz/{label_id}.csv")
            eeg_df = pd.read_csv(eeg_file)
            # eeg_df = apply_bandpass_filter(eeg_df)

            if self.eeg_transform:
                eeg = self.eeg_transform(eeg_df)

            # print(f"EEG shape: {eeg.shape}")

            return eeg, label

        elif self.mode == 'spectr':
            spectr_file = os.path.join(self.data_dir, "spectr_windows", f"{label_id}.png")
            image = Image.open(spectr_file).convert('RGB')

            if self.spectr_transform:
                image = self.spectr_transform(image)

            return image, label
    
        elif self.mode == 'eegsspectr' and self.freeze==False:
            # eeg_file = os.path.join(self.data_dir, f"{self.stage}_{self.mode}", f"{label_id}.csv")
            eeg_file = os.path.join(self.data_dir, f"filtered_eeg_windows_40Hz/{label_id}.csv")
            eeg_df = pd.read_csv(eeg_file)
            # eeg_values = eeg_df.values.astype('float32').T
            # eeg = torch.tensor(eeg_values)
            if self.eeg_transform:
                eeg = self.eeg_transform(eeg_df)

            spectr_file = os.path.join(self.data_dir, "spectr_windows", f"{label_id}.png")
            image = Image.open(spectr_file).convert('RGB')

            if self.spectr_transform:
                image = self.spectr_transform(image)

            return (eeg, image), label



class FeatureDataset(Dataset):
    def __init__(self, stage, data_path, transform=None):

        data_path = os.path.join(data_path, "features", stage)

        self.data_path = data_path
        self.transform = transform 
        self.stage = stage

        self.eeg_transform, self.spectr_transform = transform

        eeg_features_list = []
        spec_features_list = []
        labels_list = []

        # ordinamento file in base al numero del nome del file                
        eeg_files = sorted([f for f in os.listdir(data_path) if 'features_eeg' in f], key=lambda x: int(x.rstrip(".pt").split("_")[-1]))
        spec_files = sorted([f for f in os.listdir(data_path) if 'features_spec' in f], key=lambda x: int(x.rstrip(".pt").split("_")[-1]))
        label_files = sorted([f for f in os.listdir(data_path) if 'label' in f], key=lambda x: int(x.rstrip(".pt").split("_")[-1]))

        assert len(eeg_files) == len(spec_files) == len(label_files), \
            "The number of EEG, spectral features, and label files must be equal."
        
        # Load all features and labels into memory
        for eeg_file, spec_file, label_file in zip(eeg_files, spec_files, label_files):
            # print(eeg_file, spec_file, label_file, "\n")
            eeg_feature_path = os.path.join(data_path, eeg_file)
            spec_feature_path = os.path.join(data_path, spec_file)
            label_path = os.path.join(data_path, label_file)
            
            eeg_feature = torch.load(eeg_feature_path)
            spec_feature = torch.load(spec_feature_path)
            label = torch.load(label_path)

            # se è la prima tripletta stampa le shape
            if len(eeg_features_list) == 0:
                print(f"EEG features shape: {eeg_feature.shape}")
                print(f"Spectrogram features shape: {spec_feature.shape}")
                print(f"Labels shape: {label.shape}")
            

            #transform eeg_features and spectrogram features
            if self.transform:
                eeg_feature = self.transform(eeg_feature)
                spec_feature = self.transform(spec_feature)

            eeg_features_list.append(eeg_feature)
            spec_features_list.append(spec_feature)
            labels_list.append(label)


            #read global min and max values from file
            global_min = pd.read_csv('../dataset/scripts/features_min_max.txt', header=None, skiprows=1).values.flatten()[0]
            global_max = pd.read_csv('../dataset/scripts/features_min_max.txt', header=None, skiprows=2).values.flatten()[0]

            print("Global min: ", global_min)
            print("Global max: ", global_max)

            #normalize eeg features and spectrogram features
            eeg_feature = (eeg_feature - global_min) / (global_max - global_min)
            spec_feature = (spec_feature - global_min) / (global_max - global_min)

            
            #convert eeeg_features_list, spec_features_list, labels_list to tensors
            self.eeg_features = torch.stack(eeg_features_list)
            self.spec_features = torch.stack(spec_features_list)
            self.labels = torch.stack(labels_list)

        #print tensors dimensions
        # print(f"EEG features tensor shape: {self.eeg_features.shape}")
        # print(f"Spectrogram features tensor shape: {self.spec_features.shape}")
        # print(f"Labels tensor shape: {self.labels.shape}")

        
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