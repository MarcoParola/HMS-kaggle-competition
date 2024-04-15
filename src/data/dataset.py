import torch
import torchvision.transforms as transforms
import os
import pandas as pd
import hydra


class HMSSignalClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, stage, data_dir, transform=None):
        self.stage = stage
        self.data_dir = data_dir
        csv_file = os.path.join(data_dir, f"{stage}.csv")
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

        self.class_to_int = {
            "Seizure": 0,
            "LPD": 1,
            "GPD": 2,
            "LRDA": 3,
            "GRDA": 4,
            "Other": 5
        }

        self.int_to_class = {
            0: "Seizure",
            1: "LPD",
            2: "GPD",
            3: "LRDA",
            4: "GRDA",
            5: "Other"
        }

        self.transform = transform

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        eeg_sub_id = self.eeg_sub_ids[idx]
        eeg_label_offset_seconds = self.eeg_label_offset_seconds[idx]

        label = self.expert_consensus[idx]

        # print("INFO: ", eeg_id, eeg_sub_id, eeg_label_offset_seconds, label)

        eeg_path = f"{self.data_dir}{self.stage}_eegs/{eeg_id}.parquet"

        eeg = pd.read_parquet(eeg_path)
        eeg_offset = int(eeg_label_offset_seconds)
        eeg = eeg.iloc[eeg_offset * 200:(eeg_offset + 50) * 200]  # extract 50 seconds

        central_index = len(eeg) // 2  # central index
        labeled_eeg = eeg.iloc[central_index - 1000:central_index + 1000]   # extract central 10 seconds

        # print("Forma dei dati __get_item__:", labeled_eeg.shape)

        eeg_tensor = torch.tensor(labeled_eeg.values, dtype=torch.float32)

        # print("Forma dei dati __get_item__:", eeg_tensor.shape)

        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)

        # Transpose the tensor to have the 20 channels as the last dimension
        eeg_tensor = eeg_tensor.permute(1, 0)  # (channels, time_steps) -> (time_steps, channels)


        label_int = self.class_to_int[label]

        return eeg_tensor, label_int        # tensor 2000 (10 second) time steps per 20 channels


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    from matplotlib import pyplot as plt

    dataset = HMSSignalClassificationDataset(
        "train", "../." + cfg.dataset.data_dir,
        transform=transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.ToTensor(),
        ])
    )

    eeg, label = dataset.__getitem__(0)
    print("Tipo di eeg:", type(eeg))
    print("Tipo di label:", type(label))
    print(label)

    # (eeg), (label) = dataset.__getitem__(0)
    #
    # print("Tipo di eeg:", type(eeg[3]))
    #
    # print("Dimensioni di eeg:", len(eeg))
    # print("Dimensioni di label:", len(label))
    #
    # print("Esempio di EEG:", eeg[0], eeg[1], eeg[2], eeg[3].shape)
    # print("Esempio di etichetta:", label[0], label[1], label[2])

    # ---------------------------------------------------

    df = pd.read_parquet('../.' + cfg.dataset.data_dir + '/train_eegs/568657.parquet')
    features = df.columns
    print(f'There are {len(features)} raw eeg features')
    print(list(features))
    # 'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG'


if __name__ == "__main__":
    main()
