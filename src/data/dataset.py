import torch
import torchvision.transforms as transforms
import os
import pandas as pd
import hydra

class HMSSignalClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, stage, data_dir, transform=None):
        self.stage = stage
        csv_file = os.path.join(data_dir, f"{stage}.csv")
        data = pd.read_csv(csv_file)

        self.eeg_ids = data["eeg_id"]
        self.eeg_sub_ids = data["eeg_sub_id"]
        self.spectrogram_ids = data["spectrogram_id"]
        self.spectrogram_sub_ids = data["spectrogram_sub_id"]

        self.seizure_vote = data["seizure_vote"]
        self.lpd_vote = data["lpd_vote"]
        self.gpd_vote = data["gpd_vote"]
        self.lrda_vote = data["lrda_vote"]
        self.grda_vote = data["grda_vote"]
        self.other_vote = data["other_vote"]

        self.transform = transform

    def __len__(self):
        return len(self.eeg_ids)

    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        eeg_sub_id = self.eeg_sub_ids[idx]
        spectrogram_id = self.spectrogram_ids[idx]
        spectrogram_sub_id = self.spectrogram_sub_ids[idx]

        seizure_vote = self.seizure_vote[idx]
        lpd_vote = self.lpd_vote[idx]
        gpd_vote = self.gpd_vote[idx]
        lrda_vote = self.lrda_vote[idx]
        grda_vote = self.grda_vote[idx]
        other_vote = self.other_vote[idx]

        eeg_path = f"data/{self.stage}_eegs/{eeg_id}.parquet"
        spectrogram_path = f"data/{self.stage}_spectrograms/{spectrogram_id}.parquet"

        eeg = pd.read_parquet(eeg_path)
        spectrogram = pd.read_parquet(spectrogram_path)

        '''
        TODO i due segnale sono in formato pandas, bisogna trasformarli in tensori.
        Questa cosa si può fare con transform, che compone le trasformazioni di pytorch in una pipeline.
        oppure si può fare direttamente qui, ma è più pulito farlo con transform (NB il parametro transform è già predisposto tra gli argomenti della classe).
        
        if self.transform:
            eeg = self.transform(eeg)
            spectrogram = self.transform(spectrogram)
        '''

        return (eeg, spectrogram), (seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote)



@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg):
    from matplotlib import pyplot as plt

    dataset = HMSSignalClassificationDataset(
        "train", cfg.dataset.data_dir,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    (egg, spectrogram), label = dataset.__getitem__(0)

    # 
    # 'Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG'
    print(spectrogram.keys())
    print(egg.keys())

    print("len eeg['Fp1']", len(egg['Fp1']))
    print("len spectrogram['LL_0.59']", len(spectrogram['LL_0.59']))

    plt.plot(spectrogram['LL_0.59'])
    plt.show()


if __name__ == "__main__":
    main()



