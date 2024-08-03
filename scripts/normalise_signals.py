import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
import argparse

# Preleva valore highcut da terminale
parser = argparse.ArgumentParser()
parser.add_argument("--highcut", type=float, default=40.0, help="Highcut value for the filter")
args = parser.parse_args()

highcut = args.highcut

def normalise_signals():
    print("Using EEG features normalisation")

    # Leggi statistiche da CSV e sposta su GPU
    eegs_stats = pd.read_csv(f"../dataset/filtered_{int(highcut)}Hz_eeg_windows_stats.csv")
    train_eegs_min = torch.tensor(eegs_stats['Min'].values, dtype=torch.float32).cuda()
    train_eegs_max = torch.tensor(eegs_stats['Max'].values, dtype=torch.float32).cuda()

    # Crea directory se non esistono
    base_dir = f"../dataset/normalised_filtered_eeg_windows_{int(highcut)}Hz"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        os.makedirs(os.path.join(base_dir, "train"))
        os.makedirs(os.path.join(base_dir, "val"))
        os.makedirs(os.path.join(base_dir, "test"))

    # Leggi dataframes di train, validation e test
    train_df = pd.read_csv("../dataset/train_eegs.csv")
    validation_df = pd.read_csv("../dataset/val_eegs.csv")
    test_df = pd.read_csv("../dataset/test_eegs.csv")

    train_labels = train_df['label_id'].astype(str).values
    validation_labels = validation_df['label_id'].astype(str).values
    test_labels = test_df['label_id'].astype(str).values

    # Funzione di normalizzazione
    def normalise_and_save(file_path, dest_dir):
        try:
            eeg = pd.read_parquet(file_path)
            eeg_tensor = torch.tensor(eeg.values, dtype=torch.float32).cuda()
            normalised_eeg = (eeg_tensor - train_eegs_min) / (train_eegs_max - train_eegs_min)
            dest_path = os.path.join(dest_dir, os.path.basename(file_path))
            normalised_eeg_df = pd.DataFrame(normalised_eeg.cpu().numpy(), columns=eeg.columns)
            normalised_eeg_df = normalised_eeg_df.astype(np.float16)
            normalised_eeg_df.to_parquet(dest_path, index=False)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Normalizza tutti i file nella cartella
    eeg_dir = f"../dataset/filtered_eeg_windows_{int(highcut)}Hz"

    for filename in tqdm(train_labels, desc="Normalising train EEGs"):
        file_path = os.path.join(eeg_dir, "train", filename + ".parquet")
        normalise_and_save(file_path, os.path.join(base_dir, "train"))

    for filename in tqdm(validation_labels, desc="Normalising val EEGs"):
        file_path = os.path.join(eeg_dir, "val", filename + ".parquet")
        normalise_and_save(file_path, os.path.join(base_dir, "val"))

    for filename in tqdm(test_labels, desc="Normalising test EEGs"):
        file_path = os.path.join(eeg_dir, "test", filename + ".parquet")
        normalise_and_save(file_path, os.path.join(base_dir, "test"))

    print("EEG files normalised")

if __name__ == "__main__":
    normalise_signals()
