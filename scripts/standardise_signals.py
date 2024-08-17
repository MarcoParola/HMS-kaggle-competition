import os
import pandas as pd
import torch
from tqdm import tqdm
import numpy as np

#preleva valore highcut da terminale
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--highcut", type=float, default=40.0, help="Highcut value for the filter")
args = parser.parse_args()

highcut = args.highcut

def standardise_signals():
    print("Using eeg features standardisation")

    # Read stats from CSV and move to GPU
    eegs_stats = pd.read_csv(f"../dataset/filtered_{int(highcut)}Hz_eeg_windows_stats.csv")
    train_eegs_mean = torch.tensor(eegs_stats['Mean'].values, dtype=torch.float32).cuda()
    train_eegs_std = torch.tensor(eegs_stats['Std'].values, dtype=torch.float32).cuda()

    # Create directories if they don't exist
    base_dir = f"../dataset/standardised_filtered_eeg_windows_{int(highcut)}Hz"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        os.makedirs(os.path.join(base_dir, "train"))
        os.makedirs(os.path.join(base_dir, "val"))
        os.makedirs(os.path.join(base_dir, "test"))

    # Read train, validation and test dataframes
    train_df = pd.read_csv("../dataset/train_eegs.csv")
    validation_df = pd.read_csv("../dataset/val_eegs.csv")
    test_df = pd.read_csv("../dataset/test_eegs.csv")

    train_labels = train_df['label_id'].astype(str).values
    validation_labels = validation_df['label_id'].astype(str).values
    test_labels = test_df['label_id'].astype(str).values

    # Standardise all files in the directory
    eeg_dir = f"../dataset/filtered_eeg_windows_{int(highcut)}Hz"

    for filename in tqdm(train_labels, desc="Standardising train EEGs"):
        file_path = os.path.join(eeg_dir, "train", filename + ".parquet")
        eeg = pd.read_parquet(file_path)
        
        eeg_tensor = torch.tensor(eeg.values, dtype=torch.float32).cuda()
        standardised_eeg = (eeg_tensor - train_eegs_mean) / train_eegs_std
        dest_dir = os.path.join(base_dir, "train")
        
        dest_path = os.path.join(dest_dir, filename + ".parquet")

        # Move standardized EEG back to CPU and save as parquet
        standardised_eeg_df = pd.DataFrame(standardised_eeg.cpu().numpy(), columns=eeg.columns)
        standardised_eeg_df = np.around(standardised_eeg_df, decimals=6)
        standardised_eeg_df.to_parquet(dest_path, index=False)

    for filename in tqdm(validation_labels, desc="Standardising val EEGs"):
        file_path = os.path.join(eeg_dir, "val", filename + ".parquet")
        eeg = pd.read_parquet(file_path)
        
        eeg_tensor = torch.tensor(eeg.values, dtype=torch.float32).cuda()
        standardised_eeg = (eeg_tensor - train_eegs_mean) / train_eegs_std
        dest_dir = os.path.join(base_dir, "val")
        
        dest_path = os.path.join(dest_dir, filename + ".parquet")

        # Move standardized EEG back to CPU and save as parquet
        standardised_eeg_df = pd.DataFrame(standardised_eeg.cpu().numpy(), columns=eeg.columns)
        standardised_eeg_df = np.around(standardised_eeg_df, decimals=6)
        standardised_eeg_df.to_parquet(dest_path, index=False)

    for filename in tqdm(test_labels, desc="Standardising test EEGs"):
        file_path = os.path.join(eeg_dir, "test", filename + ".parquet")
        eeg = pd.read_parquet(file_path)
        
        eeg_tensor = torch.tensor(eeg.values, dtype=torch.float32).cuda()
        standardised_eeg = (eeg_tensor - train_eegs_mean) / train_eegs_std
        dest_dir = os.path.join(base_dir, "test")
        
        dest_path = os.path.join(dest_dir, filename + ".parquet")

        # Move standardized EEG back to CPU and save as parquet
        standardised_eeg_df = pd.DataFrame(standardised_eeg.cpu().numpy(), columns=eeg.columns)
        standardised_eeg_df = np.around(standardised_eeg_df, decimals=6)
        standardised_eeg_df.to_parquet(dest_path, index=False)

    print("EEG files standardised")

standardise_signals()
