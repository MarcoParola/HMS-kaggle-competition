import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from tqdm import tqdm

#preleva valore highcut da terminale
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--highcut", type=float, default=40.0, help="Highcut value for the filter")
args = parser.parse_args()

task = "eegs"

dataset_dir = "../dataset"
eeg_windows_dir = os.path.join(dataset_dir, "eeg_windows")
dataset_df = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"))

eeg_files_with_nans = pd.read_csv(f"{dataset_dir}/eeg_files_with_nans.csv", header=None).values.flatten()
spectr_files_with_nans = pd.read_csv(f"{dataset_dir}/spectr_files_with_nans.csv", header=None).values.flatten()

 #remove rows with NaN values
if task == "eegs" or task == "eegsspectr":
    dataset_df = dataset_df[~dataset_df["label_id"].isin(eeg_files_with_nans)]
    dataset_df.reset_index(drop=True, inplace=True)
    print(f"Number of rows in the dataset after removing NaN values: {dataset_df.shape[0]}")
if task == "spectr" or task == "eegsspectr":
    dataset_df = dataset_df[~dataset_df["label_id"].isin(spectr_files_with_nans)]
    dataset_df.reset_index(drop=True, inplace=True)
    print(f"Number of rows in the dataset after removing NaN values: {dataset_df.shape[0]}")


# Stampa delle proporzioni di ogni classe
print(dataset_df["expert_consensus"].value_counts(normalize=True))


# Funzione per creare il filtro passabanda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Funzione per applicare il filtro passabanda
def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y


def plot_signals(before, after, n_channels=20, columns=None):
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3 * n_channels))
    time = np.arange(before.shape[0])
    
    for i in range(n_channels):
        axes[i, 0].plot(time, before[:, i], label=f'Canale {columns[i] if columns is not None else i+1}')
        axes[i, 0].set_title(f'Segnale originale - Canale {columns[i] if columns is not None else i+1}')
        axes[i, 0].set_xlabel('Tempo')
        axes[i, 0].set_ylabel('Ampiezza')
        axes[i, 0].legend()
        
        axes[i, 1].plot(time, after[:, i], label=f'Canale {columns[i] if columns is not None else i+1}')
        axes[i, 1].set_title(f'Segnale filtrato - Canale {columns[i] if columns is not None else i+1}')
        axes[i, 1].set_xlabel('Tempo')
        axes[i, 1].set_ylabel('Ampiezza')
        axes[i, 1].legend()
    
    plt.tight_layout()
    plt.show()

# Parametri del filtro
lowcut = 0.5
highcut = args.highcut
fs = 200.0  # Frequenza di campionamento (in Hz)


#filter entire dataset and save files in a new directory
filtered_eeg_windows_dir = os.path.join(dataset_dir, f'{task}_filtered_eeg_windows_{int(highcut)}Hz')
if not os.path.exists(filtered_eeg_windows_dir):
    os.makedirs(filtered_eeg_windows_dir)
    # crea all'interno le cartelle train, val e test
    os.makedirs(os.path.join(filtered_eeg_windows_dir, 'train'))
    os.makedirs(os.path.join(filtered_eeg_windows_dir, 'val'))
    os.makedirs(os.path.join(filtered_eeg_windows_dir, 'test'))

# leggi train, val e test csv
train_df = pd.read_csv(f"../dataset/train_{task}.csv")
validation_df = pd.read_csv(f"../dataset/val_{task}.csv")
test_df = pd.read_csv(f"../dataset/test_{task}.csv")

train_labels = train_df['label_id'].astype(str).values
validation_labels = validation_df['label_id'].astype(str).values
test_labels = test_df['label_id'].astype(str).values

def filter_dataset(dataset_df: pd.DataFrame, lowcut: float, highcut: float, fs: float, order: int) -> None:

    for index, row in tqdm(dataset_df.iterrows(), total=dataset_df.shape[0]):
        filename = str(row["label_id"])
        eeg_window = pd.read_csv(f"{eeg_windows_dir}/{filename}.csv")
        filtered_eeg_window = apply_bandpass_filter(eeg_window.values, lowcut, highcut, fs, order)
        #approssima tutti i valori di filtered_eeg_window a 6 decimali
        filtered_eeg_window = np.around(filtered_eeg_window, decimals=6)

        # Determina il percorso di destinazione
        if filename in train_labels:
            dest_dir = os.path.join(filtered_eeg_windows_dir, "train")
        elif filename in validation_labels:
            dest_dir = os.path.join(filtered_eeg_windows_dir, "val")
        elif filename in test_labels:
            dest_dir = os.path.join(filtered_eeg_windows_dir, "test")
        else:
            raise ValueError("Filename not found in train, validation or test datasets")
        
        dest_path = os.path.join(dest_dir, filename + ".parquet")

        #aggiungi lo stesso header del file originale
        # filtered_eeg_window = np.vstack([eeg_window.columns, filtered_eeg_window])      
        # pd.DataFrame(filtered_eeg_window).to_csv(dest_path, index=False, header=False)

        filtered_eeg_window_df = pd.DataFrame(filtered_eeg_window, columns=eeg_window.columns)
        filtered_eeg_window_df = filtered_eeg_window_df.astype(np.float32)
        # Save the filtered DataFrame as a Parquet file
        filtered_eeg_window_df.to_parquet(dest_path, index=False)    

filter_dataset(dataset_df, lowcut, highcut, fs, 5)