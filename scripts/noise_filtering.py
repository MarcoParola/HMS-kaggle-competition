import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

dataset_dir = "../dataset"
eeg_windows_dir = os.path.join(dataset_dir, "eeg_windows")
dataset_df = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"))

eeg_files_with_nans = pd.read_csv(f"{dataset_dir}/eeg_files_with_nans.csv", header=None).values.flatten()

#remove rows with NaN values
dataset_df = dataset_df[~dataset_df["label_id"].isin(eeg_files_with_nans)]
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
highcut = 80.0
fs = 200.0  # Frequenza di campionamento (in Hz)


#filter entire dataset and save files in a new directory
filtered_eeg_windows_dir = os.path.join(dataset_dir, f'filtered_eeg_windows_{int(highcut)}Hz')
if not os.path.exists(filtered_eeg_windows_dir):
    os.makedirs(filtered_eeg_windows_dir)

def filter_dataset(dataset_df: pd.DataFrame, lowcut: float, highcut: float, fs: float, order: int) -> None:
    for index, row in dataset_df.iterrows():
        filename = row["label_id"]
        eeg_window = pd.read_csv(f"{eeg_windows_dir}/{filename}.csv")
        filtered_eeg_window = apply_bandpass_filter(eeg_window.values, lowcut, highcut, fs, order)
        #approssima tutti i valori di filtered_eeg_window a 2 decimali
        filtered_eeg_window = np.around(filtered_eeg_window, decimals=2)
        #aggiungi lo stesso header del file originale
        filtered_eeg_window = np.vstack([eeg_window.columns, filtered_eeg_window])      
        pd.DataFrame(filtered_eeg_window).to_csv(f"{filtered_eeg_windows_dir}/{filename}.csv", index=False, header=False)

filter_dataset(dataset_df, lowcut, highcut, fs, 5)