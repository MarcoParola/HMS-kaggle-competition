import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import concurrent.futures

dataset_csv_path = "dataset/dataset.csv"
dataset_df = pd.read_csv(dataset_csv_path)
eegs_folder = "dataset/eegs"
eeg_windows_path = "dataset/eeg_windows"

files_with_nan = []


# Funzione per estrarre la finestra di 10 secondi
def extract_window(eeg_id, offset_seconds):
    sample_rate = 200
    window_len = 50

    eeg = pd.read_parquet(os.path.join(eegs_folder, f"{eeg_id}.parquet"))
    eeg_sub = eeg.iloc[int(offset_seconds) * sample_rate:(int(offset_seconds) + window_len) * sample_rate]  # extract 50 seconds
    labeled_eeg = eeg_sub.iloc[20 * sample_rate:30 * sample_rate]  # extract central 10 seconds

    # Controllo dei valori NaN
    if labeled_eeg.isnull().values.any():
        files_with_nan.append(eeg_id)

    return labeled_eeg



# Funzione per estrarre la finestra di 10 secondi in parallelo
def extract_window_parallel(args):
    eeg_id, offset_seconds = args
    return extract_window(eeg_id, offset_seconds)


# Controllo e rigenerazione del file se il numero di righe nel DataFrame |0 2000
def regenerate_file_if_needed(window_csv_path):
    if os.path.exists(window_csv_path):
        window_df = pd.read_csv(window_csv_path)
        if len(window_df) != 2000:
            os.remove(window_csv_path)
            return True
    return False


# Esecuzione parallela dell'estrazione delle finestre EEG
def parallel_extraction():
    args_list = [(row["eeg_id"], row["eeg_label_offset_seconds"]) for index, row in dataset_df.iterrows()]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        windows = list(tqdm(executor.map(extract_window_parallel, args_list), total=len(args_list)))

    return windows


# Controllo per verificare se label_id unique count uguale alla lunghezza del DataFrame dataset_df
if dataset_df["label_id"].nunique() != len(dataset_df):
    raise ValueError("Il conteggio unico di label_id non e uguale alla lunghezza del DataFrame dataset_df.")

if not os.path.exists(eeg_windows_path):
    os.makedirs(eeg_windows_path)

# Controllo e rigenerazione dei file delle finestre EEG
for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
    label_id = row["label_id"]
    window_csv_path = os.path.join(eeg_windows_path, f"{label_id}.csv")

    if regenerate_file_if_needed(window_csv_path):
        continue

    if os.path.exists(window_csv_path):
        continue

    window = extract_window(row["eeg_id"], row["eeg_label_offset_seconds"])
    window.to_csv(window_csv_path, index=False)

# Esecuzione parallela dell'estrazione delle finestre EEG e salvataggio in file CSV
windows = parallel_extraction()
for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
    label_id = row["label_id"]
    window_csv_path = os.path.join(eeg_windows_path, f"{label_id}.csv")

    if not os.path.exists(window_csv_path):
        window = windows[index]
        window.to_csv(window_csv_path, index=False)


