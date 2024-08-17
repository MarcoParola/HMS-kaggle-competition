import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from tqdm import tqdm

#preleva valore highcut da terminale
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--highcut", type=float, default=40.0, help="Highcut value for the filter")
args = parser.parse_args()

highcut = args.highcut
task="eegs"

#stampare il valore highcut
print(f"Highcut value: {highcut}")

dataset_dir = "../dataset"
filtered_eeg_windows_dir = os.path.join(dataset_dir, f"filtered_eeg_windows_{int(highcut)}Hz")
train_df = pd.read_csv(os.path.join(dataset_dir, "train_hq_eegs.csv"))

# componi un dataframe con la concatenazione di tutti i csv di sampled_train_df
def create_eeg_windows_df(sampled_train_df: pd.DataFrame, eeg_windows_dir: str) -> pd.DataFrame:
    eeg_windows_df = pd.DataFrame()
    
    for index, row in tqdm(sampled_train_df.iterrows(), total=sampled_train_df.shape[0], desc="Processing EEG windows"):
        eeg_windows_file = os.path.join(eeg_windows_dir, f"{row['label_id']}.csv")
        df_eeg = pd.read_csv(eeg_windows_file)
        
        eeg_windows_df = pd.concat([eeg_windows_df, df_eeg])
    
    return eeg_windows_df

# leggi sampled_train_df da file csv
sampled_train_df = pd.read_csv(os.path.join(dataset_dir, "sampled10_train_eegs.csv"))
filtered_eeg_windows_df = create_eeg_windows_df(sampled_train_df, filtered_eeg_windows_dir)
filtered_eeg_windows_df.head()

filtered_eeg_windows_df_mean = filtered_eeg_windows_df.mean()
filtered_eeg_windows_df_std = filtered_eeg_windows_df.std()
filtered_eeg_windows_df_min = filtered_eeg_windows_df.min()
filtered_eeg_windows_df_max = filtered_eeg_windows_df.max()

print(f"Mean of each channel: \n{filtered_eeg_windows_df_mean}")
print(f"Standard deviation of each channel: \n{filtered_eeg_windows_df_std}")
print(f"Min of each channel: \n{filtered_eeg_windows_df_min}")
print(f"Max of each channel: \n{filtered_eeg_windows_df_max}")

stats_filtered_df = pd.DataFrame({
    'Mean': filtered_eeg_windows_df_mean,
    'Std': filtered_eeg_windows_df_std,
    'Min': filtered_eeg_windows_df_min,
    'Max': filtered_eeg_windows_df_max
})

print("DataFrame stats:\n", stats_filtered_df)

output_file_path = os.path.join(dataset_dir, f"filtered_{int(highcut)}Hz_hq_eeg_windows_stats.csv")
# approssima tutti i valori di output_file_path a 6 decimali
stats_filtered_df = stats_filtered_df.round(6)
stats_filtered_df.to_csv(output_file_path, index_label='Channel', header=True)

print(f"File salvato in: {output_file_path}")
