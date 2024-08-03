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

#stampare il valore highcut
print(f"Highcut value: {highcut}")

dataset_dir = "../dataset"
filtered_eeg_windows_dir = os.path.join(dataset_dir, f"filtered_eeg_windows_{int(highcut)}Hz")

# leggere il dataset di train
train_df = pd.read_csv(os.path.join(dataset_dir, "train_eegs.csv"))
# per ogni classe stampa il totale di campioni
print(train_df["expert_consensus"].value_counts())
# calcola un vettore con il numero di campioni da prelevare per ogni classe (10% per classe)
n_samples_per_class = (train_df["expert_consensus"].value_counts() * 0.1).astype(int)
# stampa il vettore
print(n_samples_per_class)

# campiona il dataset di train per avere un numero di campioni per classe uguale a n_samples_per_class
def prop_sample_train_dataset(train_df: pd.DataFrame, n_samples_per_class: pd.Series, random_seed: int = 42):
    sampled_train_df = pd.DataFrame()
    
    for class_label in train_df["expert_consensus"].unique():
        class_df = train_df[train_df["expert_consensus"] == class_label]
        
        sampled_class_df = class_df.sample(n=n_samples_per_class[class_label], random_state=random_seed, replace=False)

        sampled_train_df = pd.concat([sampled_train_df, sampled_class_df])

    return sampled_train_df

# sampled_train_df = prop_sample_train_dataset(train_df, n_samples_per_class)
# # per ogni classe stampa il totale di campioni
# print(sampled_train_df["expert_consensus"].value_counts())

# sampled_train_df.to_csv(os.path.join(dataset_dir, "sampled10_train_eegs.csv"), index=False)

# componi un dataframe con la concatenazione di tutti i csv di sampled_train_df
def create_eeg_windows_df(sampled_train_df: pd.DataFrame, eeg_windows_dir: str) -> pd.DataFrame:
    eeg_windows_df = pd.DataFrame()
    
    for index, row in tqdm(sampled_train_df.iterrows(), total=sampled_train_df.shape[0], desc="Processing EEG windows"):
        eeg_windows_file = os.path.join(eeg_windows_dir, "train", f"{row['label_id']}.parquet")
        df_eeg = pd.read_parquet(eeg_windows_file)
        
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

output_file_path = os.path.join(dataset_dir, f"filtered_{int(highcut)}Hz_eeg_windows_stats.csv")
# approssima tutti i valori di output_file_path a 6 decimali
stats_filtered_df = stats_filtered_df.round(6)
stats_filtered_df.to_csv(output_file_path, index_label='Channel', header=True)

print(f"File salvato in: {output_file_path}")
