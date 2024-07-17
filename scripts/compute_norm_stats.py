import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import numpy as np

# Parametri del filtro
lowcut = 0.5
highcut = 80.0
fs = 200.0  # Frequenza di campionamento (in Hz)

dataset_dir = "../dataset"
eeg_windows_dir = os.path.join(dataset_dir, "eeg_windows")
filtered_eeg_windows_dir = os.path.join(dataset_dir, f"filtered_eeg_windows_{int(highcut)}Hz")
dataset_df = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"))

def plot_eeg_histograms_with_gaussian(eeg_windows_df: pd.DataFrame, save_path) -> None:
    num_channels = len(eeg_windows_df.columns)
    cols = 2
    rows = (num_channels + cols - 1) // cols 
    
    # Creazione della figura e degli assi
    fig, axs = plt.subplots(rows, cols, figsize=(20, rows * 5))
    fig.suptitle('EEG Windows Histograms with Gaussian Fit')
    
    # Aggiustamento per rendere axs un array 1D per iterazione facile
    axs = axs.flatten()
    
    for i, channel in enumerate(eeg_windows_df.columns):

        sns.histplot(eeg_windows_df[channel], ax=axs[i], kde=False, stat="density", bins=100, color='lightblue')
        
        mean = eeg_windows_df[channel].mean()
        std = eeg_windows_df[channel].std()
        
        # Crea una sequenza di valori x per la gaussiana
        x = np.linspace(eeg_windows_df[channel].min(), eeg_windows_df[channel].max(), 100)
        
        # Calcola la distribuzione normale con la media e la deviaziaone standard calcolate
        y = norm.pdf(x, mean, std)
        
        # Plot della gaussiana
        axs[i].plot(x, y, color='red')
        axs[i].set_title(f"{channel} Histogram with Gaussian Fit")
        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Density')
    
    # Rimozione di assi vuoti
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path)
    plt.show()


# componi un dataframe con la concatenazione di tutti i csv di sampled_train_df
def create_eeg_windows_df(sampled_train_df: pd.DataFrame, eeg_windows_dir: str) -> pd.DataFrame:
    eeg_windows_df = pd.DataFrame()
    
    for index, row in sampled_train_df.iterrows():
        eeg_windows_file = os.path.join(eeg_windows_dir, f"{row['label_id']}.csv")
        df_eeg = pd.read_csv(eeg_windows_file)
        
        eeg_windows_df = pd.concat([eeg_windows_df, df_eeg])
    
    return eeg_windows_df

# *****************************************************************

#leggi sampled_train_df da file csv
sampled_train_df = pd.read_csv(os.path.join(dataset_dir, "sampled_train_eegs.csv"))
filtered_eeg_windows_df = create_eeg_windows_df(sampled_train_df, filtered_eeg_windows_dir)
filtered_eeg_windows_df.head()

#save eeg_windows_df.mean() and eeg_windows_df.std() in a csv file
filtered_eeg_windows_df_mean = filtered_eeg_windows_df.mean()
filtered_eeg_windows_df_std = filtered_eeg_windows_df.std()

#calcola la media e la deviazione standard di ogni canale
print(f"Mean of each channel: \n{filtered_eeg_windows_df_mean}")
print(f"Standard deviation of each channel: \n{filtered_eeg_windows_df_std}")

stats_filtered_df = pd.DataFrame({
    'Mean': filtered_eeg_windows_df_mean,
    'Std': filtered_eeg_windows_df_std
})

print("DataFrame con media e deviazione standard:\n", stats_filtered_df)

output_file_path = os.path.join(dataset_dir, f"filtered_{int(highcut)}Hz_eeg_windows_stats.csv")
#approssima tutti i valori di output_file_path a 2 decimali
stats_filtered_df = stats_filtered_df.round(2)
stats_filtered_df.to_csv(output_file_path, index_label='Channel', header=True)

print(f"File salvato in: {output_file_path}")

plot_eeg_histograms_with_gaussian(filtered_eeg_windows_df, f"../scripts/filtered_{int(highcut)}Hz_EEG_windows_histograms_with_gaussian_fit.png")

# *****************************************************************

# #leggi sampled_train_df da file csv
# sampled_train_df = pd.read_csv(os.path.join(dataset_dir, "sampled_train_eegs.csv"))
# eeg_windows_df = create_eeg_windows_df(sampled_train_df, eeg_windows_dir)
# eeg_windows_df.head()

# #save eeg_windows_df.mean() and eeg_windows_df.std() in a csv file
# eeg_windows_df_mean = eeg_windows_df.mean()
# eeg_windows_df_std = eeg_windows_df.std()

# #stampa la media e la deviazione standard di ogni canale
# print(f"Mean of each channel: \n{eeg_windows_df_mean}")
# print(f"Standard deviation of each channel: \n{eeg_windows_df_std}")

# stats_df = pd.DataFrame({
#     'Mean': eeg_windows_df_mean,
#     'Std': eeg_windows_df_std
# })

# print("DataFrame con media e deviazione standard:\n", stats_df)

# output_file_path = os.path.join(dataset_dir, "eeg_windows_stats.csv")
# stats_df.to_csv(output_file_path, index_label='Channel', header=True)

# print(f"File salvato in: {output_file_path}")