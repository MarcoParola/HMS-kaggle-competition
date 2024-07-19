import os
import numpy as np
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt

file_path = "../dataset/features/train/features_spec_0.pt" 

# Carica il file .pt
data = torch.load(file_path)

# Determina il tipo di dati e stampa informazioni
if isinstance(data, torch.Tensor):
    print("Il file contiene un singolo tensore.")
    print("Shape del tensore:", data.shape)
    print("Contenuto del tensore:")
    print(data)
elif isinstance(data, dict):
    print("Il file contiene un dizionario di tensori.")
    for key, value in data.items():
        print(f"Chiave: {key}, Shape del tensore: {value.shape}")
        print(f"Contenuto del tensore {key}:")
        print(value)
elif isinstance(data, list):
    print("Il file contiene una lista di tensori.")
    for idx, tensor in enumerate(data):
        print(f"Tensore {idx}: Shape del tensore: {tensor.shape}")
        print(f"Contenuto del tensore {idx}:")
        print(tensor)
else:
    print("Tipo di dati non riconosciuto.")


train_dir = "../dataset/features/train"


# # stampa quanti file iniziano con features_eeg, features_spec e label
# count_eeg_feat = 0
# count_spec_feat = 0
# count_label = 0
# for filename in os.listdir(train_dir):
#     if filename.startswith("features_eeg"):
#         count_eeg_feat += 1
#     elif filename.startswith("features_spec"):
#         count_spec_feat += 1
#     elif filename.startswith("label"):
#         count_label += 1
# print(f"Numero di file che iniziano con 'features_eeg': {count_eeg_feat}")
# print(f"Numero di file che iniziano con 'features_spec': {count_spec_feat}")
# print(f"Numero di file che iniziano con 'label': {count_label}")

# #stampa numero di rows in train_eegss.csv
# train_eegss = pd.read_csv("../dataset/train_eegsspectr.csv")
# print(f"Numero di righe in train_eegss.csv: {len(train_eegss)}")

# val_dir = "../dataset/features/val"
# # stampa quanti file iniziano con features_eeg, features_spec e label
# count_eeg_feat = 0
# count_spec_feat = 0
# count_label = 0
# for filename in os.listdir(val_dir):
#     if filename.startswith("features_eeg"):
#         count_eeg_feat += 1
#     elif filename.startswith("features_spec"):
#         count_spec_feat += 1
#     elif filename.startswith("label"):
#         count_label += 1
# print(f"Numero di file che iniziano con 'features_eeg': {count_eeg_feat}")
# print(f"Numero di file che iniziano con 'features_spec': {count_spec_feat}")
# print(f"Numero di file che iniziano con 'label': {count_label}")

# #stampa numero di rows in val_eegss.csv
# val_eegss = pd.read_csv("../dataset/val_eegsspectr.csv")
# print(f"Numero di righe in val_eegss.csv: {len(val_eegss)}")


# test_dir = "../dataset/features/test"
# # stampa quanti file iniziano con features_eeg, features_spec e label
# count_eeg_feat = 0
# count_spec_feat = 0
# count_label = 0
# for filename in os.listdir(test_dir):
#     if filename.startswith("features_eeg"):
#         count_eeg_feat += 1
#     elif filename.startswith("features_spec"):
#         count_spec_feat += 1
#     elif filename.startswith("label"):
#         count_label += 1
# print(f"Numero di file che iniziano con 'features_eeg': {count_eeg_feat}")
# print(f"Numero di file che iniziano con 'features_spec': {count_spec_feat}")
# print(f"Numero di file che iniziano con 'label': {count_label}")

# #stampa numero di rows in test_eegss.csv
# test_eegss = pd.read_csv("../dataset/test_eegsspectr.csv")
# print(f"Numero di righe in test_eegss.csv: {len(test_eegss)}")




# visualizza distribuzione valori delle features eeg delle features spec

# Collect all filenames that start with features_eeg and features_spec
eeg_files = [filename for filename in os.listdir(train_dir) if filename.startswith("features_eeg")]
spec_files = [filename for filename in os.listdir(train_dir) if filename.startswith("features_spec")]

# Randomly sample 1000 files from each list
eeg_sample_files = random.sample(eeg_files, min(1000, len(eeg_files)))
spec_sample_files = random.sample(spec_files, min(1000, len(spec_files)))

# Load the sampled files into the respective lists
eegs_features_sample = [torch.load(os.path.join(train_dir, filename)) for filename in eeg_sample_files]
specs_features_sample = [torch.load(os.path.join(train_dir, filename)) for filename in spec_sample_files]

# Concatenate the tensors into a single tensor
eegs_features_sample = torch.cat(eegs_features_sample)
specs_features_sample = torch.cat(specs_features_sample)

# calculate eeg features stats
eegs_min = eegs_features_sample.min()
eegs_max = eegs_features_sample.max()
eegs_mean = eegs_features_sample.mean()
eegs_std = eegs_features_sample.std()

# Print statistics for eegs_features_sample
print(f"Dimensione tensore eegs_features_sample: {eegs_features_sample.shape}")
print(f"Minimo valore: {eegs_min}")
print(f"Massimo valore: {eegs_max}")
print(f"Media: {eegs_mean}")
print(f"Deviazione standard: {eegs_std}")
print()

# Convert tensors to scalars
eegs_min = eegs_min.item()
eegs_max = eegs_max.item()
eegs_mean = eegs_mean.item()
eegs_std = eegs_std.item()

stats_eegs_features_df = pd.DataFrame({
    'Min': [eegs_min],
    'Max': [eegs_max],
    'Mean': [eegs_mean],
    'Std': [eegs_std]
})

stats_filtered_df = stats_eegs_features_df.round(4)
stats_filtered_df.to_csv("../dataset/features_eeg_stats.csv", header=True, index=False)

#calculate spec features stats
specs_min = specs_features_sample.min()
specs_max = specs_features_sample.max()
specs_mean = specs_features_sample.mean()
specs_std = specs_features_sample.std()

# Print statistics for specs_features_sample
print(f"Dimensione tensore specs_features_sample: {specs_features_sample.shape}")
print(f"Minimo valore: {specs_min}")
print(f"Massimo valore: {specs_max}")
print(f"Media: {specs_mean}")
print(f"Deviazione standard: {specs_std}")

# Convert tensors to scalars
specs_min = specs_min.item()
specs_max = specs_max.item()
specs_mean = specs_mean.item()
specs_std = specs_std.item()

stats_specs_features_df = pd.DataFrame({
    'Min': [specs_min],
    'Max': [specs_max],
    'Mean': [specs_mean],
    'Std': [specs_std]
})

stats_filtered_df = stats_specs_features_df.round(2)
stats_filtered_df.to_csv("../dataset/features_spec_stats.csv", header=True, index=False)

# Plot histogram for eegs_features_sample
plt.figure(figsize=(12, 6))
plt.hist(eegs_features_sample.cpu().numpy().flatten(), bins=100, color='blue', alpha=0.7)
plt.title("Distribuzione dei valori delle features EEG")
plt.xlabel("Valore")
plt.ylabel("Frequenza")
plt.savefig("eegs_features_distribution.png")
plt.show()

# Plot histogram for specs_features_sample
plt.figure(figsize=(12, 6))
plt.hist(specs_features_sample.cpu().numpy().flatten(), bins=100, color='red', alpha=0.7)
plt.title("Distribuzione dei valori delle features SPEC")
plt.xlabel("Valore")
plt.ylabel("Frequenza")
plt.savefig("specs_features_distribution.png")
plt.show()