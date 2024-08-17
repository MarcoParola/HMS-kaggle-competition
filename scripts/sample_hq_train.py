import os
import pandas as pd
import numpy as np

dataset_dir = "../dataset"
percentage_tosample_per_class = 0.1

#leggere il dataset di train
train_df = pd.read_csv(os.path.join(dataset_dir, "train_hq_eegs.csv"))
# per ogni classe stampa il totale di campioni
print(train_df["expert_consensus"].value_counts())
# calcola un vettore con il numero di campioni da prelevare per ogni classe (10% per classe)
n_samples_per_class = (train_df["expert_consensus"].value_counts() * percentage_tosample_per_class).astype(int)
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


sampled_train_df = prop_sample_train_dataset(train_df, n_samples_per_class)
# per ogni classe stampa il totale di campioni
print(sampled_train_df["expert_consensus"].value_counts())

sampled_train_df.to_csv(os.path.join(dataset_dir, "sampled10_train_eegs.csv"), index=False)