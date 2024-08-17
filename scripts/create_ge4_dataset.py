import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#split dataset.csv in train, validation and test set maintaing the percentage of each class
def split_dataset(dataset_df: pd.DataFrame, train_size: float, validation_size: float, test_size: float, random_seed: int = 42):
    assert train_size + validation_size + test_size == 1.0, "train_size + validation_size + test_size must be equal to 1.0"
    assert train_size > 0 and validation_size > 0 and test_size > 0, "train_size, validation_size and test_size must be greater than 0"
    assert dataset_df["expert_consensus"].nunique() == 6, "The dataset must have only 6 classes"
    
    train_df = pd.DataFrame()
    validation_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for class_label in dataset_df["expert_consensus"].unique():
        class_df = dataset_df[dataset_df["expert_consensus"] == class_label]
        
        train_df_class, validation_test_df_class = train_test_split(class_df, test_size=validation_size + test_size, random_state=random_seed, stratify=class_df["expert_consensus"])
        validation_df_class, test_df_class = train_test_split(validation_test_df_class, test_size=test_size / (validation_size + test_size), random_state=random_seed, stratify=validation_test_df_class["expert_consensus"])

        train_df = pd.concat([train_df, train_df_class])
        validation_df = pd.concat([validation_df, validation_df_class])
        test_df = pd.concat([test_df, test_df_class])

    return train_df, validation_df, test_df


dataset_dir = "../dataset"
dataset_df = pd.read_csv(os.path.join(dataset_dir, "dataset.csv"))
print(f"Number of rows in the dataset: {dataset_df.shape[0]}")


# plotta la distribuzioe della somma delle colonne che finiscono per _vote
dataset_df.filter(regex='_vote$').sum(axis=1).hist(bins=50)
plt.title('Distribution of the Sum of Vote Counts')
plt.xlabel('Sum of Vote Counts')
plt.ylabel('Frequency')
plt.show()

# stampa il numero di sample in dataset_df con somma delle colonne che finiscono per _vote maggiore di 4
print("Number of rows in the dataset [vote_count >= 4]", dataset_df[dataset_df.filter(regex='_vote$').sum(axis=1) >= 4].shape[0])
# stampa il numero di sample in dataset_df con somma delle colonne che finiscono per _vote maggiore di 10
print("Number of rows in the dataset [vote_count >= 10]", dataset_df[dataset_df.filter(regex='_vote$').sum(axis=1) >= 10].shape[0])

dataset_ge4 = dataset_df[dataset_df.filter(regex='_vote$').sum(axis=1) >= 4]
dataset_ge10 = dataset_df[dataset_df.filter(regex='_vote$').sum(axis=1) >= 10]

eeg_files_with_nans = pd.read_csv(f"{dataset_dir}/eeg_files_with_nans.csv", header=None).values.flatten()
spectr_files_with_nans = pd.read_csv(f"{dataset_dir}/spectr_files_with_nans.csv", header=None).values.flatten()

#remove rows with NaN values
ge4_dataset = dataset_ge4[~dataset_ge4["label_id"].isin(eeg_files_with_nans)]
ge4_dataset.reset_index(drop=True, inplace=True)
print(f"Number of rows in the dataset after removing EEG NaN values: {ge4_dataset.shape[0]}")
ge4_dataset = ge4_dataset[~ge4_dataset["label_id"].isin(spectr_files_with_nans)]
ge4_dataset.reset_index(drop=True, inplace=True)
print(f"Number of rows in the dataset after removing spectr NaN values: {ge4_dataset.shape[0]}")

# Stampa delle proporzioni di ogni classe
print(ge4_dataset["expert_consensus"].value_counts(normalize=True))

# Split the dataset into train, validation, and test sets
train_df, validation_df, test_df = split_dataset(ge4_dataset, 0.7, 0.15, 0.15)
print(f"Number of rows in the training set: {train_df.shape[0]}")
print(f"Number of rows in the validation set: {validation_df.shape[0]}")
print(f"Number of rows in the test set: {test_df.shape[0]}")

# Save the train, validation, and test sets to CSV files
train_df.to_csv(os.path.join(dataset_dir, "ge4_train.csv"), index=False)
validation_df.to_csv(os.path.join(dataset_dir, "ge4_val.csv"), index=False)
test_df.to_csv(os.path.join(dataset_dir, "ge4_test.csv"), index=False)
print("Train, validation, and test sets saved to 'ge4_train.csv', 'ge4_val.csv', and 'ge4_test.csv', respectively")

print(train_df["expert_consensus"].value_counts(normalize=True))
print(validation_df["expert_consensus"].value_counts(normalize=True))
print(test_df["expert_consensus"].value_counts(normalize=True))

# Save the high-quality dataset to a CSV file
ge4_dataset.to_csv(os.path.join(dataset_dir, "ge4_dataset.csv"), index=False)
print("High-quality dataset saved to 'ge4_dataset.csv'")



percentage_tosample_per_class = 0.1

# campiona il dataset di train per avere un numero di campioni per classe uguale a n_samples_per_class
def prop_sample_train_dataset(train_df: pd.DataFrame, n_samples_per_class: pd.Series, random_seed: int = 42):
    sampled_train_df = pd.DataFrame()
    
    for class_label in train_df["expert_consensus"].unique():
        class_df = train_df[train_df["expert_consensus"] == class_label]
        
        sampled_class_df = class_df.sample(n=n_samples_per_class[class_label], random_state=random_seed, replace=False)

        sampled_train_df = pd.concat([sampled_train_df, sampled_class_df])

    return sampled_train_df

n_samples_per_class = (train_df["expert_consensus"].value_counts() * percentage_tosample_per_class).astype(int)
print(n_samples_per_class)
sampled_train_df = prop_sample_train_dataset(train_df, n_samples_per_class)
# per ogni classe stampa il totale di campioni
print(sampled_train_df["expert_consensus"].value_counts())

sampled_train_df.to_csv(os.path.join(dataset_dir, "sampled4_train_eegs.csv"), index=False)