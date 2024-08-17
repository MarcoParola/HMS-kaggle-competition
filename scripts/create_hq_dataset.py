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
hq_dataset = dataset_ge10[~dataset_ge10["label_id"].isin(eeg_files_with_nans)]
hq_dataset.reset_index(drop=True, inplace=True)
print(f"Number of rows in the dataset after removing EEG NaN values: {hq_dataset.shape[0]}")
hq_dataset = hq_dataset[~hq_dataset["label_id"].isin(spectr_files_with_nans)]
hq_dataset.reset_index(drop=True, inplace=True)
print(f"Number of rows in the dataset after removing spectr NaN values: {hq_dataset.shape[0]}")

# Stampa delle proporzioni di ogni classe
print(hq_dataset["expert_consensus"].value_counts(normalize=True))

# Split the dataset into train, validation, and test sets
train_df, validation_df, test_df = split_dataset(hq_dataset, 0.7, 0.15, 0.15)
print(f"Number of rows in the training set: {train_df.shape[0]}")
print(f"Number of rows in the validation set: {validation_df.shape[0]}")
print(f"Number of rows in the test set: {test_df.shape[0]}")

# Save the train, validation, and test sets to CSV files
train_df.to_csv(os.path.join(dataset_dir, "hq_train.csv"), index=False)
validation_df.to_csv(os.path.join(dataset_dir, "hq_val.csv"), index=False)
test_df.to_csv(os.path.join(dataset_dir, "hq_test.csv"), index=False)
print("Train, validation, and test sets saved to 'hq_train.csv', 'hq_val.csv', and 'hq_test.csv', respectively")

print(train_df["expert_consensus"].value_counts(normalize=True))
print(validation_df["expert_consensus"].value_counts(normalize=True))
print(test_df["expert_consensus"].value_counts(normalize=True))

# Save the high-quality dataset to a CSV file
hq_dataset.to_csv(os.path.join(dataset_dir, "hq_dataset.csv"), index=False)
print("High-quality dataset saved to 'hq_dataset.csv'")


