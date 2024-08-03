import os
import torch
import numpy as np

folder_path = '../dataset/features/train'

print("Current Working Directory:", os.getcwd())

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)

    # Load the tensor data from the file
    tensor_data = torch.load(file_path)

    # Get the shape of the tensor
    tensor_shape = tensor_data.shape

    # Print the file name and tensor shape
    # print(f"File: {file_name}, Tensor Shape: {tensor_shape}")

    # assert shape equals [128] if file_name contains 'features_spec'
    if 'features_spec' in file_name:
        assert tensor_shape == torch.Size([128]), f"Spec shape is not correct: {file_name} -> {tensor_shape}"
    elif 'features_eeg' in file_name:
        assert tensor_shape == torch.Size([128]), f"EEG shape is not correct: {file_name} -> {tensor_shape}"