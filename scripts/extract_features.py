import torch
import torch.nn as nn
import os
import hydra
from src.data.datamodule import HMSSignalClassificationDataModule
from src.models.classification import HMSEEGClassifierModule, HMSSpectrClassifierModule
from src.models.classification import HMSEEGSpectrClassifierModule
from src.utils import *

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):

    transformations = get_transformations(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # istanza del datamodule
    data = HMSSignalClassificationDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        transform=transformations,
        freeze=False,
        task='eegsspectr',
        highcut=cfg.dataset.highcut,
        dataset_type=cfg.dataset.dataset_type,
        augmentation=cfg.dataset.augmentation
    )

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    #stampa dimensione dei dataset
    print("Dimensione del train_dataloader:")
    print(len(data.train_dataloader().dataset))
    print("Dimensione del val_dataloader:")
    print(len(data.val_dataloader().dataset))
    print("Dimensione del test_dataloader:")
    print(len(data.test_dataloader().dataset))

    batch_size = train_loader.batch_size

    eegs_model_path=f"{cfg.train.save_path}eegs_{cfg.train.eegs_run_name}.ckpt"
    spectr_model_path=f"{cfg.train.save_path}spectr_{cfg.train.spectr_run_name}.ckpt"

    # istanza del modello
    model_eeg = HMSEEGClassifierModule.load_from_checkpoint(eegs_model_path).to(device)
    model_spec = HMSSpectrClassifierModule.load_from_checkpoint(spectr_model_path).to(device)

    print(f"EEG model loaded from {eegs_model_path}")
    print(f"Spectrogram model loaded from {spectr_model_path}")

    # freeze the weights of the models
    model_eeg.freeze()
    model_spec.freeze()
    
    if cfg.dataset.augmentation == 'both':
        print("Data augmentation enabled")
        features_path = './dataset/features_augmented/'
    elif cfg.dataset.augmentation == 'eeg':
        print("Data augmentation enabled for EEG data")
        features_path = './dataset/features_augmented_eeg/'
    elif cfg.dataset.augmentation == 'spectr':
        print("Data augmentation enabled for Spectrogram data")
        features_path = './dataset/features_augmented_spec/'
    else:
        print("Data augmentation disabled")
        features_path = './dataset/features/'
    train_feature_path = os.path.join(features_path, 'train')
    test_feature_path = os.path.join(features_path, 'test')
    val_feature_path = os.path.join(features_path, 'val')

    #create the directories if they do not exist
    if not os.path.exists(features_path):
        os.makedirs(features_path)
    if not os.path.exists(train_feature_path):
        os.makedirs(train_feature_path)
    if not os.path.exists(test_feature_path):
        os.makedirs(test_feature_path)
    if not os.path.exists(val_feature_path):
        os.makedirs(val_feature_path)

    print("Directories created")

    for i, batch in enumerate(train_loader):
        x, y = batch
        eeg, spec = x

        eeg = eeg.to(device)
        spec = spec.to(device)

        #se è la prima iterazione stampa la shape
        if i == 0:
            print("EEG shape:", eeg.shape)
            print("Spec shape:", spec.shape)

        out_eeg = model_eeg.extract_features(eeg)
        out_spec = model_spec.extract_features(spec)

        for j in range(len(out_eeg)):
            # save the features to the path
            file_name = os.path.join(train_feature_path, f'features_eeg_{i*batch_size+j}.pt')
            torch.save(out_eeg[j], file_name)
            file_name = os.path.join(train_feature_path, f'features_spec_{i*batch_size+j}.pt')
            torch.save(out_spec[j], file_name)
            # save label
            file_name = os.path.join(train_feature_path, f'label_{i*batch_size+j}.pt')
            torch.save(y[j], file_name)



    for i, batch in enumerate(test_loader):
        x, y = batch
        eeg, spec = x

        eeg = eeg.to(device)
        spec = spec.to(device)

        out_eeg = model_eeg.extract_features(eeg)
        out_spec = model_spec.extract_features(spec)

        for j in range(len(out_eeg)):
            # save the features to the path
            file_name = os.path.join(test_feature_path, f'features_eeg_{i*batch_size+j}.pt')
            torch.save(out_eeg[j], file_name)
            file_name = os.path.join(test_feature_path, f'features_spec_{i*batch_size+j}.pt')
            torch.save(out_spec[j], file_name)
            # save label
            file_name = os.path.join(test_feature_path, f'label_{i*batch_size+j}.pt')
            torch.save(y[j], file_name)



    for i, batch in enumerate(val_loader):
        x, y = batch
        eeg, spec = x

        eeg = eeg.to(device)
        spec = spec.to(device)

        out_eeg = model_eeg.extract_features(eeg)
        out_spec = model_spec.extract_features(spec)

        for j in range(len(out_eeg)):
            # save the features to the path
            file_name = os.path.join(val_feature_path, f'features_eeg_{i*batch_size+j}.pt')
            torch.save(out_eeg[j], file_name)
            file_name = os.path.join(val_feature_path, f'features_spec_{i*batch_size+j}.pt')
            torch.save(out_spec[j], file_name)
            # save label
            file_name = os.path.join(val_feature_path, f'label_{i*batch_size+j}.pt')
            torch.save(y[j], file_name)

if __name__ == '__main__':
    main()