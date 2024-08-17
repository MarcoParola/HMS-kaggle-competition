import torch
import pandas as pd
import flatdict
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torchvision.transforms as transforms
# from torchvision.transforms import v2

from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np


def get_checkpoint(cfg):
    """Returns an ModelCheckpoint callback
    cfg: hydra config
    """
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=cfg.train.save_path,
                                          filename=cfg.task+'_{epoch}-{step}',
                                          save_last = False
                                        )
    return checkpoint_callback

def get_lr_monitor(cfg):
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    return lr_monitor

def get_early_stopping(cfg):
    """Returns an EarlyStopping callback
    cfg: hydra config
    """
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=10,
    )
    return early_stopping_callback

class NormalizeEEG:
    def __init__(self, cfg):
        print("USing signal normalization")

        #leggi i valori train mean e std da file, secondo valore di ogni riga
        stats= pd.read_csv(cfg.dataset.eeg_stats)
        train_mean = stats['Mean']
        train_std = stats['Std']

        #converti in numpy array
        self.train_mean = torch.tensor(train_mean).to('cuda')
        self.train_std = torch.tensor(train_std).to('cuda')

        # print("Train mean: ", self.train_mean)
        # print("Train std: ", self.train_std)


    def __call__(self, eeg):
        # Converti eeg da dataframe a torch tensor e trasferisci sulla GPU
        eeg_tensor = torch.tensor(eeg.values, dtype=torch.float32).to('cuda')

        # print("eeg: ", eeg_tensor)
        normalized_eeg = (eeg_tensor - self.train_mean) / (self.train_std)
        # print("Normalized eeg: ", normalized_eeg)

        # transpose eeg tensor
        normalized_eeg = normalized_eeg.T

        return normalized_eeg.float()
        
class NormalizeEegFeatures:
    def __init__(self, cfg):
        print("Using eeg features normalization")

        #leggi statistiche da file
        eegs_stats= pd.read_csv(cfg.dataset.features_eeg_stats)
        
        # leggi campi Min e Max
        train_eegs_min = eegs_stats['Min']
        train_eegs_max = eegs_stats['Max']
    
        #converti in tensori
        self.train_eegs_min = torch.tensor(train_eegs_min).to('cuda')
        self.train_eegs_max = torch.tensor(train_eegs_max).to('cuda')

    def __call__(self, feature_vec):

        # print("feature_vec device:", feature_vec.device)
        # print("self.train_eegs_min device:", self.train_eegs_min.device)
        # print("self.train_eegs_max device:", self.train_eegs_max.device)

        # print("feature_vec: ", feature_vec)
        normalized_feature_vec = (feature_vec - self.train_eegs_min) / (self.train_eegs_max - self.train_eegs_min)
        # print("Normalized feature_vec: ", normalized_feature_vec)

        return normalized_feature_vec

class NormalizeSpecFeatures:
    def __init__(self, cfg):
        print("Using spec features normalization")

        #leggi statistiche da file
        specs_stats= pd.read_csv(cfg.dataset.features_spec_stats)

        train_specs_min = specs_stats['Min']
        train_specs_max = specs_stats['Max']

        #converti in numpy array
        self.train_specs_min = torch.tensor(train_specs_min).to('cuda')
        self.train_specs_max = torch.tensor(train_specs_max).to('cuda')

        # print("Train specs min: ", self.train_specs_min)
        # print("Train specs max: ", self.train_specs_max)

    def __call__(self, feature_vec):

        # print("feature_vec: ", feature_vec)
        normalized_feature_vec = (feature_vec - self.train_specs_min) / (self.train_specs_max - self.train_specs_min)
        # print("Normalized feature_vec: ", normalized_feature_vec)

        return normalized_feature_vec


def get_transformations(cfg):

    if cfg.dataset.dataset_type == 'full':
        eegs_stats= pd.read_csv(cfg.dataset.eeg_stats)
    if cfg.dataset.dataset_type == 'ge4':
        eegs_stats= pd.read_csv(cfg.dataset.eeg_stats_ge4)
    if cfg.dataset.dataset_type == 'hq':
        eegs_stats= pd.read_csv(cfg.dataset.eeg_stats_hq)

    eegs_train_mean = eegs_stats['Mean']
    eegs_train_std = eegs_stats['Std']
    eegs_train_min = eegs_stats['Min']
    eegs_train_max = eegs_stats['Max']

    eegs_train_mean = torch.tensor(eegs_train_mean).to('cuda')
    eegs_train_std = torch.tensor(eegs_train_std).to('cuda')
    eegs_train_min = torch.tensor(eegs_train_min).to('cuda')
    eegs_train_max = torch.tensor(eegs_train_max).to('cuda')

    def scale(x):
        if cfg.dataset.norm_type == 'mean_std':
            x = (x - eegs_train_mean) / eegs_train_std
        elif cfg.dataset.norm_type == 'min_max':
            x = (x - eegs_train_min) / (eegs_train_max - eegs_train_min)
        else:   
            raise ValueError("Invalid norm_type")
        return x.T.float()
    
    eegs_transform = transforms.Compose([
        # NormalizeEEG(cfg),
        scale,
    ])

    spectr_transform = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        # MixUp and RandomCutout augmentation
        # v2.RandomCutout(num_holes=1, max_h_size=10, max_w_size=10, fill_value=0, p=0.5),
        # v2.MixUp(),
        transforms.ToTensor(),     
                    
    ])

    eeg_features_transform = transforms.Compose([
        NormalizeEegFeatures(cfg)
    ])

    spec_features_transform = transforms.Compose([
        NormalizeSpecFeatures(cfg)
    ])

    return eegs_transform, spectr_transform, eeg_features_transform, spec_features_transform



def log_confusion_matrix_wandb(list_loggers, logger, y_true, preds, class_names):
    # check if wandb is in the list of loggers
    if 'wandb' in list_loggers:
        # logging confusion matrix on wandb
        logger.log({"conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_true,
                                                            preds=preds,
                                                            class_names=class_names)})


def hp_from_cfg(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return dict(flatdict.FlatDict(cfg, delimiter="/"))


def get_loggers(cfg):
    """Returns a list of loggers
    cfg: hydra config
    """
    loggers = list()
    if cfg.log.wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb
        hyperparameters = hp_from_cfg(cfg)
        wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
        wandb.config.update(hyperparameters)
        wandb_logger = WandbLogger()
        loggers.append(wandb_logger)

    return loggers


# Funzione per creare il filtro passabanda
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Funzione per applicare il filtro passabanda
def apply_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=200.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data, axis=0)
    return y