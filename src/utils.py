import flatdict
import torchvision
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import numpy as np
import pandas as pd
import pywt

import torch

import wandb
import torchvision.transforms as transforms
#import torcheeg.transforms as tet


def get_checkpoint(cfg):
    """Returns an ModelCheckpoint callback
    cfg: hydra config
    """
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=cfg.train.save_path,
                                          filename='{cfg.task}-{epoch}-{step}',
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
        patience=20,
    )
    return early_stopping_callback



def get_transformations(cfg):

    transform = transforms.Compose([
        # tet.BandDifferentialEntropy(),
        # tet.BaselineRemoval(),
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
    ])

    return transform


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

def denoise(x, wavelet='db8', level=1):
    def _maddest(d, axis=None):
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)
    ret = {key:[] for key in x.columns}
    for pos in x.columns:
        coeff = pywt.wavedec(x[pos], wavelet, mode="per")
        sigma = (1/0.6745) * _maddest(coeff[-level])
        uthresh = sigma * np.sqrt(2*np.log(len(x)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        ret[pos]=pywt.waverec(coeff, wavelet, mode='per')
    return pd.DataFrame(ret)

def interpolate(raw_df):
    df = raw_df.copy()
    df = df.interpolate(
        method='linear',
        axis=0,
        limit=1, # ref to 1 value
        limit_direction="both", # interpolate from pre and post values
        limit_area='inside',
    )
    return df

def replace_outlier(series, bias=1.5, upper=0.95, lower=0.05):
    lower_clip = series.quantile(lower)
    upper_clip = series.quantile(upper)
    iqr = upper_clip - lower_clip

    outlier_min = lower_clip - (iqr) * bias
    outlier_max = upper_clip + (iqr) * bias

    series = series.clip(outlier_min, outlier_max)
    series = series.fillna(series.median())  # Replace NaN values with median
    return series
