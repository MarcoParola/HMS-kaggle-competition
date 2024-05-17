import os
import hydra
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.datamodule import HMSSignalClassificationDataModule
from src.models.classification import HMSEEGClassifierModule, HMSSpectrClassifierModule
from src.utils import *

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):
    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.extend([get_early_stopping(cfg), get_checkpoint(cfg), get_lr_monitor(cfg)])
    loggers = get_loggers(cfg)

    transformations = get_transformations(cfg)


    if cfg.task == 'eegs':
        ckpt_name = "{cf.save_path}my_checkpoint_file.ckpt"
        model = HMSEEGClassifierModule.load_from_checkpoint(ckpt_name)
    elif cfg.task == 'spectr':
        ckpt_name = "{cf.save_path}my_checkpoint_file.ckpt"
        model = HMSSpectrClassifierModule.load_from_checkpoint(ckpt_name)
    elif cfg.task == 'eeg-spectr':
        ckpt_name = "{cf.save_path}my_checkpoint_file.ckpt"
        model = HMSEEGSpectrClassifierModule.load_from_checkpoint(ckpt_name)


    data = HMSSignalClassificationDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        mode=cfg.task,
        transform=transformations,
    )

    # training
    trainer = pl.Trainer(
        default_root_dir='logs/hms/',
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.max_epochs,
        fast_dev_run=False,
        enable_progress_bar=True
    )

    # test model
    trainer.test(model, data.test_dataloader())


if __name__ == "__main__":
    main()
