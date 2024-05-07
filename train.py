import os

import hydra
import pytorch_lightning as pl
import torch

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
    callbacks.append(get_early_stopping(cfg))
    loggers = get_loggers(cfg)

    transformations = get_transformations(cfg)

    if cfg.task == 'eeg':
        model = HMSEEGClassifierModule(
            signal_len=cfg.dataset.signal_length,
            num_classes=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )
    elif cfg.task == 'spectr':
        model = HMSSpectrClassifierModule(
            img_size=cfg.dataset.img_size,
            num_classes=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )
    

    data = HMSSignalClassificationDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.train.batch_size,
        mode = cfg.task,
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
        fast_dev_run=False
    )

    trainer.fit(model, data)
    
    # test step
    trainer.test(model, data.test_dataloader())


if __name__ == "__main__":
    main()
