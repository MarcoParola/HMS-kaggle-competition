import os
import hydra
import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.data.datamodule import HMSSignalClassificationDataModule
from src.models.classification import HMSEEGClassifierModule, HMSSpectrClassifierModule
from src.models.classification import HMSEEGSpectrClassifierModule
from src.utils import *

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg):

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")

    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.extend([get_early_stopping(cfg), get_checkpoint(cfg)])
    loggers = get_loggers(cfg)


    if cfg.task == 'eegs':
        print("EEG only")
        model = HMSEEGClassifierModule(
            signal_len=cfg.dataset.signal_length,
            num_classes=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs,
        )

    elif cfg.task == 'spectr':
        print("Spectrogram only")
        model = HMSSpectrClassifierModule(
            img_size=cfg.dataset.img_size,
            num_classes=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )

    elif cfg.task == 'eegsspectr':
        
        if cfg.train.freeze:
            print("***** EEG and Spectrogram - freezed backbone *****")
        else:
            print("***** EEG and Spectrogram - unfreezed backbone *****")
        print(f"feat_comb_mode: {cfg.train.feat_comb_mode}")

        model = HMSEEGSpectrClassifierModule(
            eegs_model_path=f"{cfg.train.save_path}eegs_{cfg.train.eegs_run_name}.ckpt",
            spectr_model_path=f"{cfg.train.save_path}spectr_{cfg.train.spectr_run_name}.ckpt",
            freeze = cfg.train.freeze,
            feat_comb_mode=cfg.train.feat_comb_mode,
            num_classes=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )

    transformations = get_transformations(cfg)

    data = HMSSignalClassificationDataModule(
        data_dir=cfg.dataset.data_dir,
        mode=cfg.task,
        freeze=cfg.train.freeze,
        highcut=cfg.dataset.highcut,
        norm_type=cfg.dataset.norm_type,
        batch_size=cfg.train.batch_size,
        transform=transformations        
    )

    # training
    trainer = pl.Trainer(
        default_root_dir='logs/hms/',
        logger=loggers,
        callbacks=callbacks,
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        max_epochs=cfg.train.max_epochs,
        enable_progress_bar=True
    )

    # train model
    trainer.fit(model, data)

    # test model
    trainer.test(model, data.test_dataloader())


if __name__ == "__main__":
    main()
