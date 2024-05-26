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
    if cfg.train.seed == -1:
        random_data = os.urandom(4)
        seed = int.from_bytes(random_data, byteorder="big")
        cfg.train.seed = seed
    torch.manual_seed(cfg.train.seed)

    callbacks = list()
    callbacks.extend([get_early_stopping(cfg), get_checkpoint(cfg)])
    loggers = get_loggers(cfg)

    transformations = get_transformations(cfg)

    if cfg.task == 'eegs':
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
    elif cfg.task == 'eegsspectr':
        model = HMSEEGSpectrClassifierModule(
            freeze=cfg.train.freeze,
            eegs_model_path=f"{cfg.train.save_path}eegs_{cfg.train.eegs_run_name}.ckpt",
            spectr_model_path=f"{cfg.train.save_path}spectr_{cfg.train.spectr_run_name}.ckpt",
            num_classes=cfg.dataset.num_classes,
            lr=cfg.train.lr,
            max_epochs=cfg.train.max_epochs
        )


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

    # train model
    trainer.fit(model, data)

    # test model
    trainer.test(model, data.test_dataloader())

    # Get predictions and ground truth labels
    y_true = []
    y_pred = []
    for batch in data.test_dataloader():
        x, y = batch
        y_true.extend(y.numpy())
        y_pred.extend(model(x).argmax(dim=1).numpy())

    #cm = confusion_matrix(y_true, y_pred)

    #plt.figure(figsize=(10, 8))
    #sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.train_dataset.class_names, yticklabels=data.train_dataset.class_names)
    #plt.xlabel("Predicted labels")
    #plt.ylabel("True labels")
    #plt.title("Confusion Matrix")
    #plt.show()

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=data.train_dataset.class_names))


if __name__ == "__main__":
    main()
