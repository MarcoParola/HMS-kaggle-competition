import flatdict
import torchvision
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torcheeg import transforms
import torcheeg.transforms as transforms
import torchvision.transforms as tv_transforms


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
    """Returns the transformations for the dataset
    cfg: hydra config
    """
    transform = torchvision.transforms.Compose([
        transforms.BaselineRemoval(),  # Rimuovi il segnale di base
        transforms.CWTSpectrum(),  # Converti il segnale in spettrogrammi con trasformata wavelet
        transforms.BandSignal(),  # Dividi il segnale in segnali nelle diverse bande di frequenza
        transforms.PearsonCorrelation(),  # Calcola i coefficienti di correlazione tra i segnali di diversi elettrodi
        transforms.PhaseLockingCorrelation(),  # Calcola i valori di blocco di fase tra i segnali di diversi elettrodi
        transforms.BandPowerSpectralDensity(),  # Calcola la densità spettrale di potenza del segnale nelle diverse
                                                # bande di frequenza
        transforms.MeanStdNormalize(),  # Normalizza il segnale con z-score
        transforms.RandomNoise(),  # Aggiungi rumore casuale al segnale
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


