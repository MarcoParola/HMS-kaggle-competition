import torch
import pandas as pd
import flatdict
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torchaudio
import torchaudio.transforms as T
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms

from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import resample


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
        
# Trasformazioni per segnali audio
class RandomTimeShift:
    def __init__(self, max_shift_sec=5):
        self.max_shift_sec = max_shift_sec

    def __call__(self, waveform, sample_rate=200):
        max_shift = int(self.max_shift_sec * sample_rate)
        shift = np.random.randint(-max_shift, max_shift)
        return torch.roll(waveform, shifts=shift, dims=1)

class RandomHorizontalFlipTime:
    def __call__(self, waveform):
        return waveform.flip(dims=[1])

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, waveform):
        noise = torch.normal(self.mean, self.std, size=waveform.size()).to(waveform.device)
        return waveform + noise

# Trasformazioni per spettrogrammi
class XYMasking:
    def __init__(self, mask_size=(10, 10)):
        self.mask_size = mask_size

    def __call__(self, image):
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=self.mask_size
        )
        image[i:i+h, j:j+w] = 0
        return image

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
        scale,
    ])

    spectr_transform = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor(),     
    ])

    eeg_features_transform = transforms.Compose([
        NormalizeEegFeatures(cfg)
    ])

    spec_features_transform = transforms.Compose([
        NormalizeSpecFeatures(cfg)
    ])

    eeg_augment = transforms.Compose([
        scale,
        RandomTimeShift(max_shift_sec=5),
        RandomHorizontalFlipTime(),
        AddGaussianNoise(mean=0.0, std=0.1)
    ])

    spectr_augment = transforms.Compose([
        transforms.Resize((cfg.dataset.img_size, cfg.dataset.img_size)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),  # Replaces RandomCutout
        transforms.RandomHorizontalFlip(),
        XYMasking(mask_size=(20, 20)),
    ])

    return eegs_transform, spectr_transform, eeg_features_transform, spec_features_transform, eeg_augment, spectr_augment



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


def augment(data):
    """
    Effettua l'oversampling del dataset per bilanciare le classi e 
    aggiunge una colonna "augmented" per indicare quali campioni sono duplicati.

    Args:
        data (pd.DataFrame): DataFrame contenente il dataset originale, 
        con una colonna 'expert_consensus' che indica le classi.

    Returns:
        pd.DataFrame: DataFrame bilanciato con la colonna "augmented" aggiunta.
    """
    # Determina la classe maggioritaria
    max_class_count = data['expert_consensus'].value_counts().max()

    # Lista per raccogliere i DataFrame di ogni classe bilanciata
    df_list = []

    # Itera su ogni classe
    for class_value in data['expert_consensus'].unique():
        # Seleziona i campioni della classe corrente
        df_class = data[data['expert_consensus'] == class_value]
        # Numero di campioni da aggiungere
        n_samples_to_add = max_class_count - len(df_class)

        if n_samples_to_add > 0:
            # Duplica i campioni della classe corrente fino a raggiungere la dimensione della classe maggioritaria
            df_class_balanced_original = df_class.copy()  # Mantieni gli originali
            df_class_balanced_augmented = resample(df_class, 
                                                   replace=True,  # Duplicazione permessa
                                                   n_samples=n_samples_to_add,  # Numero di campioni da aggiungere
                                                   random_state=42)  # Riproducibilità

            # Aggiungi la colonna "augmented"
            df_class_balanced_original['augmented'] = 'no'
            df_class_balanced_augmented['augmented'] = 'yes'

            # Combina originali e duplicati
            df_class_balanced = pd.concat([df_class_balanced_original, df_class_balanced_augmented])
        else:
            # Se la classe è già bilanciata, non duplicare
            df_class_balanced = df_class.copy()
            df_class_balanced['augmented'] = 'no'

        # Aggiungi alla lista
        df_list.append(df_class_balanced)

    # Combina tutte le classi bilanciate in un unico DataFrame
    df_balanced = pd.concat(df_list)
    #resetta l'indice
    df_balanced.reset_index(drop=True, inplace=True)
    print(f"Tipo di balanced: {type(df_balanced)}")

    return df_balanced

def apply_fft(self, eeg_tensor):
    # Applica la FFT al segnale EEG e ritorna il modulo della trasformata
    eeg_freq = torch.fft.fft(eeg_tensor, dim=-1)  # FFT lungo l'ultima dimensione
    eeg_freq = torch.abs(eeg_freq)  # Prendi il modulo della FFT
    return eeg_freq