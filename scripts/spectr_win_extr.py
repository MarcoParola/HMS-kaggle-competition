import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
from PIL import Image

dataset_csv_path = "dataset/dataset.csv"
dataset_df = pd.read_csv(dataset_csv_path)
spectr_folder = "dataset/spectrograms"
spectr_windows_path = "dataset/spectr_windows"

spec_zones=['LL','RL','LP','RP']

def generate_spectrogram(specid, specoffset, labelid='sample',
                         input_path="dataset/spectrograms/", spec_out="dataset/spectr_windows/",
                         spec_zones=spec_zones,
                         output_filter=0.7):

    #get spec data
    spec = pd.read_parquet(f'{input_path}{specid}.parquet')
    
    #take subsample of 600sec (10min)
    spec= spec.loc[(spec.time>=specoffset) & (spec.time<specoffset+600)]
    spec = spec.loc[(spec.time>specoffset+300-5) & (spec.time<=specoffset+300+5)]
    spec = spec.fillna(0)

    #adpat dataset
    spec=spec.set_index('time')
    spec=spec.T
    spec['column']=spec.index.str.split('_', expand=True)

    spec['freq'] = spec.column.apply(lambda x: x[1]).astype(float)
    spec['brainreg'] = spec.column.apply(lambda x: x[0]).astype(str)

    spec=spec.drop('column', axis=1)
    spec.set_index('freq',inplace=True)
    #generate subdatases from brain zones
    subspec=dict()
    for zone in spec_zones:
        subspec[f'{zone}_sub']=spec[spec.brainreg==zone]
        subspec[f'{zone}_sub']= subspec[f'{zone}_sub'].drop('brainreg', axis=1)

    # Genera il grafico dello spettrogramma con dimensioni specificate
    fig, ax = plt.subplots(nrows=len(spec_zones), figsize=(6.4, 6.4), sharex=True)  # 256x256 pixel a 100 dpi
    for row in range(len(spec_zones)):
        data=subspec[f'{spec_zones[row]}_sub']
        ax[row].imshow(data, cmap='turbo',
                       aspect='auto',
                       origin='lower',
                       extent=[data.columns.min(),data.columns.max(),data.index.min(),data.index.max()],
                      vmin=0,vmax=data.max().max()*output_filter)

        ax[row].set_xticks([])
        ax[row].set_yticks([])

    plt.subplots_adjust(hspace=0.01)

    # Salva l'immagine in formato .png
    save_path= f'{spec_out}{labelid}.png'
    fig.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()

    return save_path


if not os.path.exists(spectr_windows_path):
    os.makedirs(spectr_windows_path)


args_list = [(row["spectrogram_id"], row["spectrogram_label_offset_seconds"], row["label_id"]) for index, row in dataset_df.iterrows()]

with concurrent.futures.ProcessPoolExecutor() as executor:
    windows = list(tqdm(executor.map(generate_spectrogram, *zip(*args_list)), total=len(args_list)))
