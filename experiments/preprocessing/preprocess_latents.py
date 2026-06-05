
import os
import sys
sys.path.append('/home/agustin/phd/synthesis')
sys.path.append('/home/agustin/phd/synthesis/tests/D3/preprocessing/brainst_preprocessing_pipeline/preprocessing')

import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util as util

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# import prep_images as prep_images
from autoencoder_declaration import AutoencoderPrediction


device_name = f"cuda:2"
device = torch.device(device_name)



def preprocess_latents(split="train"):
    autoencoder_chk_path = "/home/agustin/phd/BrainST/models/autoencoder/weights/autoencoder_epoch273.pt"
    half = True
    autoencoder = AutoencoderPrediction(autoencoder_chk_path, device, half=half)

    # read the csv file created by create_csv.py
    # csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/validation_data.csv"
    csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"
    df = pd.read_csv(csv_path)#.head(1)

    base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/maisi_latents"
    # base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/supersynth"
    bar = tqdm(total=df.shape[0], desc="Preprocessing images with mri_super_synth")
    
    latent_paths = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        subject_id = row['iid']
        resolution = row['resolution']
        modality = row['modality']
        # split = row['split']
        img_path = row['org_img_path']


        # print(type(subject_id), type(resolution), type(modality), type(split), type(img_path))
        output_path = os.path.join(base_output_path, modality, f"{str(resolution)}T")
        # output_latent_name = f"{modality}_{resolution}T_{subject_id[1:]}_latent.npy"
        output_latent_name = f"{subject_id}_latent.npy"
        output_latent_path_name = os.path.join(output_path, output_latent_name)
        os.makedirs(output_path, exist_ok=True)

        latent_paths.append(output_latent_path_name)

        # verify if latent already exists        
        if os.path.exists(output_latent_path_name):
            print(f"Skipping {subject_id} {resolution}T {modality}, latent already exists")
            continue

        img, aff =  nfc.load_nifti(img_path)
        img, new_aff, aff = fc.resize_center_crop_pad(img, (384, 448, 384), aff)
        img = util.robust_normalize(img, percentile=(0,100), strictly_positive=True)

        latent = autoencoder.encode(img)

        # save latents
        np.save(output_latent_path_name, latent.cpu().squeeze().numpy())


        bar.update(1)
    bar.close()
    
    # save the dataframe to a new csv file
    df["latent_path"] = latent_paths
    output_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data_with_latents.csv"
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    # preprocess_latents(split="train")
    preprocess_latents(split="val")
    # preprocess_latents(split="pr_train")
