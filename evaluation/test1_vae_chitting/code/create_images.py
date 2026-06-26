import argparse
import json
import os
import tempfile
import numpy as np
import subprocess
from tqdm import tqdm
import pandas as pd
import datetime
import time
import gc
import random
import sys
import shutil
import glob

sys.path.append("/home/agustin/phd/synthesis")
sys.path.append("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils")

# pytorch
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
import torch.nn.functional as F

# data loader
from sklearn.preprocessing import MinMaxScaler

# mine
import utils.nifti_functions as nfc
import utils.util as util
import utils.functions as fc
import utils.util_freesurfer_segmentation as ufs
import utils.gpu_selector as gpu_selector
import data_loaders.load_dataset as load_dataset
import utils.data_normalization as data_normalization

import prep_image as prep_image

# monai
from monai.bundle import ConfigParser
from monai.networks.utils import copy_model_state

# images
from PIL import Image

sys.path.append("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils")
from autoencoder_declaration import AutoencoderPrediction

device_name = f"cuda:3"
device = torch.device(device_name)


def set_seed(seed: int):
    # random.seed(seed)  # Semilla para Python
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed(seed)  # Semilla para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
    torch.backends.cudnn.deterministic = True  # Garantizar reproducibilidad en CNNs
    torch.backends.cudnn.benchmark = False  # Desactivar optimización no determinista



@torch.no_grad()
def validation(
    src_image,
    autoencoder,
):

    latents = autoencoder.encode(src_image, seed=42)
    synthetic_images = autoencoder.decode(
            latents, decode_complete=True, sliding_window_size=(64, 64, 64)
        )

    # clip the values to the range [0, 1]
    synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
    synthetic_images = synthetic_images.squeeze().numpy()
    return synthetic_images




def evaluate(
    df_val,
    base_path,
    autoencoder,
    # normalize_wm=True,
):
    # base_path = os.path.join(base_path)
    # if normalize_wm:
    #     base_path += "_wm_normalized"

    bar = tqdm(
        df_val.iterrows(),
        total=len(df_val),
        desc="Generating reconstructed images",
    )
    # filter
    for i, row in df_val.iterrows():

        modality = row["modality"]
        resolution = row["resolution"]
        iid = row["iid"]

        if resolution not in [0.1, 1.5]:  # float naming problem
            resolution = int(resolution)
        save_path = os.path.join(base_path, modality, str(resolution), "pred")
        save_name = f"{iid}.nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        # verify if the image already exists, if it does, skip it
        if os.path.exists(save_path_name):
            print(f"Image {save_path_name} already exists, skipping...")
            bar.update(1)
            continue

        os.makedirs(save_path, exist_ok=True)

        src_img, org_aff = nfc.load_nifti(row["org_img_path"])

        org_shape = src_img.shape

        # if normalize_wm:
        #     src_seg, _ = nfc.load_nifti(row["seg_synthseg_path"])
        #     wm_mask = ufs.merge_seg96_to_mask(src_seg, [ufs.CEREBRAL_WM])
        #     src_img = prep_image.normalize_image_by_cerebral_wm_mean(src_img, wm_mask)

        src_img = fc.resize_center_crop_pad(src_img, (384, 448, 384), None)[0]
        src_img = util.robust_normalize(src_img, percentile=(0,100), strictly_positive=True)


        synthetic_image = validation(
            src_image=src_img,
            autoencoder=autoencoder
        )

        synthetic_image = prep_image.postprocess_img(
            synthetic_image, original_size=org_shape
        )
        nfc.save_nifti(synthetic_image, org_aff, save_path_name)
        bar.update(1)

        # break

    bar.close()





def evaluate_task1(
    df_val,
    df_paired_train,
    base_path,
    autoencoder
):
    base_path = os.path.join(base_path, "task1")

    # remove 7 test images if they already exist
    df_val_task1 = df_val[df_val["resolution"] != 7]

    bar = tqdm(
        df_val_task1.iterrows(),
        total=len(df_val_task1),
        desc="Generating synthetic images for Task 1",
    )
    # filter
    for i, row in df_val_task1.iterrows():

        iid = row["iid"]
        int_id = row["int_id"]
        modality = row["modality"]
        resolution = row["resolution"]


        if resolution not in [0.1, 1.5]:  # float naming problem
            resolution = int(resolution)

        save_path = os.path.join(base_path, modality, f"{resolution}T_to_7T", "pred")
        save_name = iid.replace(f"{resolution}T", f"{7}T") + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        bar.set_description(
            f"Generating synthetic images for Task 1 - sid: {row['sid']} Modality {modality} Resolution {resolution}T"
        )

        # verify if the image already exists, if it does, skip it
        if os.path.exists(save_path_name):
            print(f"Image {save_path_name} already exists, skipping...")
            bar.update(1)
            continue

        os.makedirs(save_path, exist_ok=True)

        # find the corresponding paired image in df_paired_train
        paired_row = df_paired_train[(df_paired_train["int_id"] == int_id) & (df_paired_train["modality"] == modality) & (df_paired_train["resolution"] == 7)].iloc[0]


        src_img, org_aff = nfc.load_nifti(paired_row["org_img_path"])

        org_shape = src_img.shape

        src_img = fc.resize_center_crop_pad(src_img, (384, 448, 384), None)[0]
        src_img = util.robust_normalize(src_img, percentile=(0,100), strictly_positive=True)


        synthetic_image = validation(
            src_image=src_img,
            autoencoder=autoencoder
        )

        synthetic_image = prep_image.postprocess_img(
            synthetic_image, original_size=org_shape
        )
        nfc.save_nifti(synthetic_image, org_aff, save_path_name)
        bar.update(1)

        # break

    bar.close()




if __name__ == "__main__":
    df_val = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv")
    df_paired_train = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/paired_train_data.csv")

    # obtain the columns iid, split by _, remove the first character and convert the remaining string to an integer, and save it in a new column called int_id
    df_val["int_id"] = df_val["sid"].apply(lambda x: int(x[1:]))
    df_paired_train["int_id"] = df_paired_train["iid"].apply(lambda x: int(x.split("_")[0][1:])) 



    # df_val = df_val[df_val["split"] == "val"]
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test1_vae_chitting/results"

    # df_val = df_val[(df_val["modality"] == "T1W") & (df_val["resolution"] == 3)]
    autoencoder_chekpoint_path = "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_vae/vae_weights/autoencoder_epoch273.pt"
    autoencoder = AutoencoderPrediction(autoencoder_chekpoint_path, device, half=True)
    
    # evaluate(
    #     df_val=df_val,
    #     base_path=output_path,
    #     autoencoder=autoencoder,
    #     # normalize_wm=False # is the same True or false (removed by 0-1 normalization)
    # )
    
    evaluate_task1(
        df_val=df_val,
        df_paired_train=df_paired_train,
        base_path=output_path,
        autoencoder=autoencoder
    )
