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


sys.path.append('/home/agustin/phd/synthesis')
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')

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
import prep_segmentation as prep_segmentation
import prep_vol2vol as prep_vol2vol 

# monai
from monai.bundle import ConfigParser
from monai.networks.utils import copy_model_state

import  networks_declaration.diffusion_model_unet_maisi_mask_seg as diffusion_model_unet_maisi
import  networks_declaration.conditions_model as conditions_model
import  networks_declaration.controlnet_maisi as controlnet_maisi


from networks_declaration.rectified_flow import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
# images
from PIL import Image

sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')
from autoencoder_declaration import AutoencoderPrediction




def create_df_task1(pred_path_name, modalitites, resolutions):
    data = []
    for modality in modalitites:
        for resolution in resolutions:
            pred_path = os.path.join(pred_path_name, modality, f"{resolution}T_to_7T", "pred")
            pred_files = glob.glob(os.path.join(pred_path, "*.nii.gz"))
            for pred_file in pred_files:
                iid = os.path.basename(pred_file).replace(".nii.gz", "")
                sid = iid.split("_")[-1]
                # org_img_path = pred_file.replace("pred", "org").replace(f"{resolution}T_to_7T", f"{resolution}T")
                # latent_seg_supersynth_path = org_img_path.replace("org", "latent_seg_supersynth").replace(".nii.gz", ".npy")
                data.append({
                    "sid": sid,
                    "iid": iid,
                    "modality": modality,
                    "resolution": resolution,
                    "pred_img_path": pred_file,
                })
    df = pd.DataFrame(data)
    return df



def segment_task1(pred_path_name):
    pred_path_name = os.path.join(pred_path_name, "task1")

    modalitites = ["T1W", "T2W", "T2FLAIR"]
    resolutions = [0.1, 1.5, 3, 5]

    df_val_task1 = create_df(pred_path_name, modalitites, resolutions)

    bar = tqdm(df_val_task1.iterrows(), total=len(df_val_task1), desc="Segmenting Task 1")
    # filter 
    for i, row in df_val_task1.iterrows():    

        modality = row["modality"]
        resolution = row["resolution"]
        iid = row["iid"]
        pred_img_path = row["pred_img_path"]

        if resolution not in [0.1, 1.5]:
            resolution = int(resolution)
        save_path = os.path.join(pred_path_name, modality, f"{resolution}T_to_7T", "seg")
        save_name = iid.replace(f"{resolution}T", f"{7}T") + "_seg" + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        bar.set_description(f"Segmenting Task 1 - sid: {row['sid']} Modality {modality} = Resolution {resolution}T")


        if os.path.exists(save_path_name):
            bar.update(1)  
            continue

        non_resampled_path_name = save_path_name.replace(".nii.gz", "_non_resampled.nii.gz")
        prep_segmentation.save_synthseg_segmentation(
            pred_img_path,
            non_resampled_path_name,
            verify=True,
            verbose=False,
            cortical_parcelation=False
        )
        # resampled_path_name = save_path_name.replace(".nii.gz", "_seg.nii.gz")
        prep_vol2vol.apply_vol2vol(
            pred_img_path,
            non_resampled_path_name,
            save_path_name,
            verify=True,
            verbose=False,
            nearest=True
        )

        # remove the intermediate segmentation file
        try:
            os.remove(non_resampled_path_name)
        except Exception as e:            
            pass
    
        bar.update(1)  

    bar.close()
    
    





def create_df_task2(pred_path_name, modalitites, resolutions):
    data = []
    for modality in modalitites:
        for resolution in resolutions:
            pred_path = os.path.join(pred_path_name, modality, f"0.1T_to_{resolution}T", "pred")
            pred_files = glob.glob(os.path.join(pred_path, "*.nii.gz"))
            for pred_file in pred_files:
                iid = os.path.basename(pred_file).replace(".nii.gz", "")
                sid = iid.split("_")[-1]
                # org_img_path = pred_file.replace("pred", "org").replace(f"{resolution}T_to_7T", f"{resolution}T")
                # latent_seg_supersynth_path = org_img_path.replace("org", "latent_seg_supersynth").replace(".nii.gz", ".npy")
                data.append({
                    "sid": sid,
                    "iid": iid,
                    "modality": modality,
                    "resolution": resolution,
                    "pred_img_path": pred_file,
                })
    df = pd.DataFrame(data)
    return df



def segment_task2(pred_path_name):
    pred_path_name = os.path.join(pred_path_name, "task2")

    modalitites = ["T1W", "T2W", "T2FLAIR"]
    resolutions = [3, 5, 7]

    df_val_task2 = create_df_task2(pred_path_name, modalitites, resolutions)

    bar = tqdm(df_val_task2.iterrows(), total=len(df_val_task2), desc="Segmenting Task 2")
    # filter 
    for i, row in df_val_task2.iterrows():    

        modality = row["modality"]
        resolution = row["resolution"]
        iid = row["iid"]
        pred_img_path = row["pred_img_path"]

        if resolution not in [0.1, 1.5]:
            resolution = int(resolution)
        save_path = os.path.join(pred_path_name, modality, f"0.1T_to_{resolution}T", "seg")
        save_name = iid.replace(f"0.1T", f"{resolution}T") + "_seg" + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        bar.set_description(f"Segmenting Task 2 - sid: {row['sid']} Modality {modality} = Resolution {resolution}T")


        if os.path.exists(save_path_name):
            bar.update(1)  
            continue

        non_resampled_path_name = save_path_name.replace(".nii.gz", "_non_resampled.nii.gz")
        prep_segmentation.save_synthseg_segmentation(
            pred_img_path,
            non_resampled_path_name,
            verify=True,
            verbose=False,
            cortical_parcelation=False
        )
        # resampled_path_name = save_path_name.replace(".nii.gz", "_seg.nii.gz")
        prep_vol2vol.apply_vol2vol(
            pred_img_path,
            non_resampled_path_name,
            save_path_name,
            verify=True,
            verbose=False,
            nearest=True
        )

        # remove the intermediate segmentation file
        try:
            os.remove(non_resampled_path_name)
        except Exception as e:            
            pass
    
        bar.update(1)  

    bar.close()
    



if __name__ == "__main__":
    # output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results"
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results"

    use_controlnet = False
    n_inference_steps = 30

    used_chk = 200000
    # used_chk = 210000
    # controlnet_chk = 95000

    network_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated/test1/check_points/model_{used_chk}.pt"

    if not use_controlnet:
        output_path = os.path.join(output_path, f"basic", f"chk_{used_chk}_steps_{n_inference_steps}")
    else:
        controlnet_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated_controlnet2/check_points/model_{controlnet_chk}.pt"
        output_path = os.path.join(output_path, f"controlnet", f"chk_{used_chk}_cnchk_{controlnet_chk}_steps_{n_inference_steps}")
        # raise NotImplementedError("ControlNet is not implemented yet for the evaluation, but it will be in the future. For now, just set use_controlnet to False if you want to run the evaluation.")

    # segment_task1(output_path)
    segment_task2(output_path)



