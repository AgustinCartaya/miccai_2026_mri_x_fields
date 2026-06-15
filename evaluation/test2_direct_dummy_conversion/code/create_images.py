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

sys.path.append(
    "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test2_direct_dummy_conversion/training/networks_declaration"
)

from monai.bundle import ConfigParser
import diffusion_model_unet_maisi as diffusion_model_unet_maisi
from autoencoder_declaration import AutoencoderPrediction

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


def instantiate_unconditioned_models(device, dm_chk_path, used_resolutions, used_networks="diffusion_unet_def"):

    networks_config = {
        # "autoencoder_def": {
        #     "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
        #     "spatial_dims": 3,
        #     "in_channels": 1,
        #     "out_channels": 1,
        #     "latent_channels": 4,
        #     "num_channels": [
        #         64,
        #         128,
        #         256
        #     ],
        #     "num_res_blocks": [2,2,2],
        #     "norm_num_groups": 32,
        #     "norm_eps": 1e-06,
        #     "attention_levels": [
        #         False,
        #         False,
        #         False
        #     ],
        #     "with_encoder_nonlocal_attn": False,
        #     "with_decoder_nonlocal_attn": False,
        #     "use_checkpointing": False,
        #     "use_convtranspose": False,
        #     "norm_float16": True,
        #     "num_splits": 8,
        #     "dim_split": 1
        # },
        "diffusion_unet_def_t1w": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": 2,
            "num_channels": [64, 128, 256, 512],
            "attention_levels": [False, False, True, True],
            "num_head_channels": [0, 0, 32, 32],
            "use_flash_attention": True,
            "with_conditioning": False,
            "nb_modalities": used_resolutions,
        },
        
        "diffusion_unet_def": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": 2,
            "num_channels": [
                # 64,
                # 128,
                # 256,
                # 512
                32,
                64,
                128,
            ],
            "attention_levels": [
                False,
                False,
                True,
                # True,
                # True
            ],
            "num_head_channels": [
                0,
                0,
                32,
                # 32
            ],
            "use_flash_attention": True,
            "with_conditioning": False,
            "nb_modalities": used_resolutions,
        },
    }

    # instantiate model
    parser = ConfigParser(networks_config)
    parser.parse(True)

    args = fc.dict_to_args(networks_config, deep_conversion=True)

    # unet
    unet = diffusion_model_unet_maisi.DiffusionModelUNetMaisi(
        spatial_dims=args.diffusion_unet_def.spatial_dims,
        in_channels=args.diffusion_unet_def.in_channels,
        out_channels=args.diffusion_unet_def.out_channels,
        num_res_blocks=args.diffusion_unet_def.num_res_blocks,
        num_channels=args.diffusion_unet_def.num_channels,
        attention_levels=args.diffusion_unet_def.attention_levels,
        #    norm_num_groups = args.diffusion_unet_def.norm_num_groups,
        #    norm_eps = args.diffusion_unet_def.norm_eps,
        #    resblock_updown = args.diffusion_unet_def.resblock_updown,
        num_head_channels=args.diffusion_unet_def.num_head_channels,
        with_conditioning=args.diffusion_unet_def.with_conditioning,
        use_flash_attention=args.diffusion_unet_def.use_flash_attention,
        nb_modalities=args.diffusion_unet_def.nb_modalities,
    )

    # autoencoder (just for validation)
    # autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    autoencoder_chekpoint_path = "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_vae/vae_weights/autoencoder_epoch273.pt"
    # checkpoint_autoencoder = torch.load(autoencoder_chekpoint_path, weights_only=True, map_location=device)
    # autoencoder.load_state_dict(checkpoint_autoencoder)
    # autoencoder.eval()

    autoencoder = AutoencoderPrediction(autoencoder_chekpoint_path, device, half=True)

    # load unet checkpoint
    checkpoint_unet = torch.load(dm_chk_path, map_location=device)
    unet.load_state_dict(checkpoint_unet["unet_state_dict"])
    unet.to(device).eval()

    return {
        "unet": unet,
        "autoencoder": autoencoder,
        "networks_config": args,
    }


@torch.no_grad()
def validation(
    src_latent,
    src_resolution_idx,
    unet,
    autoencoder,
    x_valid_resolutions,
):

    src_latents = src_latent.to(device)

    # create one hot encoding for the missing modality
    resolution_one_hot = torch.zeros((src_latents.shape[0], x_valid_resolutions))
    resolution_one_hot[torch.arange(src_latents.shape[0]), src_resolution_idx] = 1.0
    resolution_one_hot = resolution_one_hot.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        latents = unet(src_latents, modality_tensor=resolution_one_hot)

        # decode the latents to images
        synthetic_images = autoencoder.decode(
            latents, decode_complete=False, sliding_window_size=(64, 64, 64)
        )
        synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
        synthetic_images = synthetic_images.squeeze().numpy()

    return synthetic_images


def evaluate_task3(
    df_val,
    base_path,
    models,
    tar_resolution,
    valid_resolutions_list,
):
    base_path = os.path.join(base_path, "task3")
    df_val_task3 = df_val.copy()

    bar = tqdm(
        df_val_task3.iterrows(),
        total=len(df_val_task3),
        desc="Generating synthetic images for Task 3",
    )
    # filter
    for i, row in df_val_task3.iterrows():

        modality = row["modality"]
        src_resolution = row["resolution"]
        iid = row["iid"]
        # modality_idx = row["modality_idx"]
        src_resolution_idx = row["resolution_idx"]
        src_latent = row["latent_path"]

        if src_resolution not in [0.1, 1.5]:  # float naming problem
            src_resolution = int(src_resolution)

        if tar_resolution not in [0.1, 1.5]:  # float naming problem
            tar_resolution = int(tar_resolution)

        save_path = os.path.join(
            base_path, modality, f"{src_resolution}T_to_{tar_resolution}T", "pred"
        )
        save_name = iid.replace(f"{src_resolution}T", f"{tar_resolution}T") + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        bar.set_description(
            f"Generating synthetic images for Task 3 - sid: {row['sid']} Modality {modality} Resolution {src_resolution}T = {src_resolution_idx} "
        )

        # verify if the image already exists, if it does, skip it
        if os.path.exists(save_path_name):
            print(f"Image {save_path_name} already exists, skipping...")
            bar.update(1)
            continue

        os.makedirs(save_path, exist_ok=True)

        src_latent = (
            torch.from_numpy(np.load(src_latent)).float().unsqueeze(0)
        )  # add batch dimension
        synthetic_image = validation(
            src_latent=src_latent,
            src_resolution_idx=src_resolution_idx,
            unet=models["unet"],
            autoencoder=models["autoencoder"],
            x_valid_resolutions=len(valid_resolutions_list),
        )

        org_img, org_aff = nfc.load_nifti(row["org_img_path"])
        synthetic_image = prep_image.postprocess_img(
            synthetic_image, original_size=org_img.shape
        )
        nfc.save_nifti(synthetic_image, org_aff, save_path_name)
        bar.update(1)

        # break

    bar.close()


if __name__ == "__main__":
    df_val = pd.read_csv(
        "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv"
    )
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test2_direct_dummy_conversion/results/val"

    # df_val = pd.read_csv(
    #     "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"
    # )
    # df_val = df_val[df_val["split"] == "val"]
    # output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/train"

    to_resolution = 0.1
    used_modality = "T2FLAIR"
    dm_chk_number = 5000

    output_path = os.path.join(
        output_path, f"{used_modality}", f"to_{to_resolution}t", f"chk_{dm_chk_number}"
    )

    dm_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test2_direct_dummy_conversion/training/models/to_{to_resolution}t/test1_{used_modality}/check_points/model_{dm_chk_number}.pt"

    possible_resolutions = [0.1, 1.5, 3, 5, 7]  # 0.1, 1.5, 3, 5, 7

    # find the index of the target resolution in the possible resolutions list
    to_resolution_idxx = possible_resolutions.index(to_resolution)
    # remove the target resolution and all previous resolutions from the possible resolutions list, since they are not valid target resolutions
    valid_resolutions_list = possible_resolutions[to_resolution_idxx + 1 :]

    df_val = df_val[df_val["resolution"].isin(valid_resolutions_list) & (df_val["modality"] == used_modality)].reset_index(drop=True)

    resolution_idx_mapping = {
        resolution: idx for idx, resolution in enumerate(valid_resolutions_list)
    }
    # map modality to index
    df_val["resolution_idx"] = df_val["resolution"].map(resolution_idx_mapping)

    models_dict = instantiate_unconditioned_models(
        device,
        dm_chk_path=dm_chk_path,
        used_resolutions=len(valid_resolutions_list),
    )

    evaluate_task3(
        df_val,
        output_path,
        models=models_dict,
        tar_resolution=to_resolution,
        valid_resolutions_list=valid_resolutions_list,
    )
