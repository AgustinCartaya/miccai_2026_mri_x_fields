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



def instantiate_unconditioned_models(
    device,
    dm_chk_path,
    dm_seg_channels=3,
    dm_num_modalitites=3,
    dm_num_resolutions=3,
    dm_noise_scheduler_type="rflow",
    num_inference_steps=30,
):

    networks_config = {
        "diffusion_unet_def": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4 + 4 + dm_seg_channels,
            "out_channels": 4,
            "num_res_blocks": 2,
            "num_channels": [
                # 32,
                # 64,
                # 128,
                # 256,
                64,
                128,
                256,
                512,
            ],
            "self_attention_levels": [False, False, True, True],
            "num_self_head_channels": [0, 0, 16, 16],
            "cross_attention_levels": [False, False, False, False],
            "num_cross_head_channels": [0, 0, 0, 0],
            "use_flash_attention": True,
            "with_conditioning": False,
            "cross_attention_dim": None,
            "transformer_num_layers": 1,  # number of transformer blocks
            "upcast_attention": True,
        },

        "modality_encoder_def": {  # this is volumetric conditioning
            "num_conditions": dm_num_modalitites,  # number of conditions
            # "embed_dim": 512,  # this will be automatically set to be the same as the unet embedding dimension
        },
        "resolution_encoder_def": {  # this is volumetric conditioning
            "num_conditions": dm_num_resolutions,  # number of conditions
            # "embed_dim": 512,  # this will be automatically set to be the same as the unet embedding dimension
        },

        # "controlnet_def": {
        #     "_target_": "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
        #     "spatial_dims": 3,
        #     "in_channels": 4,  # this is the nosy latent space
        #     "num_channels": [64, 128, 256, 512],
        #     "attention_levels": [False, False, True, True],
        #     "num_head_channels": [0, 0, 16, 16],
        #     "num_res_blocks": 2,
        #     "use_flash_attention": True,
        #     "conditioning_embedding_in_channels": 4,  # this is the condition image (e.g. low res/0.1T)
        #     "conditioning_embedding_num_channels": [2, 4, 8],  # [8, 32, 64],
        #     "num_class_embeds": cnet_num_resolutions,  # this is the number of modalities that we have as conditions
        # },


        "noise_scheduler": {
            "_target_": "monai.networks.schedulers.DDIMScheduler",  # faster scheduler
            "beta_start": 0.0015,
            "beta_end": 0.0205,
            "num_train_timesteps": 1000,
            "schedule": "scaled_linear_beta",
            "clip_sample": False,
        },
        "noise_scheduler_rf": {
            "_target_": "monai.networks.schedulers.rectified_flow.RFlowScheduler",
            "num_train_timesteps": 1000,
            "use_discrete_timesteps": False,
            "use_timestep_transform": True,
            "sample_method": "uniform",
            "scale": 1.4,
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
        # attention_levels = args.diffusion_unet_def.attention_levels,
        self_attention_levels=args.diffusion_unet_def.self_attention_levels,
        cross_attention_levels=args.diffusion_unet_def.cross_attention_levels,
        # num_head_channels = args.diffusion_unet_def.num_head_channels,
        num_self_head_channels=args.diffusion_unet_def.num_self_head_channels,
        num_cross_head_channels=args.diffusion_unet_def.num_cross_head_channels,
        with_conditioning=args.diffusion_unet_def.with_conditioning,
        transformer_num_layers=args.diffusion_unet_def.transformer_num_layers,
        cross_attention_dim=args.diffusion_unet_def.cross_attention_dim,
        upcast_attention=args.diffusion_unet_def.upcast_attention,
        use_flash_attention=args.diffusion_unet_def.use_flash_attention,
        include_top_region_index_input=False,
        include_bottom_region_index_input=False,
        include_spacing_input=False,
    )

    modality_encoder_model = conditions_model.SimpleConditionEmbedding(
        num_conditions=args.modality_encoder_def.num_conditions,
        embed_dim=unet.new_time_embed_dim,
    )

    resolution_encoder_model = conditions_model.SimpleConditionEmbedding(
        num_conditions=args.resolution_encoder_def.num_conditions,
        embed_dim=unet.new_time_embed_dim,
    )

    # controlnet_model = controlnet_maisi.ControlNetMaisi(
    #     spatial_dims=args.controlnet_def.spatial_dims,
    #     in_channels=args.controlnet_def.in_channels,
    #     num_res_blocks=args.controlnet_def.num_res_blocks,
    #     num_channels=args.controlnet_def.num_channels,
    #     attention_levels=args.controlnet_def.attention_levels,
    #     num_head_channels=args.controlnet_def.num_head_channels,
    #     use_flash_attention=args.controlnet_def.use_flash_attention,
    #     conditioning_embedding_in_channels=args.controlnet_def.conditioning_embedding_in_channels,
    #     conditioning_embedding_num_channels=args.controlnet_def.conditioning_embedding_num_channels,
    #     num_class_embeds=None,
    # )
    # ### HERE


    # noise scheduler
    if dm_noise_scheduler_type == "ddim":
        noise_scheduler = parser.get_parsed_content("noise_scheduler", instantiate=True)
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    elif dm_noise_scheduler_type == "rflow":
        # noise_scheduler = parser.get_parsed_content("noise_scheduler_rf", instantiate=True)
        noise_scheduler = RFlowScheduler(
            num_train_timesteps=args.noise_scheduler_rf.num_train_timesteps,
            use_discrete_timesteps=args.noise_scheduler_rf.use_discrete_timesteps,
            use_timestep_transform=args.noise_scheduler_rf.use_timestep_transform,
            sample_method=args.noise_scheduler_rf.sample_method,
            scale=args.noise_scheduler_rf.scale,
        )
        noise_scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            input_img_size_numel=torch.prod(torch.tensor((48, 64, 48))),
        )

    autoencoder_chekpoint_path = "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_vae/vae_weights/autoencoder_epoch273.pt"
    autoencoder = AutoencoderPrediction(autoencoder_chekpoint_path, device, half=True)
    # Load pretrained model weights
    unet_checkpoint = torch.load(dm_chk_path, map_location=device_name)
    unet.load_state_dict(unet_checkpoint["ema_state_dict"], strict=False)
    # segmentation_encoder_model.load_state_dict(unet_checkpoint["segmentation_encoder_model_state_dict"], strict=False)
    modality_encoder_model.load_state_dict(
        unet_checkpoint["modality_encoder_model_state_dict"], strict=False
    )
    resolution_encoder_model.load_state_dict(
        unet_checkpoint["resolution_encoder_model_state_dict"], strict=False
    )

    unet.to(device).eval()
    modality_encoder_model.to(device).eval()
    resolution_encoder_model.to(device).eval()

    # if cnet_chk_path is not None:
    #     controlnet_checkpoint = torch.load(cnet_chk_path, map_location=device_name)
    #     controlnet_model.load_state_dict(
    #         controlnet_checkpoint["controlnet_state_dict"], strict=False
    #     )
    #     controlnet_model.to(device).eval()
    # # controlnet_model.to(device).eval()
    return {
        "unet": unet,
        "autoencoder": autoencoder,
        "modality_encoder_model": modality_encoder_model,
        "resolution_encoder_model": resolution_encoder_model,
        # "controlnet_model": controlnet_model,
        "noise_scheduler": noise_scheduler,
        "networks_config": args,
    }




@torch.no_grad()
def validation(
    src_latent_mask,
    src_latent_synthsr,
    modality_idx,
    tar_resolution_idx,
    unet,
    noise_scheduler,
    modality_encoder_model,
    resolution_encoder_model,
    autoencoder,
    seed=42, 
    decode=True
):

    latents_shape = src_latent_mask.shape

    # instantiate every time to generate using the same initial noise (using CPU generator)
    _l_shape = [1, 4, latents_shape[-3], latents_shape[-2], latents_shape[-1]]
    gen_randn = torch.Generator().manual_seed(seed) 
    latents = torch.randn(_l_shape, generator=gen_randn).half().to(device)

    src_segmentation = src_latent_mask.to(device)
    if src_latent_synthsr is not None:
        latents_synthsr = src_latent_synthsr.to(device)
    else:
        latents_synthsr = torch.zeros_like(latents).to(device)

    modality_idx = modality_idx.to(device)
    tar_resolution_idx = tar_resolution_idx.to(device)

    modality_embedding = modality_encoder_model(modality_idx)
    tar_resolution_embedding = resolution_encoder_model(tar_resolution_idx)


    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
    )
    with torch.no_grad(), torch.amp.autocast("cuda"):

        for t, next_t in progress_bar:
            # print(f"latents.shape: {latents.shape}, src_segmentation.shape: {src_segmentation.shape}, latents_synthsr.shape: {latents_synthsr.shape}")
            timesteps= torch.Tensor((t,)).to(device)
            model_output = unet(
                x=torch.cat([latents, src_segmentation, latents_synthsr], dim=1),
                timesteps=timesteps,
                modallity_embedding = modality_embedding,
                resolution_embedding = tar_resolution_embedding,
            )

            if not isinstance(noise_scheduler, RFlowScheduler):
                latents, _ = noise_scheduler.step(model_output, t, latents)
            else:
                latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore

        # free memory for the autoencoder
        del model_output
        torch.cuda.empty_cache()

        # decode the latents to images
        if decode:
            synthetic_images = autoencoder.decode(latents, decode_complete=True, sliding_window_size=(64, 64, 64))
            synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
            synthetic_images = synthetic_images.squeeze().numpy()
        else:
            synthetic_images = None

        latents = latents.squeeze().cpu().numpy().astype(np.float32)

    return {"image": synthetic_images, "latent": latents}


def load_segmentation(segmentation_npy_path, nb_classes=3):
    segmentation = np.load(segmentation_npy_path)
    # convert into one-hot encoding
    # unique_labels = np.unique(segmentation)
    unique_labels = np.arange(1, nb_classes+1)
    # remove zero if it is in the unique labels (assuming zero is the background)
    if 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]

    seg_onehot = []     
    for label in unique_labels:
        seg_onehot.append(np.where(segmentation == label, 1.0, 0.0))   
    seg_onehot = np.stack(seg_onehot, axis=0)  # (C, D, H, W)
    
    return torch.from_numpy(seg_onehot)


def get_best_subject_latent_masks(sdf, nb_classes, modalities=["T1W", "T2W", "T2FLAIR"], prefer_t1=True):
    # find the higest resolution available for each modality, and return the corresponding row
    # best_rows = {}
    best_latent_mask = {}
    for modality in modalities:
        modality_rows = sdf[sdf["modality"] == modality]
        if modality_rows.empty:
            # best_rows[modality] = None
            best_latent_mask[modality] = None
        else:
            best_row = modality_rows.loc[modality_rows["resolution"].idxmax()]
            # best_rows[modality] = best_row
            # _latent = np.load(best_row["latent_mask_path"])
            _latent = load_segmentation(best_row[f"latent_seg_supersynth_merged_{nb_classes}_path"], nb_classes=nb_classes)  # (C, D, H, W)
            best_latent_mask[modality] = _latent
            print(f"Seg: sid {sdf['sid'].iloc[0]} modality {modality} resolution {best_row['resolution']} latent synthsr shape {_latent.shape}")
    
    # for those with no available modality, we take the mean of the available ones
    for modality in modalities:
        if prefer_t1 and best_latent_mask["T1W"] is not None:
            best_latent_mask[modality] = best_latent_mask["T1W"]
            print(f"Using T1W latent synthsr for modality {modality} for subject {sdf['sid'].iloc[0]}")
        elif best_latent_mask[modality] is None:
            available_latent_masks = [best_latent_mask[m] for m in modalities if best_latent_mask[m] is not None]
            if len(available_latent_masks) > 0:
                mean_latent_mask = torch.mean(torch.stack(available_latent_masks), dim=0)
                best_latent_mask[modality] = mean_latent_mask
            else:
                raise ValueError(f"No available modality for subject {sdf['sid'].iloc[0]}")
    return best_latent_mask


def get_best_subject_synthsr(sdf, modalities=["T1W", "T2W", "T2FLAIR"], prefer_t1=True):
    # find the higest resolution available for each modality, and return the corresponding row
    # best_rows = {}
    best_latent_mask = {}
    for modality in modalities:
        modality_rows = sdf[sdf["modality"] == modality]
        if modality_rows.empty:
            best_latent_mask[modality] = None
        else:
            best_row = modality_rows.loc[modality_rows["resolution"].idxmax()]
            _latent = np.load(best_row[f"latent_synthsr_path"])  # (C, D, H, W)
            best_latent_mask[modality] = _latent
            print(f"Synthsr: sid {sdf['sid'].iloc[0]} modality {modality} resolution {best_row['resolution']} latent synthsr shape {_latent.shape}")
    
    # for those with no available modality, we take the mean of the available ones
    for modality in modalities:
        if prefer_t1 and best_latent_mask["T1W"] is not None:
            best_latent_mask[modality] = best_latent_mask["T1W"]
            print(f"Using T1W latent synthsr for modality {modality} for subject {sdf['sid'].iloc[0]}")
        elif best_latent_mask[modality] is None:
            available_latent_masks = [best_latent_mask[m] for m in modalities if best_latent_mask[m] is not None]
            if len(available_latent_masks) > 0:
                mean_latent_mask = torch.mean(torch.stack(available_latent_masks), dim=0)
                best_latent_mask[modality] = mean_latent_mask
            else:
                raise ValueError(f"No available modality for subject {sdf['sid'].iloc[0]}")
    return best_latent_mask



def generate_identity_dataset(dataset_df, output_path, models,nb_classes):
    # used_modalities = ["T1W", "T2W", "T2FLAIR"] # "T1W", "T2W", "T2FLAIR"
    # used_resolutions = [0.1, 1.5, 3, 5, 7] #0.1, 1.5, 3, 5, 7
    # os.makedirs(output_path, exist_ok=True)

    generated_paths = []
    org_img, org_aff = nfc.load_nifti(dataset_df["org_img_path"].iloc[0])
    bar = tqdm(total=dataset_df.shape[0], desc="Generating images with SynthSR model")
    for i, row in dataset_df.iterrows():

        sid = row["sid"]
        iid = row["iid"]
        modality = row["modality"]
        resolution = row["resolution"]

        if resolution not in [0.1, 1.5]:  # float naming problem
            resolution = int(resolution)

        output_img_path_name = os.path.join(output_path, modality, f"{resolution}T", f"{iid}_gen.nii.gz")
        generated_paths.append(output_img_path_name)
        if os.path.exists(output_img_path_name):
            print(f"Skipping {sid} {modality} {resolution}T, image already exists")
            bar.update(1)
            continue

        
        modality_idx = row["modality_idx"]
        resolution_idx = row["resolution_idx"]

        segmentation_path = row[f"latent_seg_supersynth_merged_{nb_classes}_path"]
        latent_synthsr_path = row["latent_synthsr_path"]

        # change for T1W if available and resolution <= 1.5T
        if resolution <= 1.5 and modality != "T1W":
            # verify if T1W is available for this subject and resolution
            t1w_row = dataset_df[(dataset_df["sid"] == sid) & (dataset_df["modality"] == "T1W") & (dataset_df["resolution"] == resolution)]
            if not t1w_row.empty:
                print(f"Using T1W for {sid} {modality} {resolution}T")
                segmentation_path = t1w_row[f"latent_seg_supersynth_merged_{nb_classes}_path"].values[0]
                latent_synthsr_path = t1w_row["latent_synthsr_path"].values[0]

        segmentation = load_segmentation(segmentation_path, nb_classes=nb_classes).float().unsqueeze(0)
        latent_synthsr = torch.from_numpy(np.load(latent_synthsr_path)).float().unsqueeze(0)
        modality_idx = torch.tensor(modality_idx).unsqueeze(0)  # convert to tensor and add batch dimension
        resolution_idx = torch.tensor(resolution_idx).unsqueeze(0)  # convert to tensor and add batch dimension

        generation = validation(
            src_latent_mask=segmentation,
            src_latent_synthsr=latent_synthsr,
            modality_idx=modality_idx,
            tar_resolution_idx=resolution_idx,
            unet=models["unet"],
            noise_scheduler=models["noise_scheduler"],
            modality_encoder_model=models["modality_encoder_model"],
            resolution_encoder_model=models["resolution_encoder_model"],
            autoencoder=models["autoencoder"],
            decode=True
        )

        os.makedirs(os.path.dirname(output_img_path_name), exist_ok=True)
        synthetic_image = prep_image.postprocess_img(
            generation["image"], original_size=org_img.shape
        )
        nfc.save_nifti(synthetic_image, org_aff, output_img_path_name)
        
        # save latents as well
        output_latent_path_name = os.path.join(output_path, modality, f"{resolution}T", f"{iid}_gen_latent.npy")
        np.save(output_latent_path_name, generation["latent"])
        
        bar.update(1)
        # break

    return generated_paths

# def generate_all_dataset(dataset_df, output_path, models, used_modalities, used_resolutions, nb_classes, tar_resolution_offset=0):
#     # used_modalities = ["T1W", "T2W", "T2FLAIR"] # "T1W", "T2W", "T2FLAIR"
#     # used_resolutions = [0.1, 1.5, 3, 5, 7] #0.1, 1.5, 3, 5, 7
#     # os.makedirs(output_path, exist_ok=True)
#     sids = dataset_df["sid"].unique()
#     for sid in sids:
#         sid_df = dataset_df[dataset_df["sid"] == sid]
#         best_latent_mask = get_best_subject_latent_masks(sid_df, nb_classes=nb_classes, modalities=used_modalities, prefer_t1=True)
#         best_latent_synthsr = get_best_subject_synthsr(sid_df, modalities=used_modalities, prefer_t1=True)
#         org_img, org_aff = nfc.load_nifti(sid_df["org_img_path"].iloc[0])
        
#         for modality in used_modalities:
#             src_latent_mask = best_latent_mask[modality]
#             src_latent_synthsr = best_latent_synthsr[modality]
#             for resolution in used_resolutions:


#                 output_dir = os.path.join(output_path, modality, f"{resolution}T")
#                 output_name = f"GEN_{modality}_{resolution}T_{sid[1:]}_latent.npy"
#                 output_path_name = os.path.join(output_dir, output_name)
                
#                 if not os.path.exists(output_path_name):
#                     row = sid_df[(sid_df["modality"] == modality) & (sid_df["resolution"] == resolution)]

#                     # if row.empty: # this is for training
#                     if not row.empty: # this is for validation, we only generate images for the available modalities and resolutions
#                         # generate image
#                         # print(f"Generating image for SID {sid} Modality {modality} Resolution {resolution}")
#                         synthetic_gen = validation(
#                             src_latent_mask=src_latent_mask,
#                             modality_idx=torch.tensor(used_modalities.index(modality)),
#                             tar_resolution_idx=torch.tensor(used_resolutions.index(resolution)),
#                             unet=models["unet"],
#                             noise_scheduler=models["noise_scheduler"],
#                             modality_encoder_model=models["modality_encoder_model"],
#                             resolution_encoder_model=models["resolution_encoder_model"],
#                             autoencoder=models["autoencoder"],
#                             src_latent_synthsr=torch.from_numpy(src_latent_synthsr).unsqueeze(0).float(),
#                             controlnet_model=None,
#                             tar_resolution_offset=tar_resolution_offset,
#                             decode = True
#                         )

#                         # if resolution not in [0.1, 1.5]: # float naming problem
#                         #     resolution = int(resolution)


#                         os.makedirs(output_dir, exist_ok=True)
#                         np.save(output_path_name, synthetic_gen["latent"])

#                         # decode and save the image
#                         synthetic_image = prep_image.postprocess_img(
#                             synthetic_gen["image"], original_size=org_img.shape
#                         )
#                         output_image_name = f"GEN_{modality}_{resolution}T_{sid[1:]}_image.nii.gz"
#                         output_image_path_name = os.path.join(output_dir, output_image_name)
#                         nfc.save_nifti(synthetic_image, org_aff, output_image_path_name)


if __name__ == "__main__":
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/train_data/generated_synthsr"
    dataset_csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"

    dataset_df = pd.read_csv(dataset_csv_path)
    # dataset_df = dataset_df[dataset_df["split"] == "val"]
    # dataset_df = dataset_df[dataset_df["sid"] == "S0006"]
    # dataset_df = dataset_df[dataset_df["modality"] == "T1W"]
    # dataset_df = dataset_df[dataset_df["resolution"] == 7]


    use_controlnet = False
    
    dm_chk_number = 280000
    n_inference_steps = 30
    dm_seg_channels = 3
    used_mask = f"merged_{dm_seg_channels}"

    dm_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated_synthsr/test2_merged3_4res_probseg_nofgr_combined_data/check_points/model_{dm_chk_number}.pt"

    output_path = os.path.join(output_path, f"basic", used_mask, f"chk_{dm_chk_number}_steps_{n_inference_steps}")


    used_modalities = ["T1W", "T2W", "T2FLAIR"]  # "T1W", "T2W", "T2FLAIR"
    used_resolutions = [0.1, 1.5, 3, 5, 7]  # 0.1, 1.5, 3, 5, 7
    dm_tar_resolution_offset = 0  # this is the offset that we will apply to the target resolution index when feeding it to the model, because during training we only used 3 resolutions (0.1, 1.5 and 3), so the target resolution index for 1.5T during training was 0, for 3T was 1 and for 5T was 2. Now, during inference, we want to be able to use the model to generate images for all the resolutions, so we will apply an offset to the target resolution index to match the indices that were used during training. For example, if we want to generate an image for 7T, which has an index of 4 in our current setup, we will apply an offset of 2 to get an index of 2, which is the index that was used during training for 5T.

    modality_idx_mapping = {
        modality: idx for idx, modality in enumerate(used_modalities)
    }
    resolution_idx_mapping = {
        resolution: idx for idx, resolution in enumerate(used_resolutions)
    }

    # map modality to index
    dataset_df["modality_idx"] = dataset_df["modality"].map(modality_idx_mapping)
    dataset_df["resolution_idx"] = dataset_df["resolution"].map(resolution_idx_mapping)

    models_dict = instantiate_unconditioned_models(device, 
                                                dm_chk_path=dm_chk_path,
                                                dm_seg_channels=dm_seg_channels,
                                                dm_num_modalitites=len(used_modalities),
                                                dm_num_resolutions=len(used_resolutions), # because of the offset that we apply to the target resolution index
                                                dm_noise_scheduler_type="rflow",
                                                num_inference_steps=n_inference_steps
                                                )
    
    generated_paths = generate_identity_dataset(dataset_df, 
                              output_path, 
                              models_dict, 
                              nb_classes=dm_seg_channels)
    
    dataset_df["generated_image_path"] = generated_paths

    # save the dataset_df with the generated image paths
    dataset_df.to_csv(os.path.join(output_path, "generated_dataset.csv"), index=False)