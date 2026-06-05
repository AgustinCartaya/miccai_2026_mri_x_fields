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

device_name = f"cuda:0"
device = torch.device(device_name)


def set_seed(seed: int):
    # random.seed(seed)  # Semilla para Python
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed(seed)  # Semilla para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
    torch.backends.cudnn.deterministic = True  # Garantizar reproducibilidad en CNNs
    torch.backends.cudnn.benchmark = False  # Desactivar optimización no determinista



def instantiate_unconditioned_models(device, chk_path, used_resolutions, noise_scheduler_type="rflow", controlnet_chk_path=None, num_inference_steps=30):

    networks_config =  {
        
        "diffusion_unet_def": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4 + 3,  # 4 for the latent and 3 for the segmentation conditioning (concatenated)
            "out_channels": 4,
            "num_res_blocks": 2,
            "num_channels": [
                64,
                128,
                256,
                512,

            ],
            "self_attention_levels": [
                False,
                False,
                True,
                True
            ],

            "num_self_head_channels": [
                0,
                0,
                16,
                16
            ],
            "cross_attention_levels": [
                False,
                False,
                False,
                False
            ],
            "num_cross_head_channels": [
                0,
                0,
                0,
                0
            ],

            "use_flash_attention": True,
            "with_conditioning": False,
            "cross_attention_dim": None,
            "transformer_num_layers": 1, # number of transformer blocks
            "upcast_attention": True,

        },
        
        "controlnet_def": {
            "_target_": "monai.apps.generation.maisi.networks.controlnet_maisi.ControlNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4, # this is the nosy latent space
            "num_channels": [
                64,
                128,
                256,
                512
            ],
            "attention_levels": [
                False,
                False,
                True,
                True
            ],
            "num_head_channels": [
                0,
                0,
                16,
                16
            ],
            "num_res_blocks": 2,
            "use_flash_attention": True,
            "conditioning_embedding_in_channels": 4, # this is the condition image (e.g. low res/0.1T)
            "conditioning_embedding_num_channels": [2,4,8],#[8, 32, 64],
            "num_class_embeds": len(used_resolutions)  # this is the number of modalities that we have as conditions
        },
                
            
        "modality_encoder_def": { # this is volumetric conditioning
            "num_conditions": 3,  # number of conditions
            # "embed_dim": 512,  # this will be automatically set to be the same as the unet embedding dimension
        },

        "resolution_encoder_def": { # this is volumetric conditioning
            "num_conditions": 3,  # number of conditions
            # "embed_dim": 512,  # this will be automatically set to be the same as the unet embedding dimension
        },


        "noise_scheduler": {
            "_target_": "monai.networks.schedulers.DDIMScheduler", # faster scheduler
            "beta_start": 0.0015,
            "beta_end": 0.0205,
            "num_train_timesteps": 1000,
            "schedule": "scaled_linear_beta",
            "clip_sample": False
        },


        "noise_scheduler_rf": {
        "_target_": "monai.networks.schedulers.rectified_flow.RFlowScheduler",
        "num_train_timesteps": 1000,
        "use_discrete_timesteps": False,
        "use_timestep_transform": True,
        "sample_method": "uniform",
        "scale":1.4
        },


    }


    # instantiate model
    parser = ConfigParser(networks_config)
    parser.parse(True)

    args = fc.dict_to_args(networks_config, deep_conversion=True)

    # unet
    unet = diffusion_model_unet_maisi.DiffusionModelUNetMaisi(spatial_dims = args.diffusion_unet_def.spatial_dims,
                                                in_channels = args.diffusion_unet_def.in_channels,
                                                out_channels = args.diffusion_unet_def.out_channels,
                                                num_res_blocks = args.diffusion_unet_def.num_res_blocks,
                                                num_channels = args.diffusion_unet_def.num_channels,
                                                self_attention_levels = args.diffusion_unet_def.self_attention_levels,
                                                cross_attention_levels = args.diffusion_unet_def.cross_attention_levels,
                                                num_self_head_channels = args.diffusion_unet_def.num_self_head_channels,
                                                num_cross_head_channels = args.diffusion_unet_def.num_cross_head_channels,
                                                with_conditioning = args.diffusion_unet_def.with_conditioning,
                                                transformer_num_layers = args.diffusion_unet_def.transformer_num_layers,
                                                cross_attention_dim = args.diffusion_unet_def.cross_attention_dim,
                                                upcast_attention = args.diffusion_unet_def.upcast_attention,
                                                use_flash_attention = args.diffusion_unet_def.use_flash_attention,
                                                include_top_region_index_input=False,
                                                include_bottom_region_index_input=False,
                                                include_spacing_input=False
                                                )
    

    modality_encoder_model = conditions_model.SimpleConditionEmbedding(
                                                                    num_conditions=args.modality_encoder_def.num_conditions,
                                                                    embed_dim=unet.new_time_embed_dim,
                                                                    )
    
    resolution_encoder_model = conditions_model.SimpleConditionEmbedding(
                                                                    num_conditions=args.resolution_encoder_def.num_conditions,
                                                                    embed_dim=unet.new_time_embed_dim,
                                                                    )
    
    controlnet_model = controlnet_maisi.ControlNetMaisi(spatial_dims = args.controlnet_def.spatial_dims,
                                            in_channels = args.controlnet_def.in_channels,
                                            num_res_blocks = args.controlnet_def.num_res_blocks,
                                            num_channels = args.controlnet_def.num_channels,
                                            attention_levels = args.controlnet_def.attention_levels,
                                            num_head_channels = args.controlnet_def.num_head_channels,
                                            use_flash_attention = args.controlnet_def.use_flash_attention,
                                            conditioning_embedding_in_channels = args.controlnet_def.conditioning_embedding_in_channels,
                                            conditioning_embedding_num_channels = args.controlnet_def.conditioning_embedding_num_channels,
                                            num_class_embeds = None
                                            )
    ### HERE
        
    # noise scheduler
    if noise_scheduler_type == "ddim":
        noise_scheduler = parser.get_parsed_content("noise_scheduler", instantiate=True)
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    elif noise_scheduler_type == "rflow":
        # noise_scheduler = parser.get_parsed_content("noise_scheduler_rf", instantiate=True)
        noise_scheduler = RFlowScheduler(num_train_timesteps=args.noise_scheduler_rf.num_train_timesteps,
                                        use_discrete_timesteps=args.noise_scheduler_rf.use_discrete_timesteps,
                                        use_timestep_transform=args.noise_scheduler_rf.use_timestep_transform,
                                        sample_method=args.noise_scheduler_rf.sample_method,
                                        scale=args.noise_scheduler_rf.scale)
        noise_scheduler.set_timesteps(num_inference_steps=num_inference_steps,
                                    input_img_size_numel=torch.prod(torch.tensor((48,64,48))))


    autoencoder_chekpoint_path = "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_vae/vae_weights/autoencoder_epoch273.pt"
    autoencoder = AutoencoderPrediction(autoencoder_chekpoint_path, device, half=True)
    

    # Load pretrained model weights
    unet_checkpoint = torch.load(chk_path, map_location=device_name)
    unet.load_state_dict(unet_checkpoint["unet_state_dict"], strict=False)
    # segmentation_encoder_model.load_state_dict(unet_checkpoint["segmentation_encoder_model_state_dict"], strict=False)
    modality_encoder_model.load_state_dict(unet_checkpoint["modality_encoder_model_state_dict"], strict=False)
    resolution_encoder_model.load_state_dict(unet_checkpoint["resolution_encoder_model_state_dict"], strict=False)


    unet.to(device).eval()
    modality_encoder_model.to(device).eval()
    resolution_encoder_model.to(device).eval()


    if controlnet_chk_path is not None:
        controlnet_checkpoint = torch.load(controlnet_chk_path, map_location=device_name)
        controlnet_model.load_state_dict(controlnet_checkpoint["controlnet_state_dict"], strict=False)
        controlnet_model.to(device).eval()
    # controlnet_model.to(device).eval()

    return {"unet": unet, 
              "autoencoder": autoencoder, 
            "modality_encoder_model": modality_encoder_model,
            "resolution_encoder_model": resolution_encoder_model,
            "controlnet_model": controlnet_model,
              "noise_scheduler": noise_scheduler,
              "networks_config": args}





@torch.no_grad()
def validation(
    src_latent_mask,
    modality_idx,
    tar_resolution_idx,
    unet,
    noise_scheduler,
    modality_encoder_model,
    resolution_encoder_model,
    autoencoder,
    controlnet_model=None,
    src_latent=None,
    src_resolution_idx=None,
    seed=0, 
):

    use_controlnet = controlnet_model is not None and src_latent is not None and src_resolution_idx is not None
    latents_shape = src_latent_mask.shape

    # instantiate every time to generate using the same initial noise (using CPU generator)
    _l_shape = [1, 4, latents_shape[-3], latents_shape[-2], latents_shape[-1]]
    gen_randn = torch.Generator().manual_seed(seed) 
    latents = torch.randn(_l_shape, generator=gen_randn).half().to(device)

    src_segmentation = src_latent_mask.to(device)

    modality_idx = modality_idx.to(device)
    tar_resolution_idx = tar_resolution_idx.to(device)

    if use_controlnet:
        src_latents = src_latent.to(device)
        src_resolution_idx = src_resolution_idx.to(device)

    modality_embedding = modality_encoder_model(modality_idx)
    tar_resolution_embedding = resolution_encoder_model(tar_resolution_idx-2)


    all_timesteps = noise_scheduler.timesteps
    all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
    progress_bar = tqdm(
        zip(all_timesteps, all_next_timesteps),
        total=min(len(all_timesteps), len(all_next_timesteps)),
        # desc=f"Modality {modality_idx.item()} Resolution {batch['src_resolution']} -> {batch['tar_resolution']} {c+1}/{total_images}"
    )
    with torch.no_grad(), torch.amp.autocast("cuda"):

        for t, next_t in progress_bar:

            timesteps= torch.Tensor((t,)).to(device)
            # get controlnet output
            if use_controlnet:
                down_block_res_samples, mid_block_res_sample = controlnet_model(
                    x=latents, timesteps=timesteps, controlnet_cond=src_latents, class_labels =src_resolution_idx
                )

            # print(latents.shape, src_segmentation.shape, modality_embedding.shape, tar_resolution_embedding.shape)
            model_output = unet(
                x=torch.cat([latents, src_segmentation], dim=1),
                timesteps=timesteps,
                # mask_features = src_segmentation_embedding,
                modallity_embedding = modality_embedding,
                resolution_embedding = tar_resolution_embedding,
                down_block_additional_residuals = down_block_res_samples if use_controlnet else None,
                mid_block_additional_residual = mid_block_res_sample if use_controlnet else None
            )

            if not isinstance(noise_scheduler, RFlowScheduler):
                latents, _ = noise_scheduler.step(model_output, t, latents)
            else:
                latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore

        # free memory for the autoencoder
        # del model_output, down_block_res_samples, mid_block_res_sample
        del model_output
        # remove unet from GPU
        # unet.cpu()
        # modality_encoder_model.cpu()
        # resolution_encoder_model.cpu()
        torch.cuda.empty_cache()

        # decode the latents to images
        synthetic_images = autoencoder.decode(latents, decode_complete=False, sliding_window_size=(64, 64, 64))
        synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
        synthetic_images = synthetic_images.squeeze().numpy()

        # 
        # set unet and controlnet back to GPU for the next inference
        # unet.to(device).eval()
        # modality_encoder_model.to(device).eval()
        # resolution_encoder_model.to(device).eval()
        # controlnet_model.to(device).eval()

    return synthetic_images



def load_segmentation(segmentation_npy_path):
    segmentation = np.load(segmentation_npy_path)
    # convert into one-hot encoding
    unique_labels = np.unique(segmentation)
    # remove zero if it is in the unique labels (assuming zero is the background)
    if 0 in unique_labels:
        unique_labels = unique_labels[unique_labels != 0]

    seg_onehot = []     
    for label in unique_labels:
        seg_onehot.append(np.where(segmentation == label, 1.0, 0.0))   
    seg_onehot = np.stack(seg_onehot, axis=0)  # (C, D, H, W)
    
    return torch.from_numpy(seg_onehot).float()



def evaluate_task1(df_val, base_path, models, use_controlnet=False):
    base_path = os.path.join(base_path, "task1")

    # remove 7 test images if they already exist
    df_val_task1 = df_val[df_val["resolution"] != 7]

    bar = tqdm(df_val_task1.iterrows(), total=len(df_val_task1), desc="Generating synthetic images for Task 1")
    # filter 
    for i, row in df_val_task1.iterrows():    

        modality = row["modality"]
        resolution = row["resolution"]
        iid = row["iid"]
        src_latent_mask_path = row["latent_seg_supersynth_merged_3_path"]
        modality_idx = row["modality_idx"]
        src_resolution_idx = row["resolution_idx"]
        tar_resolution_idx = 4 # = 7T

        if resolution not in [0.1, 1.5]: # float naming problem
            resolution = int(resolution)
        save_path = os.path.join(base_path, modality, f"{resolution}T_to_7T", "pred")
        save_name = iid.replace(f"{resolution}T", f"{7}T") + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)


        bar.set_description(f"Generating synthetic images for Task 1 - sid: {row['sid']} Modality {modality} = {modality_idx} Resolution {resolution}T = {src_resolution_idx} -> {tar_resolution_idx}")

        # verify if the image already exists, if it does, skip it
        if os.path.exists(save_path_name):
            print(f"Image {save_path_name} already exists, skipping...")
            bar.update(1)  
            continue

        os.makedirs(save_path, exist_ok=True)


        # src_latent_mask = torch.from_numpy(np.load(src_latent_mask_path)).float().unsqueeze(0)  # load the latent mask as float32
        src_latent_mask = load_segmentation(src_latent_mask_path).unsqueeze(0)  # load the latent mask as float32 and add batch dimension
        modality_idx = torch.tensor(modality_idx).unsqueeze(0)  # convert to tensor and add batch dimension
        tar_resolution_idx = torch.tensor(tar_resolution_idx).unsqueeze(0)  # convert to tensor and add batch dimension

        synthetic_image = validation(
            src_latent_mask=src_latent_mask,
            modality_idx=modality_idx,
            tar_resolution_idx=tar_resolution_idx,
            unet=models["unet"],
            noise_scheduler=models["noise_scheduler"],
            modality_encoder_model=models["modality_encoder_model"],
            resolution_encoder_model=models["resolution_encoder_model"],
            autoencoder=models["autoencoder"],
            controlnet_model=models["controlnet_model"] if use_controlnet else None,
            src_latent=torch.from_numpy(np.load(row["latent_path"])).unsqueeze(0) if use_controlnet else None,
            src_resolution_idx=torch.tensor(src_resolution_idx).unsqueeze(0) if use_controlnet else None,
        )


        org_img, org_aff = nfc.load_nifti(row["org_img_path"])
        synthetic_image = prep_image.postprocess_img(synthetic_image, original_size=org_img.shape)
        nfc.save_nifti(synthetic_image, org_aff, save_path_name)
        bar.update(1)  


        break

    bar.close()
    

def evaluate_task2(df_val, base_path, models, use_controlnet=False):
    base_path = os.path.join(base_path, "task2")

    # remove 7 test images if they already exist
    df_val_task2 = df_val[df_val["resolution"] == 0.1]

    desired_resolution_list = [ 3, 5, 7]
    desired_resolution_idx_list = [2,3,4]

    bar = tqdm(df_val_task2.iterrows(), total=len(df_val_task2) * len(desired_resolution_idx_list), desc="Generating synthetic images for Task 2")
    # filter 
    for i, row in df_val_task2.iterrows(): 


        for tar_resolution_idx, tar_resolution in zip(desired_resolution_idx_list, desired_resolution_list):  


            modality = row["modality"]
            src_resolution = row["resolution"]
            iid = row["iid"]
            src_latent_mask_path = row["latent_seg_supersynth_merged_3_path"]
            modality_idx = row["modality_idx"]
            src_resolution_idx = row["resolution_idx"]


            if src_resolution not in [0.1, 1.5]: # float naming problem
                src_resolution = int(src_resolution)

            if tar_resolution not in [0.1, 1.5]: # float naming problem
                tar_resolution = int(tar_resolution)

            save_path = os.path.join(base_path, modality, f"{src_resolution}T_to_{tar_resolution}T", "pred")
            save_name = iid.replace(f"{src_resolution}T", f"{tar_resolution}T") + ".nii.gz"
            save_path_name = os.path.join(save_path, save_name)

            bar.set_description(f"Generating synthetic images for Task 2 - sid: {row['sid']} Modality {modality} = {modality_idx} Resolution {src_resolution}T = {src_resolution_idx} -> {tar_resolution_idx}")

            # verify if the image already exists, if it does, skip it
            if os.path.exists(save_path_name):
                print(f"Image {save_path_name} already exists, skipping...")
                bar.update(1)  
                continue

            os.makedirs(save_path, exist_ok=True)


            # src_latent_mask = torch.from_numpy(np.load(src_latent_mask_path)).float().unsqueeze(0)  # load the latent mask as float32
            src_latent_mask = load_segmentation(src_latent_mask_path).unsqueeze(0)  # load the latent mask as float32 and add batch dimension
            modality_idx = torch.tensor(modality_idx).unsqueeze(0)  # convert to tensor and add batch dimension
            tar_resolution_idx = torch.tensor(tar_resolution_idx).unsqueeze(0)  # convert to tensor and add batch dimension

            synthetic_image = validation(
                src_latent_mask=src_latent_mask,
                modality_idx=modality_idx,
                tar_resolution_idx=tar_resolution_idx,
                unet=models["unet"],
                noise_scheduler=models["noise_scheduler"],
                modality_encoder_model=models["modality_encoder_model"],
                resolution_encoder_model=models["resolution_encoder_model"],
                autoencoder=models["autoencoder"],
                controlnet_model=models["controlnet_model"] if use_controlnet else None,
                src_latent=torch.from_numpy(np.load(row["latent_path"])).unsqueeze(0) if use_controlnet else None,
                src_resolution_idx=torch.tensor(src_resolution_idx).unsqueeze(0) if use_controlnet else None,
            )


            org_img, org_aff = nfc.load_nifti(row["org_img_path"])
            synthetic_image = prep_image.postprocess_img(synthetic_image, original_size=org_img.shape)
            nfc.save_nifti(synthetic_image, org_aff, save_path_name)
            bar.update(1)  


        #     break
        # break

    bar.close()
    

if __name__ == "__main__":
    df_val = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv")
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



    used_modalities = ["T1W", "T2W", "T2FLAIR"] # "T1W", "T2W", "T2FLAIR"
    used_resolutions = [0.1, 1.5, 3, 5, 7] #0.1, 1.5, 3, 5, 7


    modality_idx_mapping = {modality: idx for idx, modality in enumerate(used_modalities)}
    resolution_idx_mapping = {resolution: idx for idx, resolution in enumerate(used_resolutions)}

    # map modality to index
    df_val["modality_idx"] = df_val["modality"].map(modality_idx_mapping)
    df_val["resolution_idx"] = df_val["resolution"].map(resolution_idx_mapping)


    models_dict = instantiate_unconditioned_models(device, 
                                                   chk_path=network_chk_path,
                                                   used_resolutions=used_resolutions, # this if for the controlnet
                                                   noise_scheduler_type="rflow",
                                                   controlnet_chk_path=controlnet_chk_path if use_controlnet else None,
                                                   num_inference_steps=n_inference_steps
                                                   )
 

    # evaluate_task1(df_val, output_path, models=models_dict, use_controlnet=use_controlnet)
    evaluate_task2(df_val, output_path, models=models_dict, use_controlnet=use_controlnet)



