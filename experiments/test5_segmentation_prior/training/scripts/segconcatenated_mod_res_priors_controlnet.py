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
# import  networks_declaration.segmentation_encoder as segmentation_encoder
import  networks_declaration.conditions_model as conditions_model
import  networks_declaration.controlnet_maisi as controlnet_maisi


# import attention_controller as attention_controller
from networks_declaration.rectified_flow import RFlowScheduler
# from monai.networks.schedulers.rectified_flow import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
# images
from PIL import Image

sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')
from autoencoder_declaration import AutoencoderPrediction

# device_name = f"cuda:{gpu_selector.get_least_used_gpu()}"
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





def instantiate_unconditioned_models(device, used_resolutions, noise_scheduler_type="rflow"):

    networks_config =  {
        
        "diffusion_unet_def": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4 + 3,  # 4 for the latent and 3 for the segmentation conditioning (concatenated)
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
                
            

        # "segmentation_encoder_def": { # this is volumetric conditioning
        #     "spatial_dims": 3,  # number of conditions
        #     "in_channels": 3,
        #     "num_res_blocks": 2,  # whether to use self-attention
        #     "num_channels": [
        #         # 32, 
        #         64, 
        #         128, 
        #         256,
        #         512
        #     ],  # half of the embedding dimension
        # },
                
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


  # class ConditionEmbedding(nn.Module):
    # def __init__(
    #     self,
    #     num_conditions=3,          # [T1, T2, FLAIR]
    #     embed_dim=512,             # output embedding dim
    #     proj_hidden_dim=256,
    #     use_gelu=True,
    #     dropout=0.1
    # ):


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
                                                # attention_levels = args.diffusion_unet_def.attention_levels,
                                                self_attention_levels = args.diffusion_unet_def.self_attention_levels,
                                                cross_attention_levels = args.diffusion_unet_def.cross_attention_levels,
                                                # num_head_channels = args.diffusion_unet_def.num_head_channels,
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
    

    # # ConditionTokens
    # conditions_model = volumne_encoder.ConditionTokens(num_conditions=args.conditions_model.num_conditions, 
    #                                 embed_dim=args.conditions_model.embed_dim,
    #                                 hidden_dim=args.conditions_model.hidden_dim,
    #                                 use_self_attention=args.conditions_model.use_self_attention, 
    #                                 n_heads=args.conditions_model.n_heads, 
    #                                 n_layers=args.conditions_model.n_att_layers,
    #                                 use_gelu=args.conditions_model.use_gelu
    #                                 )

    # segmentation_encoder_model = segmentation_encoder.SegmentationEncoder(
    #                                     spatial_dims=args.segmentation_encoder_def.spatial_dims,
    #                                     in_channels=args.segmentation_encoder_def.in_channels,
    #                                     num_res_blocks=args.segmentation_encoder_def.num_res_blocks,
    #                                     num_channels=args.segmentation_encoder_def.num_channels,
    #                                 )
    
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
        noise_scheduler.set_timesteps(num_inference_steps=50)
    elif noise_scheduler_type == "rflow":
        # noise_scheduler = parser.get_parsed_content("noise_scheduler_rf", instantiate=True)
        noise_scheduler = RFlowScheduler(num_train_timesteps=args.noise_scheduler_rf.num_train_timesteps,
                                        use_discrete_timesteps=args.noise_scheduler_rf.use_discrete_timesteps,
                                        use_timestep_transform=args.noise_scheduler_rf.use_timestep_transform,
                                        sample_method=args.noise_scheduler_rf.sample_method,
                                        scale=args.noise_scheduler_rf.scale)
        noise_scheduler.set_timesteps(num_inference_steps=30,
                                    input_img_size_numel=torch.prod(torch.tensor((48,64,48))))

    # autoencoder (just for validation)
    # autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    autoencoder_chekpoint_path = "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_vae/vae_weights/autoencoder_epoch273.pt"
    # checkpoint_autoencoder = torch.load(autoencoder_chekpoint_path, weights_only=True, map_location=device)
    # autoencoder.load_state_dict(checkpoint_autoencoder)
    # autoencoder.eval()
    
    autoencoder = AutoencoderPrediction(autoencoder_chekpoint_path, device, half=True)
    

    return {"unet": unet, 
              "autoencoder": autoencoder, 
            #   "segmentation_encoder_model": segmentation_encoder_model,
            "modality_encoder_model": modality_encoder_model,
            "resolution_encoder_model": resolution_encoder_model,
            "controlnet_model": controlnet_model,
              "noise_scheduler": noise_scheduler,
              "networks_config": args}





class LoadPaths:
    def __init__(self, 
                 dataset_path_name, 
                 used_modalities,
                   used_resolutions, 
                   dataset_filters=None,
                   max_subjects=None):
        """Load the dataset and latents from the specified paths.
        Args:
          - dataset_path_name: Path to the dataset.
          - conditions_keys_ordered: List of condition keys in the desired order.
          - dataset_filters: Optional filters to apply to the dataset in the form of a dictionary where keys are column names and values are lists of values to filter by.
          - att_mask_path: Path to the attention masks.
          - att_mask_resolution_list: List of resolutions for the attention masks.
          - att_mask_structure_mapping: Mapping of the attention mask structure.
        """
        # self.complete_dataset = load_dataset.LoadDataset(training_dataset_path_name) 
        self.df = pd.read_csv(dataset_path_name)

        if dataset_filters is not None:
            for column, values_list in dataset_filters.items():
                self.df = self.df[self.df[column].isin(values_list)]
                
        self.df = self.df[( self.df["modality"].isin(used_modalities)) & (self.df["resolution"].isin(used_resolutions)) ]

        self.modality_idx_mapping = {modality: idx for idx, modality in enumerate(used_modalities)}
        self.resolution_idx_mapping = {resolution: idx for idx, resolution in enumerate(used_resolutions)}

        # map modality to index
        self.df["modality_idx"] = self.df["modality"].map(self.modality_idx_mapping)
        self.df["resolution_idx"] = self.df["resolution"].map(self.resolution_idx_mapping)

        self.used_modalities = used_modalities
        self.used_resolutions = used_resolutions
        if max_subjects is not None:
            # self.df = self.df[self.df["sid"].isin(self.df["sid"].unique()[:max_subjects])]
            self.df = self.df.head(max_subjects)

    def get_data(self, split="train"):
        complete_df = self.df.copy()
        complete_df = complete_df[complete_df["split"] == split]
        
        self.sids = complete_df["sid"].unique()

        # instances = []
        # for i, row in complete_df.iterrows():
        #     instance_dict = {}
        #     latent_path = row["latent_path"]
        #     # verify that the latent path exists
        #     if not os.path.exists(latent_path):
        #         # print(f"Latent path {latent_path} does not exist. Skipping this instance.")
        #         continue
        #     instance_dict["latent_path"] = latent_path
        #     instance_dict["sid"] = row["sid"]
        #     instance_dict["modality"] = row["modality"]
        #     instance_dict["modality_idx"] = row["modality_idx"]
        #     instance_dict["resolution"] = row["resolution"]
        #     instance_dict["resolution_idx"] = row["resolution_idx"]
        #     instance_dict["segmentation_npy_path"] = row["latent_seg_synthseg_path"]
        #     instances.append(instance_dict)
        # return instances

        instances = {} 
        # for i, row in complete_df.iterrows():
        for subject_id in self.sids:
            instances[subject_id] = {}
            s_row = complete_df[complete_df["sid"] == subject_id]
            for modality in self.used_modalities:
                instances[subject_id][modality] = {}
                for resolution in self.used_resolutions:

                    row = s_row[(s_row["modality"] == modality) & (s_row["resolution"] == resolution)].iloc[0]
                    _instance = {}
                    _instance["latent_path"] = row["latent_path"]
                    _instance["segmentation_npy_path"] = row["latent_seg_supersynth_path"] if "latent_seg_supersynth_path" in row else row["latent_seg_synthseg_path"]
                    _instance["org_img_path"] = row["org_img_path"]

                    instances[subject_id][modality][resolution] = _instance
                    
        return instances



# sid,paired,resolution,modality,split,iid,org_img_path,seg_synthseg_path,seg_supersynth_path,latent_path,latent_seg_synthseg_path


class PrepareDataset(Dataset):
    def __init__(self, 
                 dataset_path_name,
                 used_modalities,
                 used_resolutions,
                    gen_modality,
                    gen_resolution,
                 dataset_filters=None,
                 split="train",
                 max_subjects=None,

                 ):

        # load data
        data_loader = LoadPaths(dataset_path_name,
                                    used_modalities=used_modalities,
                                    used_resolutions=used_resolutions,
                                  dataset_filters=dataset_filters,
                                    max_subjects=max_subjects
                                  )
        
        self.train_data = data_loader.get_data(split=split)
        self.ids_list = list(self.train_data.keys())
        self.split = split

    
        print(f"Number of {split} images: {len(self.train_data)}")

        # number of latent in the folder
        self.num_instances = len(self.train_data) 
        self._length = self.num_instances
        # self.ids_list = list(self.train_data.keys())

        self.used_modalities = used_modalities
        self.used_resolutions = used_resolutions
        self.gen_modality = gen_modality
        self.gen_resolution = gen_resolution


    def __len__(self):
        return self._length
    


    def load_segmentation(self, segmentation_npy_path):
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

    def __getitem__(self, index):
        subject_id = self.ids_list[index % len(self.ids_list)]
        instance = self.train_data[subject_id]

        example = {}

        # generate 2 random different numbers for modality using the gen_modality
        # src_resolution_idx, tar_resolution_idx = torch.randperm(len(self.used_resolutions), generator=self.gen_resolution)[:2].tolist()
        
        # this way we ensure that the target resolution is 3,5,or 7
        if self.split == "train":
            modality_idx = torch.randint(len(self.used_modalities), (1,), generator=self.gen_modality).item()
            src_resolution_idx = torch.randint(len(self.used_resolutions),(1,),generator=self.gen_resolution).item()
            target_candidates_idx = torch.tensor([2, 3, 4])
            tar_resolution_idx = target_candidates_idx[torch.randint(len(target_candidates_idx), (1,), generator=self.gen_resolution).item()].item()
        else:
            modality_idx = 0 # always start from T1 in validation
            src_resolution_idx = 1 # always start from the lowest resolution in validation
            tar_resolution_idx = 4 # always go to the highest resolution in validation

        modality = self.used_modalities[modality_idx]
        src_resolution = self.used_resolutions[src_resolution_idx]
        tar_resolution = self.used_resolutions[tar_resolution_idx]

        example["sid"] = subject_id
        example["modality"] = modality

        example["src_resolution"] = src_resolution
        example["tar_resolution"] = tar_resolution

        example["modality_idx"] = torch.tensor(modality_idx)
        example["src_resolution_idx"] = torch.tensor(src_resolution_idx)
        example["tar_resolution_idx"] = torch.tensor(tar_resolution_idx)


        example["src_latent"] = torch.from_numpy(np.load(instance[modality][src_resolution]["latent_path"]))
        example["src_segmentation"] = self.load_segmentation(instance[modality][src_resolution]["segmentation_npy_path"])


        example["tar_latent"] = torch.from_numpy(np.load(instance[modality][tar_resolution]["latent_path"]))
        example["tar_segmentation"] = self.load_segmentation(instance[modality][tar_resolution]["segmentation_npy_path"])


        example["src_org_img_path"] = instance[modality][src_resolution]["org_img_path"]
        example["tar_org_img_path"] = instance[modality][tar_resolution]["org_img_path"]
        return example
    

def collate_fn(examples):
    res_dict = {}

    res_dict["sid"] = [example["sid"] for example in examples]
    res_dict["modality"] = [example["modality"] for example in examples]
    res_dict["src_org_img_path"] = [example["src_org_img_path"] for example in examples]
    res_dict["tar_org_img_path"] = [example["tar_org_img_path"] for example in examples]

    res_dict["src_resolution"] = [example["src_resolution"] for example in examples]
    res_dict["tar_resolution"] = [example["tar_resolution"] for example in examples]

    modality_idx = torch.stack([example["modality_idx"] for example in examples])
    modality_idx = modality_idx.to(memory_format=torch.contiguous_format).long()
    res_dict["modality_idx"] = modality_idx

    src_resolution_idx = torch.stack([example["src_resolution_idx"] for example in examples])
    src_resolution_idx = src_resolution_idx.to(memory_format=torch.contiguous_format).long()
    res_dict["src_resolution_idx"] = src_resolution_idx

    tar_resolution_idx = torch.stack([example["tar_resolution_idx"] for example in examples])
    tar_resolution_idx = tar_resolution_idx.to(memory_format=torch.contiguous_format).long()
    res_dict["tar_resolution_idx"] = tar_resolution_idx

    src_latent = torch.stack([example["src_latent"] for example in examples])
    src_latent = src_latent.to(memory_format=torch.contiguous_format).float()
    res_dict["src_latent"] = src_latent

    tar_latent = torch.stack([example["tar_latent"] for example in examples])
    tar_latent = tar_latent.to(memory_format=torch.contiguous_format).float()
    res_dict["tar_latent"] = tar_latent

    src_segmentation = torch.stack([example["src_segmentation"] for example in examples])
    src_segmentation = src_segmentation.to(memory_format=torch.contiguous_format).float()
    res_dict["src_segmentation"] = src_segmentation

    tar_segmentation = torch.stack([example["tar_segmentation"] for example in examples])
    tar_segmentation = tar_segmentation.to(memory_format=torch.contiguous_format).float()
    res_dict["tar_segmentation"] = tar_segmentation

    return res_dict


def instantiate_dataset(dataset_path_name, used_modalities, used_resolutions,
                        gen_modality, gen_resolution,
                        batch_size, 
                        dataset_filters=None,
                        split="train",
                        max_subjects=None
                        ):
    # ---- Data set creation
    train_dataset = PrepareDataset(
        dataset_path_name=dataset_path_name,
        used_modalities=used_modalities,
        used_resolutions=used_resolutions,
        gen_modality=gen_modality,
        gen_resolution=gen_resolution,
        dataset_filters=dataset_filters,
        split=split,
        max_subjects=max_subjects
    )

    # sampler = MaxPerSubjectSampler(train_dataset, max_per_subject=max_timepoints_per_epoch, shuffle=True, generator=gen_dataloader)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=split=="train",  # shuffle only in training
        # sampler=sampler,
        collate_fn=lambda examples: collate_fn(examples),
        # generator=gen_dataloader,
        num_workers=2, 
        persistent_workers=True,
    )
    return train_dataloader



@torch.no_grad()
def validation(
    unet,
    noise_scheduler,
    # segmentation_encoder_model,
    modality_encoder_model,
    resolution_encoder_model,
    controlnet_model,
    autoencoder,
    val_dataloader,
    step,
    args,

):

    print(f"Validation step {step}")
    modality_idx_mapping = {modality: idx for idx, modality in enumerate(args.used_modalities)}
    resolution_idx_mapping = {resolution: idx for idx, resolution in enumerate(args.used_resolutions)}

    # output_path_img = os.path.join(args.output_path, args.val_imgs_dir_name, f"step_{step}", "images")
    output_path_step = os.path.join(args.output_path, args.val_imgs_dir_name, f"step_{step}")
    output_path_2D_images = os.path.join(output_path_step, "images2d")

    latents_shape = args.latents_shape
    for _output_path in [ output_path_2D_images]:
        os.makedirs(_output_path, exist_ok=True)

    imgs_list = []
    total_images = len(args.val_seeds) * len(val_dataloader)
    c = 0
    for seed in args.val_seeds:
        seed_img_list = []
        for batch in val_dataloader:
            # instantiate every time to generate using the same initial noise (using CPU generator)
            _l_shape = [1, latents_shape[-4], latents_shape[-3], latents_shape[-2], latents_shape[-1]]
            gen_randn = torch.Generator().manual_seed(seed) 
            latents = torch.randn(_l_shape, generator=gen_randn).half().to(device)

            src_latents = batch["src_latent"].to(device)
            # tar_latents = batch["tar_latent"].to(device)

            src_segmentation = batch["src_segmentation"].to(device)
            # tar_segmentation = batch["tar_segmentation"].to(device)

            modality_idx = batch["modality_idx"].to(device)
            print("modality_idx", modality_idx.shape)
            src_resolution_idx = batch["src_resolution_idx"].to(device)
            tar_resolution_idx = batch["tar_resolution_idx"].to(device)

            # src_segmentation_embedding = segmentation_encoder_model(src_segmentation)
            # segmentation = batch["segmentation"].to(device)
            modality_embedding = modality_encoder_model(modality_idx)
            tar_resolution_embedding = resolution_encoder_model(tar_resolution_idx-2)


            all_timesteps = noise_scheduler.timesteps
            all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
            progress_bar = tqdm(
                zip(all_timesteps, all_next_timesteps),
                total=min(len(all_timesteps), len(all_next_timesteps)),
                desc=f"Step {step} Modality {batch['modality']} Resolution {batch['src_resolution']} -> {batch['tar_resolution']} {c+1}/{total_images}"
            )
            c+=1
            with torch.no_grad(), torch.amp.autocast("cuda"):

                for t, next_t in progress_bar:

                    timesteps= torch.Tensor((t,)).to(device)
                    # get controlnet output
                    down_block_res_samples, mid_block_res_sample = controlnet_model(
                        x=latents, timesteps=timesteps, controlnet_cond=src_latents, class_labels =src_resolution_idx
                    )

                    model_output = unet(
                        x=torch.cat([latents, src_segmentation], dim=1),
                        timesteps=timesteps,
                        # mask_features = src_segmentation_embedding,
                        modallity_embedding = modality_embedding,
                        resolution_embedding = tar_resolution_embedding,
                        down_block_additional_residuals = down_block_res_samples,
                        mid_block_additional_residual = mid_block_res_sample
                    )

                    if not isinstance(noise_scheduler, RFlowScheduler):
                        latents, _ = noise_scheduler.step(model_output, t, latents)
                    else:
                        latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore

                # free memory for the autoencoder
                del model_output, down_block_res_samples, mid_block_res_sample
                torch.cuda.empty_cache()
                
                # decode the latents to images
                synthetic_images = autoencoder.decode(latents, decode_complete=True)
                synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
                synthetic_images = synthetic_images.squeeze().numpy()

                # decode the latents to images
                # path_name_img = os.path.join(output_path_images, f"img_step_{step}_seed_{seed}_cond_{i}.nii.gz")
                # path_name_imgs_list.append(path_name_img)
                # nfc.save_nifti(synthetic_images, args.ref_aff, path_name_img)
                
                # if not evaluate:
                #     print("Skipping rest of the conditions and seeds because evaluate=False")
                #     break


            # load the original image for visualization            org_img_path = batch["org_img_path"][0]
            src_img = nfc.load_nifti(batch["src_org_img_path"][0])[0]
            src_img = prep_image.prepare_img(src_img)
            seed_img_list.append(src_img)

            seed_img_list.append(synthetic_images)

            tar_img = nfc.load_nifti(batch["tar_org_img_path"][0])[0]
            tar_img = prep_image.prepare_img(tar_img)
            seed_img_list.append(tar_img)

        imgs_list.extend(seed_img_list)

    # ---- save 2D images for visualization
    # # obtain 3 layers of the images
    imgs_list_2D = fc.cat_n_views_different_layers(imgs_list, 
                                                view_layersoffset_list=[(2, 0), (2, -15), (1, 0), (0, 10)], 
                                                axis=0, 
                                                img_cropping=50,
                                                to_rgb=True)
    # save synthetic images
    complete_img = np.concatenate(imgs_list_2D, axis=1)
    complete_img = Image.fromarray(complete_img)
    complete_img.save(f"{output_path_2D_images}/imgs2D_step_{step}_seed_{seed}.png" )


    return None





def save_model(controlnet_model, optimizer, lr_scheduler, global_step, out_model_path, ema=None, best=False):  # MOD: se añade parámetro ema
    # Guardar el modelo
    controlnet_state_dict = controlnet_model.module.state_dict() if torch.distributed.is_initialized() else controlnet_model.state_dict()
    checkpoint = {
        "controlnet_state_dict": controlnet_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "num_train_timesteps": global_step,
    }
    # MOD: Agregar los pesos EMA en el checkpoint
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.shadow

    if best:
        # find the best checkpoint (is the file that ends by _best.pt)
        path_name_old_best_chk = glob.glob(os.path.join(out_model_path, "*_best.pt"))
        if path_name_old_best_chk:
            os.remove(path_name_old_best_chk[0])
        global_step = f"{global_step}_best"

    torch.save(checkpoint, f"{out_model_path}/model_{global_step}.pt")
    print(f"Model saved in {out_model_path}/model_{global_step}.pt")
    
    del checkpoint, controlnet_state_dict
    torch.cuda.empty_cache()
    gc.collect()


def load_checkpoint(checkpoint_path, controlnet_model, device, train_dataloader_len,
                    gradient_accumulation_steps, batch_size, optimizer=None, lr_scheduler=None, ema=None):
    # 1. Load checkpoint on CPU to avoid using VRAM
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Load weights into models
    controlnet_model.load_state_dict(checkpoint["controlnet_model_state_dict"], strict=False)

    # 3. Move models to GPU
    controlnet_model.to(device)

    # 4. EMA (optional)
    if ema is not None and "ema_state_dict" in checkpoint:
        # Move only the EMA tensors to GPU
        ema.shadow = {k: v.to(device) for k, v in checkpoint["ema_state_dict"].items()}

    # 5. Optimizer and scheduler (optional)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        # For Adam, some states may be on CPU and others on GPU
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Move optimizer buffers to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)


    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint and checkpoint["lr_scheduler_state_dict"] is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

    # 6. Compute first_epoch and global_step
    global_step = checkpoint["num_train_timesteps"]
    first_epoch = (global_step * gradient_accumulation_steps * batch_size) // train_dataloader_len + 1

    print(f"Model loaded from {checkpoint_path}")
    print(f"Resuming training from epoch {first_epoch} and global step {global_step}")

    return global_step, first_epoch





def save_configurations(args_train, networks_config, config_path, config_name="model_config.json"):
    argparse_dict = {
        "args_train": fc.args_to_dict(args_train, deep_conversion=True),
        "networks_config": fc.args_to_dict(networks_config, deep_conversion=True)
    }
    argparse_json = json.dumps(argparse_dict, indent=4)
    with open(os.path.join(config_path, config_name), "w") as outfile:
        outfile.write(argparse_json)
    print(f"Model configurations saved in: {os.path.join(config_path, config_name)}")





class EMA:
    def __init__(self, model, decay, warm_up_steps=0, warm_up_decay=0.1, optimize_cpu=False):
        """
        Inicializa la clase EMA para gestionar la media móvil exponencial de los parámetros del modelo.
        Args:
            model (torch.nn.Module): El modelo cuyos parámetros se van a promediar.
            decay (float): Tasa de decaimiento para la EMA.
        """
        self.model = model
        self.decay = decay
        self.warm_up_steps = warm_up_steps
        self.warm_up_decay = warm_up_decay
        
        self.shadow = {}
        self.backup = {}
        self.optimize_cpu = optimize_cpu

        for name, param in model.named_parameters():
            if param.requires_grad:
                if self.optimize_cpu:
                    self.shadow[name] = param.detach().cpu().clone()
                else:
                    self.shadow[name] = param.detach().clone()

    def update(self, step=None):
        """
        Actualiza los parámetros sombra utilizando la EMA.
        """
        decay = self.decay
        if step is not None and self.warm_up_steps > 0 and step < self.warm_up_steps:
            decay = self.warm_up_decay
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.optimize_cpu:
                    new_avg = (1.0 - decay) * param.detach().cpu() + decay * self.shadow[name]
                else:
                    new_avg = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_avg

    def apply_shadow(self):
        """
        Aplica los parámetros promediados (EMA) al modelo, guardando los originales.
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # self.backup[name] = param.data.clone()
                # param.data.copy_(self.shadow[name])
                if self.optimize_cpu:
                    self.backup[name] = param.detach().clone()  # keep original
                    param.data.copy_(self.shadow[name].to(param.device)) 
                else:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self):
        """
        Restaura los parámetros originales del modelo.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}





def create_warmup_cosine_scheduler(optimizer, warmup_start_factor, warmup_steps, max_train_steps, eta_min):
    """
    Creates a SequentialLR with a Linear warmup followed by CosineAnnealingLR.

    "warmup_start_factor": 1e-2 # Initial learning rate factor (multiplied by base_lr) during warmup
    "warmup_steps": 1000, # Number of steps for warmup
    "eta_min": 1e-6, # Minimum learning rate after warmup

    """

    # Warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    # Cosine decay scheduler
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_train_steps - warmup_steps,
        eta_min=eta_min
    )

    # Combine them
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler



def train(
    args_train,
    device,
):

    # ---- reproducibility
    set_seed(args_train.seed)
    gen_t = torch.Generator().manual_seed(args_train.seed) 
    gen_noise = torch.Generator().manual_seed(args_train.seed)
    gen_modality = torch.Generator().manual_seed(args_train.seed)
    gen_resolution = torch.Generator().manual_seed(args_train.seed)

    models_dict = instantiate_unconditioned_models(device, used_resolutions=args_train.used_resolutions,noise_scheduler_type=args_train.noise_scheduler_type)
    unet = models_dict["unet"]
    # segmentation_encoder_model = models_dict["segmentation_encoder_model"]
    modality_encoder_model = models_dict["modality_encoder_model"]
    resolution_encoder_model = models_dict["resolution_encoder_model"]
    controlnet_model = models_dict["controlnet_model"]
    autoencoder = models_dict["autoencoder"]
    networks_config = models_dict["networks_config"]
    noise_scheduler = models_dict["noise_scheduler"]


    def freeze_model(model):
        for param in model.parameters():
            param.requires_grad = False

    # Load pretrained model weights
    unet_checkpoint = torch.load(args_train.pretrained_models_path, map_location=device_name)
    unet.load_state_dict(unet_checkpoint["unet_state_dict"], strict=False)
    # segmentation_encoder_model.load_state_dict(unet_checkpoint["segmentation_encoder_model_state_dict"], strict=False)
    modality_encoder_model.load_state_dict(unet_checkpoint["modality_encoder_model_state_dict"], strict=False)
    resolution_encoder_model.load_state_dict(unet_checkpoint["resolution_encoder_model_state_dict"], strict=False)

    copy_model_state(controlnet_model, unet.state_dict())
    freeze_model(unet)
    # freeze_model(segmentation_encoder_model)
    freeze_model(modality_encoder_model)
    freeze_model(resolution_encoder_model)


    
    
    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        used_modalities=args_train.used_modalities,
        used_resolutions=args_train.used_resolutions,
        gen_modality=gen_modality,
        gen_resolution=gen_resolution,
        # latents_path=args_train.latents_path,
        batch_size=args_train.batch_size,
        split="train",
        dataset_filters = {"paired":[1]}
    )

    val_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        used_modalities=args_train.used_modalities,
        used_resolutions=args_train.used_resolutions,
        gen_modality=None,  # no need to generate random modality and resolution for validation, we will always start from the same one (T1 at the lowest resolution)
        gen_resolution=None,
        batch_size=1,
        split="val",
        max_subjects=args_train.max_val_subjects,
        dataset_filters = {"paired":[1]}
    
    )


    # ---- create folders
    os.makedirs(args_train.output_path, exist_ok=True)
    _checkpoint_dir_name =  os.path.join(args_train.output_path, args_train.checkpoints_dir_name)
    _logs_dir_name = os.path.join(args_train.output_path, args_train.logs_dir_name)
    _val_imgs_dir_name = os.path.join(args_train.output_path, args_train.val_imgs_dir_name)
    os.makedirs(_checkpoint_dir_name, exist_ok=True)
    os.makedirs(_logs_dir_name, exist_ok=True)
    os.makedirs(_val_imgs_dir_name, exist_ok=True)


    # ---- save configurations
    save_configurations(args_train, networks_config, args_train.output_path)
    args_train.networks_config = networks_config  # add the networks config to the args for later use in validation and sampling

    # ---- create tensorboard writer and save configurations
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  # Formato: Año-Mes-Día_Hora-Minuto
    _sum_writter_dir = os.path.join(_logs_dir_name, f"logs_{timestamp}")
    os.makedirs(_sum_writter_dir, exist_ok=True)
    writer = SummaryWriter(_sum_writter_dir)

    # ---- optimizer and lr_scheduler
    optimizer = torch.optim.AdamW(
        list(controlnet_model.parameters()),
        lr=args_train.lr,
        # betas=(0.9, 0.999),
        # eps=1e-8,
        # weight_decay=0.01
    )

    if args_train.lr_scheduler is not None:
        if args_train.lr_scheduler.name == "PolynomialLR":
            lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args_train.max_train_steps, power=args_train.lr_scheduler.power)
        elif args_train.lr_scheduler.name == "CosineAnnealingLR":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_train.max_train_steps, eta_min=args_train.lr_scheduler.eta_min)
        elif args_train.lr_scheduler.name == "WarmupCosineLR":
            lr_scheduler = create_warmup_cosine_scheduler(optimizer,
                                                          warmup_start_factor=args_train.lr_scheduler.warmup_start_factor,
                                                          warmup_steps=args_train.lr_scheduler.warmup_steps,
                                                          max_train_steps=args_train.max_train_steps,
                                                          eta_min=args_train.lr_scheduler.eta_min)
            
                     
    else:
        lr_scheduler = None

    # ---- loss function
    # loss_pt = torch.nn.L1Loss()

    # ---- training loop
    first_epoch = 0
    global_step = 0
    max_epochs = (args_train.max_train_steps * args_train.gradient_accumulation_steps * args_train.batch_size) // len(train_dataloader) + 1
    print(f"Max epochs: {max_epochs}")

    # ---- resume from checkpoint
    unet.to(device)
    # segmentation_encoder_model.to(device)
    modality_encoder_model.to(device)
    resolution_encoder_model.to(device)
    controlnet_model.to(device)

    # Initilize ema
    if args_train.use_ema:
        ema = EMA(unet, 
                  decay=args_train.ema_params.decay, 
                  warm_up_steps=args_train.ema_params.warm_up_steps, 
                  warm_up_decay=args_train.ema_params.warm_up_decay,
                  optimize_cpu=False)
    else:
        ema = None

    # priority is to resume from check point
    if args_train.resume_from_checkpoint_path_name is not None:
        global_step, first_epoch = load_checkpoint(
            args_train.resume_from_checkpoint_path_name,
            controlnet_model,
            device=device,
            train_dataloader_len=len(train_dataloader),
            gradient_accumulation_steps=args_train.gradient_accumulation_steps,
            batch_size=args_train.batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema
        )

    elif args_train.load_pretrained_model_from is not None:
        checkpoint = torch.load(args_train.load_pretrained_model_from, map_location=device_name)
        controlnet_model.load_state_dict(checkpoint["controlnet_model_state_dict"], strict=False)
        if args_train.use_ema and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
            print("EMA state loaded from checkpoint")
        print(f"Pretrained model loaded from {args_train.load_pretrained_model_from}")
    
    controlnet_model.train()

    # ---- memory reduction
    # -------- automatic mixed precision
    if args_train.amp:
        scaler = GradScaler()
    else:
        scaler = None
    gradient_accumulation_count = 0



    # ---- training loop
    progress_bar = tqdm(
        range(0, args_train.max_train_steps),
        desc="Training",
        initial=global_step
    )

    for epoch in range(first_epoch, max_epochs):
        for batch in train_dataloader:

            # prepare inputs
            src_latents = batch["src_latent"].to(device)
            tar_latents = batch["tar_latent"].to(device)

            src_segmentation = batch["src_segmentation"].to(device)
            # tar_segmentation = batch["tar_segmentation"].to(device)

            modality_idx = batch["modality_idx"].to(device)
            src_resolution_idx = batch["src_resolution_idx"].to(device)
            tar_resolution_idx = batch["tar_resolution_idx"].to(device)


            # Forward pass
            with autocast("cuda", enabled=args_train.amp):
                # generate noise and timesteps with dedicate generatos and in the cpu for reproducibility
                noise = torch.randn(tar_latents.shape, device="cpu", generator=gen_noise).to(device)
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(tar_latents)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (tar_latents.shape[0],), device="cpu", generator=gen_t).long().to(device)

                noisy_latent = noise_scheduler.add_noise(original_samples=tar_latents, noise=noise, timesteps=timesteps)
                
                # segmentation_embedding = segmentation_encoder_model(src_segmentation.to(device))
                target_modality_embedding = modality_encoder_model(modality_idx)
                target_resolution_embedding = resolution_encoder_model(tar_resolution_idx-2)  # restar 2 para que vaya de 0 a 2 y no de 2 a 4, porque el modelo ya ha visto resoluciones de indice 0,1,2 durante el preentrenamiento

                # if batch['modality'][0] == "T1W":
                # print("-"*10, "Modality embedding", "-"*10)
                # print(f"modality: \n {batch['modality']} \n {batch['modality_idx']} \n {modality_embedding[:,::50]}")
                # print("-"*10, "Resolution embedding", "-"*10)
                # print(f"resolution: \n {batch['resolution']} \n {batch['resolution_idx']} \n {resolution_embedding[:,::50]}")

                # get controlnet output
                # print("noisy_latent shape", noisy_latent.shape)
                # print("src_latent shape", src_latents.shape)
                down_block_res_samples, mid_block_res_sample = controlnet_model(
                    x=noisy_latent, timesteps=timesteps, controlnet_cond=src_latents, class_labels =src_resolution_idx
                )

                model_output = unet(
                                torch.cat([noisy_latent, src_segmentation], dim=1), 
                                  timesteps=timesteps,
                                #   context = volumetric_embedding,
                                    # mask_features = segmentation_embedding,
                                    modallity_embedding = target_modality_embedding,
                                    resolution_embedding = target_resolution_embedding,
                                    down_block_additional_residuals=down_block_res_samples,
                                    mid_block_additional_residual=mid_block_res_sample,
                                    )
                

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    # predict noise
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    # predict sample
                    model_gt = tar_latents
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    # predict velocity
                    model_gt = tar_latents - noise
                else:
                    raise ValueError(
                        "noise scheduler prediction type has to be chosen from ",
                        f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                    )

            # loss_noise = loss_pt(model_output.float(), model_gt.float())   # Dividir para escalar la pérdida
            # loss = loss_noise / args_train.gradient_accumulation_steps  # Dividir la pérdida por los pasos de acumulación de gradientes

            loss = F.l1_loss(model_output.float(), model_gt.float())

            # Acumulación de gradientes
            if args_train.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            gradient_accumulation_count += 1  # Contador de pasos acumulados

            # Solo se actualizan los pesos cada `gradient_accumulation_steps` pasos
            if gradient_accumulation_count % args_train.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args_train.amp:
                    scaler.unscale_(optimizer)  # Desescalar antes de clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(controlnet_model.parameters()),
                        max_norm=1.0
                    )
                           
                if args_train.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # update ema
                if args_train.use_ema:
                    ema.update(step=global_step)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                gradient_accumulation_count = 0  # Reiniciar el contador

                # update writter 
                if global_step % 10 == 0:
                    writer.add_scalar("Loss/train", loss.item(), global_step)
                    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("Time_steps", timesteps[0], global_step)

                # update progress bar 
                progress_bar.update(1)

                logs = {"0loss": loss.detach().item(), 
                        "1sres": [resolution for resolution in batch["src_resolution"]],
                        "2tres": [resolution for resolution in batch["tar_resolution"]],
                        "3mod": [modality for modality in batch["modality"]],
                        "4id": [f'{__id[:4]}' for __id in batch["sid"]], 
                        # **logs_conditions,
                        }
                
                progress_bar.set_postfix(**logs)

                # update global step
                global_step += 1

                # save the model in intervals
                if global_step % args_train.save_checkpoint_interval == 0:
                    save_model(controlnet_model=controlnet_model,
                               optimizer=optimizer,
                               lr_scheduler=lr_scheduler, 
                               global_step=global_step, 
                               out_model_path=_checkpoint_dir_name, 
                               ema=ema)

                # Generar imágenes en intervalos
                if args_train.initial_val or global_step % args_train.val_interval == 0:
                    controlnet_model.eval()

                    if args_train.latents_shape is None:
                        args_train.latents_shape = tar_latents.shape[-4:]  # save latents shape for later use in validation

                    try:
                        if args_train.use_ema:
                            ema.apply_shadow()
                        validation(unet=unet, 
                                   noise_scheduler=noise_scheduler, 
                                #    segmentation_encoder_model=segmentation_encoder_model, 
                                      modality_encoder_model=modality_encoder_model,
                                      resolution_encoder_model=resolution_encoder_model,
                                      controlnet_model=controlnet_model,
                                   autoencoder=autoencoder, 
                                   val_dataloader=val_dataloader, 
                                   step=global_step, 
                                   args=args_train)
                    except Exception as e:
                        print(f"ERROR DURING VALIDATION STEP {global_step}: {e}")
                    finally:
                        if args_train.use_ema:
                            ema.restore()
                            
                    args_train.initial_val = False
                    controlnet_model.train()
                    # conditions_model.train()

                if global_step >= args_train.max_train_steps:
                    break

        if global_step >= args_train.max_train_steps:
            break

    # make sure the progress bar closes
    progress_bar.close()

    # # make  out_model_path dir if it does not exist
    save_model(controlnet_model=controlnet_model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler, 
                global_step=global_step, 
                out_model_path=_checkpoint_dir_name, 
                ema=ema)







args_train = {
    # directories 
    "output_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated_controlnet2",
    "checkpoints_dir_name": "check_points",
    "logs_dir_name": "logs",
    "val_imgs_dir_name": "val_imgs",

    # data
    "df_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/paired_train_data.csv",

    # training configuration
    "max_train_steps": 500000,
    "save_checkpoint_interval": 5000,#10000,

    # ---- memory reduction
    "amp": True,

    # ---- Training stability
    "batch_size": 6, #6 
    "gradient_accumulation_steps": 1,#2,
    "use_ema": False,
    "ema_params": {
        "decay": 0.999,
        "warm_up_steps": 2000,
        "warm_up_decay": 0.5,
    },

    # ---- optimizer
    "lr":  1e-4, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    # "segmentation_encoder_lr": 5e-5, # 1e-4 for maisi, 1e-5 for blsmd
    # "lr":  1e-3, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    # "lr":  2.5e-5, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5

    # ---- lr_scheduler
    # "lr_scheduler": None,
    # "lr_scheduler": {"name": "PolynomialLR", "power": 2.0},
    # "lr_scheduler": {"name": "CosineAnnealingLR", "eta_min": 1e-6},
    "lr_scheduler": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-2, "warmup_steps": 500, "eta_min": 1e-6},
    # "lr_scheduler_segmentation_encoder": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-4, "warmup_steps": 500, "eta_min": 1e-6},
    # "lr_scheduler": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-2, "warmup_steps": 25, "eta_min": 1e-6},

    # ---- Pretrained UNET -----
    "pretrained_models_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated/test1/check_points/model_210000.pt",

    # ---- pretrained_model
    # "load_pretrained_model_from": "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_training/no_synthsr/aaco5590_dataset_no_outliers_bfc/models/rflow/check_points/model_200000.pt",
    # "load_pretrained_model_from": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/test1/check_points/model_20.pt",
    "load_pretrained_model_from": None, # not working


    # ---- resume from checkpoint
    # "resume_from_checkpoint_path_name": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/test1_mask_features_opt2/check_points/model_5000.pt",
    "resume_from_checkpoint_path_name": None, # not working

    # reproducibility
    "seed": 42,

    # validation
    "val_interval": 500,
    "initial_val": True, # remember drop out
    # "validation_first": True, # if True, the model will be validated before the first training step, if False, the model will be validated after the first training step
    "val_seeds": [0],#[0,12357], # seeds for the noise generation during validation
    "max_val_subjects": None, # max number of subjects to be generated during validation, set to None to use all the subjects in the val dataloader
    # "val_dataset_filters": {
    #     'iid': ["SP_T1W_3T_0006", "SP_T1W_5T_0006", "SP_T1W_7T_0006", 
    #                    "SP_T2W_3T_0006", "SP_T2W_5T_0006", "SP_T2W_7T_0006",
    #                    "SP_T2FLAIR_3T_0006", "SP_T2FLAIR_5T_0006", "SP_T2FLAIR_7T_0006",]
    # },
    "val_dataset_filters":None, # filters for the validation dataset, set to None to use all the subjects in the val dataloader
    # specialied synthesis
    # "specialized_idx": 1, # None for random, 0 for t1n, 1 for t1c, 2 for t2w, 3 for t2f
    
    "used_modalities": ["T1W", "T2W", "T2FLAIR"], # "T1W", "T2W", "T2FLAIR"
    "used_resolutions": [0.1, 1.5, 3, 5, 7], #0.1, 1.5, 3, 5, 7

    # "identity_allowed": True, # if True, the model can learn the identity function, if False, the model has to learn the conversion

    "loss_weights": {
        # "mse": 1.0,
        # "charbonnier": 1.0,
        # "ssim": 0.1,

    },

    "noise_scheduler_type": "rflow", # "ddpm" or "rflow"
    "latents_shape": None, # filled automatically based on the dataset
}


args_train = fc.dict_to_args(args_train, deep_conversion=True)
train(
    args_train,
    device,
)


