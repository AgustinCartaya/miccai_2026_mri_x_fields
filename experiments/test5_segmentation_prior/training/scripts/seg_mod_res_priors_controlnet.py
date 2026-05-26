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
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test4_finetune_brainst_diffusion_model/training/networks_declaration')


# pytorch
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
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


# monai
from monai.bundle import ConfigParser

import  networks_declaration.diffusion_model_unet_maisi_mask_seg as diffusion_model_unet_maisi
import  networks_declaration.segmentation_encoder as segmentation_encoder
import  networks_declaration.conditions_model as conditions_model


# import attention_controller as attention_controller
from networks_declaration.rectified_flow import RFlowScheduler
# from monai.networks.schedulers.rectified_flow import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
# images
from PIL import Image

sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')
from autoencoder_declaration import AutoencoderPrediction

# device_name = f"cuda:{gpu_selector.get_least_used_gpu()}"
device_name = f"cuda:1"
device = torch.device(device_name)


def set_seed(seed: int):
    # random.seed(seed)  # Semilla para Python
    np.random.seed(seed)  # Semilla para NumPy
    torch.manual_seed(seed)  # Semilla para PyTorch en CPU
    torch.cuda.manual_seed(seed)  # Semilla para PyTorch en GPU
    torch.cuda.manual_seed_all(seed)  # Semilla para todas las GPUs
    torch.backends.cudnn.deterministic = True  # Garantizar reproducibilidad en CNNs
    torch.backends.cudnn.benchmark = False  # Desactivar optimización no determinista





def instantiate_unconditioned_models(device, noise_scheduler_type="rflow"):

    networks_config =  {
        
        "diffusion_unet_def": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4,
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

        "segmentation_encoder_def": { # this is volumetric conditioning
            "spatial_dims": 3,  # number of conditions
            "in_channels": 3,
            "num_res_blocks": 2,  # whether to use self-attention
            "num_channels": [
                # 32, 
                64, 
                128, 
                256,
                512
            ],  # half of the embedding dimension
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

    segmentation_encoder_model = segmentation_encoder.SegmentationEncoder(
                                        spatial_dims=args.segmentation_encoder_def.spatial_dims,
                                        in_channels=args.segmentation_encoder_def.in_channels,
                                        num_res_blocks=args.segmentation_encoder_def.num_res_blocks,
                                        num_channels=args.segmentation_encoder_def.num_channels,
                                    )
    
    modality_encoder_model = conditions_model.SimpleConditionEmbedding(
                                                                    num_conditions=args.modality_encoder_def.num_conditions,
                                                                    embed_dim=unet.new_time_embed_dim,
                                                                    )
    
    resolution_encoder_model = conditions_model.SimpleConditionEmbedding(
                                                                    num_conditions=args.resolution_encoder_def.num_conditions,
                                                                    embed_dim=unet.new_time_embed_dim,
                                                                    )
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
              "segmentation_encoder_model": segmentation_encoder_model,
            "modality_encoder_model": modality_encoder_model,
            "resolution_encoder_model": resolution_encoder_model,
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

        self.modality_index_mapping = {modality: idx for idx, modality in enumerate(used_modalities)}
        self.resolution_index_mapping = {resolution: idx for idx, resolution in enumerate(used_resolutions)}

        # map modality to index
        self.df["modality_idx"] = self.df["modality"].map(self.modality_index_mapping)
        self.df["resolution_idx"] = self.df["resolution"].map(self.resolution_index_mapping)

        if max_subjects is not None:
            # self.df = self.df[self.df["subject_id"].isin(self.df["subject_id"].unique()[:max_subjects])]
            self.df = self.df.head(max_subjects)

    def get_data(self, split="train"):
        complete_df = self.df.copy()
        complete_df = complete_df[complete_df["split"] == split]
        
        self.subject_ids = complete_df["subject_id"].unique()

        instances = []
        for i, row in complete_df.iterrows():
            instance_dict = {}
            latent_path = row["latent_path"]
            # verify that the latent path exists
            if not os.path.exists(latent_path):
                # print(f"Latent path {latent_path} does not exist. Skipping this instance.")
                continue
            instance_dict["latent_path"] = latent_path
            instance_dict["subject_id"] = row["subject_id"]
            instance_dict["modality"] = row["modality"]
            instance_dict["modality_idx"] = row["modality_idx"]
            instance_dict["resolution"] = row["resolution"]
            instance_dict["resolution_idx"] = row["resolution_idx"]
            instance_dict["segmentation_npy_path"] = row["latent_seg_mask"]
            instances.append(instance_dict)
        return instances




class PrepareDataset(Dataset):
    def __init__(self, 
                 dataset_path_name,
                 used_modalities,
                 used_resolutions,
                 dataset_filters=None,
                 split="train",
                 max_subjects=None
                 ):

        # load data
        data_loader = LoadPaths(dataset_path_name,
                                    used_modalities=used_modalities,
                                    used_resolutions=used_resolutions,
                                  dataset_filters=dataset_filters,
                                    max_subjects=max_subjects
                                  )
        
        self.train_data = data_loader.get_data(split=split)

    
        print(f"Number of {split} images: {len(self.train_data)}")

        # number of latent in the folder
        self.num_instances = len(self.train_data) 
        self._length = self.num_instances
        # self.ids_list = list(self.train_data.keys())



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
        instance = self.train_data[index]

        example = {}
        example["subject_id"] = instance["subject_id"]

        instance_latent = np.load(instance["latent_path"])
        example["latent"] = torch.from_numpy(instance_latent)

        example["modality"] = instance["modality"]
        example["modality_idx"] = torch.tensor(instance["modality_idx"])
        example["resolution"] = instance["resolution"]
        example["resolution_idx"] = torch.tensor(instance["resolution_idx"])


        # segmentation_npy_path = instance["segmentation_npy_path"]
        # segmentation = np.load(segmentation_npy_path)
        # example["segmentation"] = torch.from_numpy(segmentation).long().unsqueeze(0)  # convert to long for one-hot encoding in the model

        example["segmentation"] = self.load_segmentation(instance["segmentation_npy_path"])
        return example
    

def collate_fn(examples):
    res_dict = {}

    res_dict["subject_id"] = [example["subject_id"] for example in examples]
    res_dict["modality"] = [example["modality"] for example in examples]


    modality_idx = torch.stack([example["modality_idx"] for example in examples])
    modality_idx = modality_idx.to(memory_format=torch.contiguous_format).long()
    res_dict["modality_idx"] = modality_idx

    latent = torch.stack([example["latent"] for example in examples])
    latent = latent.to(memory_format=torch.contiguous_format).float()
    res_dict["latent"] = latent    

    res_dict["resolution"]  = [example["resolution"] for example in examples]
    res_dict["resolution_idx"] = torch.stack([example["resolution_idx"] for example in examples])

    segmentation = torch.stack([example["segmentation"] for example in examples])
    segmentation = segmentation.to(memory_format=torch.contiguous_format).float()
    res_dict["segmentation"] = segmentation

    return res_dict


def instantiate_dataset(dataset_path_name, used_modalities, used_resolutions,
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
    # conditions_model,
    segmentation_encoder_model,
    modality_encoder_model,
    resolution_encoder_model,
    autoencoder,
    val_dataloader,
    step,
    args,

):

    print(f"Validation step {step}")
    modality_index_mapping = {modality: idx for idx, modality in enumerate(args.used_modalities)}
    resolution_index_mapping = {resolution: idx for idx, resolution in enumerate(args.used_resolutions)}

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

            segmentation_embedding = segmentation_encoder_model(batch["segmentation"].to(device))
            modality_embedding = modality_encoder_model(batch["modality_idx"].to(device))
            resolution_embedding = resolution_encoder_model(batch["resolution_idx"].to(device))

            all_timesteps = noise_scheduler.timesteps
            all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype)))
            progress_bar = tqdm(
                zip(all_timesteps, all_next_timesteps),
                total=min(len(all_timesteps), len(all_next_timesteps)),
                desc=f"Step {step} Modality {batch['modality']} Resolution {batch['resolution']} {c+1}/{total_images}"
            )
            c+=1
            with torch.no_grad(), torch.amp.autocast("cuda"):

                for t, next_t in progress_bar:

                    model_output = unet(
                        x=latents,
                        timesteps=torch.Tensor((t,)).to(device),
                        mask_features = segmentation_embedding,
                        modallity_embedding = modality_embedding,
                        resolution_embedding = resolution_embedding
                    )

                    if not isinstance(noise_scheduler, RFlowScheduler):
                        latents, _ = noise_scheduler.step(model_output, t, latents)
                    else:
                        latents, _ = noise_scheduler.step(model_output, t, latents, next_t)  # type: ignore

                # free memory for the autoencoder
                del model_output
                torch.cuda.empty_cache()
                
                # decode the latents to images
                synthetic_images = autoencoder.decode(latents, decode_complete=False)
                synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
                synthetic_images = synthetic_images.squeeze().numpy()

                # decode the latents to images
                # path_name_img = os.path.join(output_path_images, f"img_step_{step}_seed_{seed}_cond_{i}.nii.gz")
                # path_name_imgs_list.append(path_name_img)
                # nfc.save_nifti(synthetic_images, args.ref_aff, path_name_img)
                
                # if not evaluate:
                #     print("Skipping rest of the conditions and seeds because evaluate=False")
                #     break

            seed_img_list.append(synthetic_images)

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





def save_model(unet, segmentation_encoder_model, modality_encoder_model, resolution_encoder_model, optimizer, optimizer_segmentation_encoder, lr_scheduler, lr_scheduler_segmentation_encoder, global_step, out_model_path, ema=None, best=False):  # MOD: se añade parámetro ema
    # Guardar el modelo
    unet_state_dict = unet.module.state_dict() if torch.distributed.is_initialized() else unet.state_dict()
    checkpoint = {
        "unet_state_dict": unet_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_segmentation_encoder_state_dict": optimizer_segmentation_encoder.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "lr_scheduler_segmentation_encoder_state_dict": lr_scheduler_segmentation_encoder.state_dict() if lr_scheduler_segmentation_encoder is not None else None,
        "segmentation_encoder_model_state_dict": segmentation_encoder_model.state_dict(),
        "modality_encoder_model_state_dict": modality_encoder_model.state_dict(),
        "resolution_encoder_model_state_dict": resolution_encoder_model.state_dict(),
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
    
    del checkpoint, unet_state_dict
    torch.cuda.empty_cache()
    gc.collect()


def load_checkpoint(checkpoint_path, unet, segmentation_encoder_model, modality_encoder_model, resolution_encoder_model, device, train_dataloader_len,
                    gradient_accumulation_steps, batch_size, optimizer=None, optimizer_segmentation_encoder=None, lr_scheduler=None, lr_scheduler_segmentation_encoder=None, ema=None):
    # 1. Load checkpoint on CPU to avoid using VRAM
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Load weights into models
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
    segmentation_encoder_model.load_state_dict(checkpoint["segmentation_encoder_model_state_dict"], strict=False)
    modality_encoder_model.load_state_dict(checkpoint["modality_encoder_model_state_dict"], strict=False)
    resolution_encoder_model.load_state_dict(checkpoint["resolution_encoder_model_state_dict"], strict=False)

    # 3. Move models to GPU
    unet.to(device)
    segmentation_encoder_model.to(device)
    modality_encoder_model.to(device)
    resolution_encoder_model.to(device)

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

    if optimizer_segmentation_encoder is not None and "optimizer_segmentation_encoder_state_dict" in checkpoint:
        optimizer_segmentation_encoder.load_state_dict(checkpoint["optimizer_segmentation_encoder_state_dict"])
        # Move segmentation encoder optimizer buffers to GPU
        for state in optimizer_segmentation_encoder.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint and checkpoint["lr_scheduler_state_dict"] is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])


    if lr_scheduler_segmentation_encoder is not None and "lr_scheduler_segmentation_encoder_state_dict" in checkpoint and checkpoint["lr_scheduler_segmentation_encoder_state_dict"] is not None:
        lr_scheduler_segmentation_encoder.load_state_dict(checkpoint["lr_scheduler_segmentation_encoder_state_dict"])

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

    models_dict = instantiate_unconditioned_models(device, noise_scheduler_type=args_train.noise_scheduler_type)
    unet = models_dict["unet"]
    segmentation_encoder_model = models_dict["segmentation_encoder_model"]
    modality_encoder_model = models_dict["modality_encoder_model"]
    resolution_encoder_model = models_dict["resolution_encoder_model"]
    autoencoder = models_dict["autoencoder"]
    networks_config = models_dict["networks_config"]
    noise_scheduler = models_dict["noise_scheduler"]
    
    
    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        used_modalities=args_train.used_modalities,
        used_resolutions=args_train.used_resolutions,
        # latents_path=args_train.latents_path,
        batch_size=args_train.batch_size,
        split="train",
    )

    val_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        used_modalities=args_train.used_modalities,
        used_resolutions=args_train.used_resolutions,
        batch_size=1,
        split="val",
        max_subjects=args_train.max_val_subjects,
        dataset_filters = fc.args_to_dict(args_train.val_dataset_filters)
    
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
        list(unet.parameters()) + list(modality_encoder_model.parameters()) + list(resolution_encoder_model.parameters()),
        # list(unet.parameters()),
        lr=args_train.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    optimizer_segmentation_encoder = torch.optim.AdamW(
        segmentation_encoder_model.parameters(),
        lr=args_train.segmentation_encoder_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay= 1e-4
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
            
    if args_train.lr_scheduler_segmentation_encoder is not None:
        if args_train.lr_scheduler_segmentation_encoder.name == "PolynomialLR":
            lr_scheduler_segmentation_encoder = torch.optim.lr_scheduler.PolynomialLR(optimizer_segmentation_encoder, total_iters=args_train.max_train_steps, power=args_train.lr_scheduler_segmentation_encoder.power)
        elif args_train.lr_scheduler_segmentation_encoder.name == "CosineAnnealingLR":
            lr_scheduler_segmentation_encoder = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_segmentation_encoder, T_max=args_train.max_train_steps, eta_min=args_train.lr_scheduler_segmentation_encoder.eta_min)
        elif args_train.lr_scheduler_segmentation_encoder.name == "WarmupCosineLR":
            lr_scheduler_segmentation_encoder = create_warmup_cosine_scheduler(optimizer_segmentation_encoder,
                                                                               warmup_start_factor=args_train.lr_scheduler_segmentation_encoder.warmup_start_factor,
                                                                               warmup_steps=args_train.lr_scheduler_segmentation_encoder.warmup_steps,
                                                                               max_train_steps=args_train.max_train_steps,
                                                                               eta_min=args_train.lr_scheduler_segmentation_encoder.eta_min)

                                                
    else:
        lr_scheduler = None
        lr_scheduler_segmentation_encoder = None

    # ---- loss function
    loss_pt = torch.nn.MSELoss()

    # ---- training loop
    first_epoch = 0
    global_step = 0
    max_epochs = (args_train.max_train_steps * args_train.gradient_accumulation_steps * args_train.batch_size) // len(train_dataloader) + 1
    print(f"Max epochs: {max_epochs}")

    # ---- resume from checkpoint
    unet.to(device)
    segmentation_encoder_model.to(device)
    modality_encoder_model.to(device)
    resolution_encoder_model.to(device)

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
            unet,
            segmentation_encoder_model,
            modality_encoder_model,
            resolution_encoder_model,
            device=device,
            train_dataloader_len=len(train_dataloader),
            gradient_accumulation_steps=args_train.gradient_accumulation_steps,
            batch_size=args_train.batch_size,
            optimizer=optimizer,
            optimizer_segmentation_encoder=optimizer_segmentation_encoder,
            lr_scheduler=lr_scheduler,
            lr_scheduler_segmentation_encoder=lr_scheduler_segmentation_encoder,
            ema=ema
        )

    elif args_train.load_pretrained_model_from is not None:
        # checkpoint = torch.load(args_train.load_pretrained_model_from, weights_only=False, map_location=device_name)
        checkpoint = torch.load(args_train.load_pretrained_model_from, map_location=device_name)
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        segmentation_encoder_model.load_state_dict(checkpoint["segmentation_encoder_model_state_dict"], strict=False)
        modality_encoder_model.load_state_dict(checkpoint["modality_encoder_model_state_dict"], strict=False)
        resolution_encoder_model.load_state_dict(checkpoint["resolution_encoder_model_state_dict"], strict=False)
        if args_train.use_ema and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
            print("EMA state loaded from checkpoint")
        print(f"Pretrained model loaded from {args_train.load_pretrained_model_from}")
    
    unet.train()
    segmentation_encoder_model.train()  
    modality_encoder_model.train()
    resolution_encoder_model.train()

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
            latents = batch["latent"].to(device)
            condition_modality_idx = batch["modality_idx"].to(device)
            condition_resolution_idx = batch["resolution_idx"].to(device)

            # Forward pass
            with autocast("cuda", enabled=args_train.amp):
                # generate noise and timesteps with dedicate generatos and in the cpu for reproducibility
                noise = torch.randn(latents.shape, device="cpu", generator=gen_noise).to(device)
                if isinstance(noise_scheduler, RFlowScheduler):
                    timesteps = noise_scheduler.sample_timesteps(latents)
                else:
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device="cpu", generator=gen_t).long().to(device)

                noisy_latent = noise_scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
                
                segmentation_embedding = segmentation_encoder_model(batch["segmentation"].to(device))
                modality_embedding = modality_encoder_model(condition_modality_idx)
                resolution_embedding = resolution_encoder_model(condition_resolution_idx)

                # if batch['modality'][0] == "T1W":
                # print("-"*10, "Modality embedding", "-"*10)
                # print(f"modality: \n {batch['modality']} \n {batch['modality_idx']} \n {modality_embedding[:,::50]}")
                # print("-"*10, "Resolution embedding", "-"*10)
                # print(f"resolution: \n {batch['resolution']} \n {batch['resolution_idx']} \n {resolution_embedding[:,::50]}")


                model_output = unet(noisy_latent, 
                                  timesteps=timesteps,
                                #   context = volumetric_embedding,
                                    mask_features = segmentation_embedding,
                                    modallity_embedding = modality_embedding,
                                    resolution_embedding = resolution_embedding
                                    )
                

                if noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
                    # predict noise
                    model_gt = noise
                elif noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
                    # predict sample
                    model_gt = latents
                elif noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
                    # predict velocity
                    model_gt = latents - noise
                else:
                    raise ValueError(
                        "noise scheduler prediction type has to be chosen from ",
                        f"[{DDPMPredictionType.EPSILON},{DDPMPredictionType.SAMPLE},{DDPMPredictionType.V_PREDICTION}]",
                    )

                loss_noise = loss_pt(model_output.float(), model_gt.float())   # Dividir para escalar la pérdida

            loss = loss_noise / args_train.gradient_accumulation_steps  # Dividir la pérdida por los pasos de acumulación de gradientes

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
                    scaler.unscale_(optimizer_segmentation_encoder)
                    torch.nn.utils.clip_grad_norm_(
                        list(unet.parameters()) + list(segmentation_encoder_model.parameters()) + list(modality_encoder_model.parameters()) + list(resolution_encoder_model.parameters()),
                        max_norm=1.0
                    )
                           
                if args_train.amp:
                    scaler.step(optimizer)
                    scaler.step(optimizer_segmentation_encoder)
                    scaler.update()
                else:
                    optimizer.step()
                    optimizer_segmentation_encoder.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_segmentation_encoder.zero_grad(set_to_none=True)
                
                # update ema
                if args_train.use_ema:
                    ema.update(step=global_step)

                if lr_scheduler is not None:
                    lr_scheduler.step()
                if lr_scheduler_segmentation_encoder is not None:
                    lr_scheduler_segmentation_encoder.step()

                gradient_accumulation_count = 0  # Reiniciar el contador

                # update writter 
                if global_step % 10 == 0:
                    writer.add_scalar("Loss/train", loss.item(), global_step)
                    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("Time_steps", timesteps[0], global_step)

                # update progress bar 
                progress_bar.update(1)

                logs = {"0loss": loss.detach().item(), 
                        "1mod": [modality for modality in batch["modality"]],
                        "1id": [f'{__id[:4]}' for __id in batch["subject_id"]], 
                        # **logs_conditions,
                        }
                
                progress_bar.set_postfix(**logs)

                # update global step
                global_step += 1

                # save the model in intervals
                if global_step % args_train.save_checkpoint_interval == 0:
                    save_model(unet=unet, 
                               segmentation_encoder_model=segmentation_encoder_model, 
                               modality_encoder_model=modality_encoder_model, 
                               resolution_encoder_model=resolution_encoder_model, 
                               optimizer=optimizer,
                                optimizer_segmentation_encoder=optimizer_segmentation_encoder, 
                               lr_scheduler=lr_scheduler, 
                                lr_scheduler_segmentation_encoder=lr_scheduler_segmentation_encoder,
                               global_step=global_step, 
                               out_model_path=_checkpoint_dir_name, 
                               ema=ema)

                # Generar imágenes en intervalos
                if args_train.initial_val or global_step % args_train.val_interval == 0:
                    unet.eval()
                    segmentation_encoder_model.eval()
                    modality_encoder_model.eval()
                    resolution_encoder_model.eval()

                    if args_train.latents_shape is None:
                        args_train.latents_shape = latents.shape[-4:]  # save latents shape for later use in validation

                    try:
                        if args_train.use_ema:
                            ema.apply_shadow()
                        validation(unet=unet, 
                                   noise_scheduler=noise_scheduler, 
                                   segmentation_encoder_model=segmentation_encoder_model, 
                                      modality_encoder_model=modality_encoder_model,
                                      resolution_encoder_model=resolution_encoder_model,
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
                    unet.train()
                    segmentation_encoder_model.train()
                    modality_encoder_model.train()
                    resolution_encoder_model.train()
                    # conditions_model.train()

                if global_step >= args_train.max_train_steps:
                    break

        if global_step >= args_train.max_train_steps:
            break

    # make sure the progress bar closes
    progress_bar.close()

    # # make  out_model_path dir if it does not exist
    save_model(unet=unet, 
                segmentation_encoder_model=segmentation_encoder_model, 
                modality_encoder_model=modality_encoder_model, 
                resolution_encoder_model=resolution_encoder_model, 
                optimizer=optimizer,
                optimizer_segmentation_encoder=optimizer_segmentation_encoder, 
                lr_scheduler=lr_scheduler, 
                lr_scheduler_segmentation_encoder=lr_scheduler_segmentation_encoder,
                global_step=global_step, 
                out_model_path=_checkpoint_dir_name, 
                ema=ema)







args_train = {
    # directories 
    "output_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/test1_mask_features_opt2",
    "checkpoints_dir_name": "check_points",
    "logs_dir_name": "logs",
    "val_imgs_dir_name": "val_imgs",

    # data
    "df_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/pr_train_data.csv",
    "latents_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/train_data/preprocessed/maisi_latents",

    # training configuration
    "max_train_steps": 500000,
    "save_checkpoint_interval": 5000,#10000,

    # ---- memory reduction
    "amp": True,

    # ---- Training stability
    "batch_size": 6, #6 
    "gradient_accumulation_steps": 2,#2,
    "use_ema": True,
    "ema_params": {
        "decay": 0.999,
        "warm_up_steps": 2000,
        "warm_up_decay": 0.5,
    },

    # ---- optimizer
    "lr":  1e-4, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    "segmentation_encoder_lr": 5e-5, # 1e-4 for maisi, 1e-5 for blsmd
    # "lr":  1e-3, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    # "lr":  2.5e-5, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5

    # ---- lr_scheduler
    # "lr_scheduler": None,
    # "lr_scheduler": {"name": "PolynomialLR", "power": 2.0},
    # "lr_scheduler": {"name": "CosineAnnealingLR", "eta_min": 1e-6},
    "lr_scheduler": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-2, "warmup_steps": 500, "eta_min": 1e-6},
    "lr_scheduler_segmentation_encoder": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-4, "warmup_steps": 500, "eta_min": 1e-6},
    # "lr_scheduler": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-2, "warmup_steps": 25, "eta_min": 1e-6},


    # ---- pretrained_model
    # "load_pretrained_model_from": "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_training/no_synthsr/aaco5590_dataset_no_outliers_bfc/models/rflow/check_points/model_200000.pt",
    # "load_pretrained_model_from": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/test1/check_points/model_20.pt",
    "load_pretrained_model_from": None, # not working


    # ---- resume from checkpoint
    "resume_from_checkpoint_path_name": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/test1_mask_features_opt2/check_points/model_5000.pt",
    # "resume_from_checkpoint_path_name": None, # not working

    # reproducibility
    "seed": 42,

    # validation
    "val_interval": 500,
    "initial_val": False, # remember drop out
    # "validation_first": True, # if True, the model will be validated before the first training step, if False, the model will be validated after the first training step
    "val_seeds": [0],#[0,12357], # seeds for the noise generation during validation
    "max_val_subjects": None, # max number of subjects to be generated during validation, set to None to use all the subjects in the val dataloader
    "val_dataset_filters": {
        'subject_id': ["SP_T1W_3T_0006", "SP_T1W_5T_0006", "SP_T1W_7T_0006", 
                       "SP_T2W_3T_0006", "SP_T2W_5T_0006", "SP_T2W_7T_0006",
                       "SP_T2FLAIR_3T_0006", "SP_T2FLAIR_5T_0006", "SP_T2FLAIR_7T_0006",]
    },
    # specialied synthesis
    # "specialized_index": 1, # None for random, 0 for t1n, 1 for t1c, 2 for t2w, 3 for t2f
    
    "used_modalities": ["T1W", "T2W", "T2FLAIR"], # "T1W", "T2W", "T2FLAIR"
    "used_resolutions": [3, 5, 7], #0.1, 1.5, 3, 5, 7

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


