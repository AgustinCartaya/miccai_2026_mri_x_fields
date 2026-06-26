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


# pytorch
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist
from torch.nn import L1Loss, MSELoss
from monai.losses.adversarial_loss import PatchAdversarialLoss

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
from monai.networks.layers import Act, get_pool_layer

from networks_declaration.generator_01 import MultiResolutionGenerator
from networks_declaration.discriminator_01 import PatchDiscriminator

# images
from PIL import Image

sys.path.append("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils")
from autoencoder_declaration import AutoencoderPrediction
import prep_image as prep_image

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


def instantiate_unconditioned_models(
    device,
    nb_resolutions=5,
    resolution_emb_channels=None,
):

    networks_config = {
        "generator": {
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": 1,
            "num_channels": [
                32,
                64,
                128,
                # 256,
            ],
            "norm_num_groups": 16,
            "use_flash_attention": True,
            "nb_resolutions": nb_resolutions,
            "resolution_emb_channels": resolution_emb_channels,

        },
        "discriminator": {
            "spatial_dims": 3,
            "num_channels": 16,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers_d": 3,
            "kernel_size": 4,
            "activation": (Act.LEAKYRELU, {"negative_slope": 0.2}),
            "norm": "INSTANCE",
            "bias": False,
            "padding": 1,
            "dropout": 0.0,
            "last_conv_kernel_size": 4,
        },
    }


    # instantiate model

    args = fc.dict_to_args(networks_config, deep_conversion=True)

    # unet
    unet = MultiResolutionGenerator(
        spatial_dims=args.generator.spatial_dims,
        in_channels=args.generator.in_channels,
        out_channels=args.generator.out_channels,
        num_res_blocks=args.generator.num_res_blocks,
        num_channels=args.generator.num_channels,
        norm_num_groups=args.generator.norm_num_groups,
        use_flash_attention=args.generator.use_flash_attention,
        nb_resolutions=args.generator.nb_resolutions,
        resolution_emb_channels=args.generator.resolution_emb_channels,
    )

    # discriminator
    discriminator = PatchDiscriminator(
        spatial_dims=args.discriminator.spatial_dims,
        num_channels=args.discriminator.num_channels,
        in_channels=args.discriminator.in_channels,
        out_channels=args.discriminator.out_channels,
        num_layers_d=args.discriminator.num_layers_d,
        kernel_size=args.discriminator.kernel_size,
        activation=args.discriminator.activation,
        norm=args.discriminator.norm,
        bias=args.discriminator.bias,
        padding=args.discriminator.padding,
        dropout=args.discriminator.dropout,
        last_conv_kernel_size=args.discriminator.last_conv_kernel_size,
    )

    autoencoder_chekpoint_path = "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_vae/vae_weights/autoencoder_epoch273.pt"
    autoencoder = AutoencoderPrediction(autoencoder_chekpoint_path, device, half=True)

    return {
        "unet": unet,
        "discriminator": discriminator,
        "autoencoder": autoencoder,
        "networks_config": args,
    }


class LoadPaths:
    def __init__(
        self,
        dataset_path_name,
        used_modalities,
        used_resolutions,
        # nb_seg_classes,
        dataset_filters=None,
        max_subjects=None,
    ):
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

        self.df = self.df[
            (self.df["modality"].isin(used_modalities))
            & (self.df["resolution"].isin(used_resolutions))
        ]

        self.modality_index_mapping = {
            modality: idx for idx, modality in enumerate(used_modalities)
        }
        self.resolution_index_mapping = {
            resolution: idx for idx, resolution in enumerate(used_resolutions)
        }

        # map modality to index
        self.df["modality_idx"] = self.df["modality"].map(self.modality_index_mapping)
        self.df["resolution_idx"] = self.df["resolution"].map(self.resolution_index_mapping)

        if max_subjects is not None:
            # self.df = self.df[self.df["subject_id"].isin(self.df["subject_id"].unique()[:max_subjects])]
            self.df = self.df.head(max_subjects)

        # self.nb_seg_classes = nb_seg_classes

    def get_data(self, split="train"):
        complete_df = self.df.copy()
        if split is not None:
            complete_df = complete_df[complete_df["split"] == split]

        # order by [sid, resolution, modality, ]
        complete_df = complete_df.sort_values(by=["sid", "modality", "resolution"])

        # self.subject_ids = complete_df["sid"].unique()

        instances = []
        for i, row in complete_df.iterrows():
            instance_dict = {}
            latent_path = row["latent_normalized_wm_path"]
            # verify that the latent path exists
            if not os.path.exists(latent_path):
                # print(f"Latent path {latent_path} does not exist. Skipping this instance.")
                continue
            instance_dict["latent_path"] = latent_path
            instance_dict["latent_synthsr_path"] = row["latent_synthsr_path"]
            instance_dict["subject_id"] = row["sid"]
            instance_dict["modality"] = row["modality"]
            instance_dict["modality_idx"] = row["modality_idx"]
            instance_dict["resolution"] = row["resolution"]
            instance_dict["resolution_idx"] = row["resolution_idx"]
            # instance_dict["segmentation_npy_path"] = row[
            #     f"latent_seg_supersynth_merged_{self.nb_seg_classes}_path"
            # ]
            instance_dict["org_img_path"] = row["org_img_path"]
            
            instances.append(instance_dict)
        return instances


class PrepareDataset(Dataset):
    def __init__(
        self,
        dataset_path_name,
        used_modalities,
        used_resolutions,
        dataset_filters=None,
        split="train",
        max_subjects=None,
        # nb_seg_classes=3,
        # used_pondered_segmentation=False,
        # pondered_seg_generator=None,
        # latent_synthsr_generator=None,
        # latent_synthsr_prob=None
    ):

        # load data
        data_loader = LoadPaths(
            dataset_path_name,
            used_modalities=used_modalities,
            used_resolutions=used_resolutions,
            dataset_filters=dataset_filters,
            max_subjects=max_subjects,
            # nb_seg_classes=nb_seg_classes,
        )

        self.train_data = data_loader.get_data(split=split)

        print(f"Number of {split} images: {len(self.train_data)}")

        # number of latent in the folder
        self.num_instances = len(self.train_data)
        self._length = self.num_instances
        # self.nb_seg_classes = nb_seg_classes
        self.used_modalities = used_modalities
        # self.use_pondered_segmentation = used_pondered_segmentation
        # self.pondered_seg_generator = pondered_seg_generator
        self.split = split
        # self.ids_list = list(self.train_data.keys())

        # self.latent_synthsr_generator = latent_synthsr_generator
        # self.latent_synthsr_prob = latent_synthsr_prob
    def __len__(self):
        return self._length

    # def load_segmentation(self, segmentation_npy_path):
    #     segmentation = np.load(segmentation_npy_path)
    #     # convert into one-hot encoding
    #     # unique_labels = np.unique(segmentation)
    #     # remove zero if it is in the unique labels (assuming zero is the background)
    #     # if 0 in unique_labels:
    #     #     unique_labels = unique_labels[unique_labels != 0]

    #     unique_labels = list(
    #         range(1, self.nb_seg_classes + 1)
    #     )  # assuming classes are labeled from 1 to nb_classes

    #     seg_onehot = []
    #     for label in unique_labels:
    #         seg_onehot.append(np.where(segmentation == label, 1.0, 0.0))
    #     seg_onehot = np.stack(seg_onehot, axis=0)  # (C, D, H, W)

    #     return torch.from_numpy(seg_onehot).float()

    # def load_pondered_segmentation(
    #     self, segmentation_npy_path, current_modality, subject_id
    # ):
    #     seg_list = []
    #     for mod in self.used_modalities:
    #         modified_path = segmentation_npy_path
    #         if mod != current_modality:
    #             modified_path = segmentation_npy_path.replace(current_modality, mod)
    #         # verify that the modified path exists
    #         if os.path.exists(modified_path):
    #             seg_list.append(self.load_segmentation(modified_path))
    #             # print(f"Loaded segmentation from {modified_path} for modality {mod}")

    #     # create weighted sum of the segmentations using random weights that sum to 1
    #     if len(seg_list) > 1:
    #         # if self.split == "train":
    #         # print(f"{subject_id} nb segmentations {len(seg_list)} for modality {current_modality}")
    #         if self.pondered_seg_generator is not None:
    #             gamma = torch.empty(len(seg_list)).exponential_(
    #                 1.0, generator=self.pondered_seg_generator
    #             )
    #         else:
    #             gamma = torch.ones(len(seg_list))
    #         weights = gamma / gamma.sum()

    #         seg_pondered = torch.zeros_like(seg_list[0])
    #         for seg, w in zip(seg_list, weights):
    #             seg_pondered += w * seg
    #     else:
    #         seg_pondered = seg_list[0]
    #     return seg_pondered

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
        example["org_img_path"] = instance["org_img_path"]

        # segmentation_npy_path = instance["segmentation_npy_path"]
        # segmentation = np.load(segmentation_npy_path)
        # example["segmentation"] = torch.from_numpy(segmentation).long().unsqueeze(0)  # convert to long for one-hot encoding in the model

        # if self.use_pondered_segmentation:
        #     example["segmentation"] = self.load_pondered_segmentation(
        #         instance["segmentation_npy_path"],
        #         instance["modality"],
        #         example["subject_id"],
        #     )
        # else:
        #     example["segmentation"] = self.load_segmentation(
        #         instance["segmentation_npy_path"]
            # )

        # # instance_latent_synthsr = np.load(instance["latent_synthsr_path"])
        # # example["latent_synthsr"] = torch.from_numpy(instance_latent_synthsr)
        
        # # load latent_synthsr depending th probability of each class and resolution
        # # max_prob_latent_synthsr = self.latent_synthsr_prob[instance["modality"]][instance["resolution"]]
        # load_latent_synthsr = True
        # # generate a random number between 0 and 1 using the latent_synthsr_generator
        # if self.latent_synthsr_generator is not None:
        #     rand_num = torch.rand(1, generator=self.latent_synthsr_generator).item()
        #     load_latent_synthsr = rand_num < self.latent_synthsr_prob
        # if load_latent_synthsr:
        #     instance_latent_synthsr = np.load(instance["latent_synthsr_path"])
        #     example["latent_synthsr"] = torch.from_numpy(instance_latent_synthsr)
        # else:
        #     example["latent_synthsr"] = torch.zeros_like(example["latent"])

        return example


def collate_fn(examples):
    res_dict = {}

    res_dict["subject_id"] = [example["subject_id"] for example in examples]
    res_dict["modality"] = [example["modality"] for example in examples]
    res_dict["org_img_path"] = [example["org_img_path"] for example in examples]

    modality_idx = torch.stack([example["modality_idx"] for example in examples])
    modality_idx = modality_idx.to(memory_format=torch.contiguous_format).long()
    res_dict["modality_idx"] = modality_idx

    latent = torch.stack([example["latent"] for example in examples])
    latent = latent.to(memory_format=torch.contiguous_format).float()
    res_dict["latent"] = latent
    
    # latent_synthsr = torch.stack([example["latent_synthsr"] for example in examples])
    # latent_synthsr = latent_synthsr.to(memory_format=torch.contiguous_format).float()
    # res_dict["latent_synthsr"] = latent_synthsr


    res_dict["resolution"] = [example["resolution"] for example in examples]
    res_dict["resolution_idx"] = torch.stack(
        [example["resolution_idx"] for example in examples]
    )

    # segmentation = torch.stack([example["segmentation"] for example in examples])
    # segmentation = segmentation.to(memory_format=torch.contiguous_format).float()
    # res_dict["segmentation"] = segmentation

    return res_dict


def instantiate_dataset(
    dataset_path_name,
    used_modalities,
    used_resolutions,
    batch_size,
    dataset_filters=None,
    split="train",
    max_subjects=None,
    # nb_seg_classes=3,
    # used_pondered_segmentation=False,
    # pondered_seg_generator=None,
    # latent_synthsr_generator=None,
    # latent_synthsr_prob=None
):
    # ---- Data set creation
    train_dataset = PrepareDataset(
        dataset_path_name=dataset_path_name,
        used_modalities=used_modalities,
        used_resolutions=used_resolutions,
        dataset_filters=dataset_filters,
        split=split,
        max_subjects=max_subjects,
        # nb_seg_classes=nb_seg_classes,
        # used_pondered_segmentation=used_pondered_segmentation,
        # pondered_seg_generator=pondered_seg_generator,
        # latent_synthsr_generator=latent_synthsr_generator,
        # latent_synthsr_prob=latent_synthsr_prob
    )

    # sampler = MaxPerSubjectSampler(train_dataset, max_per_subject=max_timepoints_per_epoch, shuffle=True, generator=gen_dataloader)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=split == "train",  # shuffle only in training
        # sampler=sampler,
        collate_fn=lambda examples: collate_fn(examples),
        # generator=gen_dataloader,
        num_workers=2,
        persistent_workers=True,
    )
    return train_dataloader


Z_CLIP_RANGE = (150, 180)
def compute_image_metrics(synthetic_image, org_image, crop_z=True):
    if crop_z:
        synthetic_image = synthetic_image[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]
        org_image = org_image[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]

    ssim = util.compute_ssim(synthetic_image, org_image)
    rmse = util.compute_RMS(synthetic_image, org_image)
    return {"ssim": ssim, "rmse": rmse}


def save_2D_images(image_list, image_path_name):
    # for name_save, imgs_list in zip(name_list, list_to_save):
    imgs_list_2D = fc.cat_n_views_different_layers(
        image_list,
        view_layersoffset_list=[(2, 0), (2, -15), (1, 0), (0, 10), (0, 0)],
        axis=0,
        img_cropping=50,
        to_rgb=True,
    )
    # save synthetic images
    complete_img = np.concatenate(imgs_list_2D, axis=1)
    complete_img = Image.fromarray(complete_img)
    # complete_img.save(f"{output_path_2D_images}/imgs2D_step_{step}_subject_{batch['subject_id'][0]}.png")
    complete_img.save(image_path_name)





@torch.no_grad()
def validation(
    unet,
    autoencoder,
    val_dataloader,
    step,
    args,
):

    print(f"Validation step {step}")

    output_path_step = os.path.join(
        args.output_path, args.val_imgs_dir_name, f"step_{step}"
    )
    output_path_2D_images = os.path.join(output_path_step, "images2d")
    output_path_metrics = os.path.join(output_path_step, "metrics")

    for _output_path in [output_path_2D_images, output_path_metrics]:
        os.makedirs(_output_path, exist_ok=True)

    # total_images = len(val_dataloader)
    # c = 0
    # val_results = []
    # subject_img_list = []
    
    # prev_sid = None
    # prev_modality = None
    # save_subject_img_list = False

    org_img_list = []
    resolution_results = [None] * len(args.used_resolutions)
    bar = tqdm(val_dataloader, desc=f"Validation step {step}", unit="batch")
    # for batch in val_dataloader:
    for batch in bar:

        latents = batch["latent"].to(device)
        resolution_idx = batch["resolution_idx"].to(device)
        
        
        with torch.no_grad(), torch.amp.autocast("cuda"):
            model_output = unet(latents, cond=resolution_idx)

            pred_images = []
            for i, res in enumerate(args.used_resolutions):
                synthetic_images = autoencoder.decode(model_output[i], decode_complete=True)
                synthetic_images = torch.clip(synthetic_images, 0.0, 1.0).cpu()
                synthetic_images = synthetic_images.squeeze().numpy()
                pred_images.append(synthetic_images)

        resolution_results[resolution_idx.item()] = pred_images

        org_img = nfc.load_nifti(batch["org_img_path"][0])[0]
        org_img = prep_image.prepare_img(org_img)
        org_img_list.append(org_img)
        

    org_img_np = np.stack(org_img_list, axis=0) # [res, H, W, D] = [5, H, W, D]
    resolution_results_np = [np.stack(res, axis=0) for res in resolution_results] # [res, res_pred, H, W, D] = [5, 5, H, W, D]
    
    # compute metrics and save images for each resolution
    val_results = []
    for i, res in enumerate(args.used_resolutions):
        for j, res_pred in enumerate(args.used_resolutions):
            metrics_dict = compute_image_metrics(resolution_results_np[j][i], org_img_np[i])
            results_dict = {
                "sid": batch["subject_id"][0],
                "modality": batch["modality"][0],
                "src_resolution": res,
                "pred_resolution": res_pred,
                "ssim": metrics_dict["ssim"],
                "rmse": metrics_dict["rmse"],

            }
            val_results.append(results_dict)

            # image_path_name = os.path.join(
            #     output_path_2D_images,
            #     f"imgs2D_step_{step}_{batch['subject_id'][0]}_{batch['modality'][0]}_src{res}_pred{res_pred}.png"
            # )
            # save_2D_images(
            # [org_img_np[i], resolution_results_np[j][i]],
            # image_path_name
            # )
    

        # save images for each resolution
        image_path_name = os.path.join(
            output_path_2D_images,
            f"imgs2D_step_{step}_{batch['subject_id'][0]}_{batch['modality'][0]}_{res}.png"
        )
        save_2D_images(
            [org_img_np[i]] + [_im for _im in resolution_results_np[i]],
            image_path_name
        )
    


    # save the results to a csv file
    df_val_results = pd.DataFrame(val_results)
    df_val_results.to_csv(f"{output_path_metrics}/metrics_step_{step}.csv", index=False)

    # clean memory
    del resolution_results, resolution_results_np, org_img_list, org_img_np
    torch.cuda.empty_cache()
    return df_val_results


# def save_model(unet, segmentation_encoder_model, modality_encoder_model, resolution_encoder_model, optimizer, optimizer_segmentation_encoder, lr_scheduler, lr_scheduler_segmentation_encoder, global_step, out_model_path, ema=None, best=False):  # MOD: se añade parámetro ema
def save_model(
    unet,
    discriminator,
    optimizer_g,
    optimizer_d,
    lr_scheduler_g,
    lr_scheduler_d,
    global_step,
    out_model_path,
    ema=None,
    best=False,
):  # MOD: se añade parámetro ema
    # Guardar el modelo
    unet_state_dict = (
        unet.module.state_dict()
        if torch.distributed.is_initialized()
        else unet.state_dict()
    )
    checkpoint = {
        "generator_state_dict": unet_state_dict,
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "optimizer_d_state_dict": optimizer_d.state_dict(),
        "lr_scheduler_g_state_dict": (
            lr_scheduler_g.state_dict() if lr_scheduler_g is not None else None
        ),
        "lr_scheduler_d_state_dict": (
            lr_scheduler_d.state_dict() if lr_scheduler_d is not None else None
        ),
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


# def load_checkpoint(checkpoint_path, unet, segmentation_encoder_model, modality_encoder_model, resolution_encoder_model, device, train_dataloader_len,
#                     gradient_accumulation_steps, batch_size, optimizer=None, optimizer_segmentation_encoder=None, lr_scheduler=None, lr_scheduler_segmentation_encoder=None, ema=None):


def load_checkpoint(
    checkpoint_path,
    unet,
    discriminator,
    device,
    train_dataloader_len,
    gradient_accumulation_steps,
    batch_size,
    optimizer_g=None,
    optimizer_d=None,
    lr_scheduler_g=None,
    lr_scheduler_d=None,
    ema=None,
):

    # 1. Load checkpoint on CPU to avoid using VRAM
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Load weights into models
    unet.load_state_dict(checkpoint["generator_state_dict"], strict=False)
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)

    # 3. Move models to GPU
    unet.to(device)
    discriminator.to(device)

    # 4. EMA (optional)
    if ema is not None and "ema_state_dict" in checkpoint:
        # Move only the EMA tensors to GPU
        ema.shadow = {k: v.to(device) for k, v in checkpoint["ema_state_dict"].items()}

    # 5. Optimizer and scheduler (optional)
    if optimizer_g is not None and "optimizer_g_state_dict" in checkpoint:
        # For Adam, some states may be on CPU and others on GPU
        optimizer_g.load_state_dict(checkpoint["optimizer_g_state_dict"])
        # Move optimizer buffers to GPU
        for state in optimizer_g.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    if optimizer_d is not None and "optimizer_d_state_dict" in checkpoint:
        optimizer_d.load_state_dict(checkpoint["optimizer_d_state_dict"])
        # Move optimizer buffers to GPU
        for state in optimizer_d.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # if optimizer_segmentation_encoder is not None and "optimizer_segmentation_encoder_state_dict" in checkpoint:
    #     optimizer_segmentation_encoder.load_state_dict(checkpoint["optimizer_segmentation_encoder_state_dict"])
    #     # Move segmentation encoder optimizer buffers to GPU
    #     for state in optimizer_segmentation_encoder.state.values():
    #         for k, v in state.items():
    #             if isinstance(v, torch.Tensor):
    #                 state[k] = v.to(device)

    if (
        lr_scheduler_g is not None
        and "lr_scheduler_g_state_dict" in checkpoint
        and checkpoint["lr_scheduler_g_state_dict"] is not None
    ):
        lr_scheduler_g.load_state_dict(checkpoint["lr_scheduler_g_state_dict"])

    if lr_scheduler_d is not None and "lr_scheduler_d_state_dict" in checkpoint and checkpoint["lr_scheduler_d_state_dict"] is not None:
        lr_scheduler_d.load_state_dict(checkpoint["lr_scheduler_d_state_dict"])

    # 6. Compute first_epoch and global_step
    global_step = checkpoint["num_train_timesteps"]
    first_epoch = (
        global_step * gradient_accumulation_steps * batch_size
    ) // train_dataloader_len + 1

    print(f"Model loaded from {checkpoint_path}")
    print(f"Resuming training from epoch {first_epoch} and global step {global_step}")

    return global_step, first_epoch


def save_configurations(
    args_train, networks_config, config_path, config_name="model_config.json"
):
    argparse_dict = {
        "args_train": fc.args_to_dict(args_train, deep_conversion=True),
        "networks_config": fc.args_to_dict(networks_config, deep_conversion=True),
    }
    argparse_json = json.dumps(argparse_dict, indent=4)
    with open(os.path.join(config_path, config_name), "w") as outfile:
        outfile.write(argparse_json)
    print(f"Model configurations saved in: {os.path.join(config_path, config_name)}")


class EMA:
    def __init__(
        self, model, decay, warm_up_steps=0, warm_up_decay=0.1, optimize_cpu=False
    ):
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
                    new_avg = (
                        1.0 - decay
                    ) * param.detach().cpu() + decay * self.shadow[name]
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


def create_warmup_cosine_scheduler(
    optimizer, warmup_start_factor, warmup_steps, max_train_steps, eta_min
):
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
        total_iters=warmup_steps,
    )

    # Cosine decay scheduler
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_train_steps - warmup_steps, eta_min=eta_min
    )

    # Combine them
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler


def train(
    args_train,
    device,
):

    # ---- reproducibility
    set_seed(args_train.seed)
    # gen_t = torch.Generator().manual_seed(args_train.seed)
    # gen_noise = torch.Generator().manual_seed(args_train.seed)
    # gen_pondered_seg = torch.Generator().manual_seed(args_train.seed)
    # gen_free_guidance = torch.Generator().manual_seed(args_train.seed)
    # gen_synthsr = torch.Generator().manual_seed(args_train.seed)

    models_dict = instantiate_unconditioned_models(
        device,
        nb_resolutions=len(args_train.used_resolutions),
        resolution_emb_channels=args_train.resolution_emb_channels,
    )

    unet = models_dict["unet"]
    discriminator = models_dict["discriminator"]
    autoencoder = models_dict["autoencoder"]
    networks_config = models_dict["networks_config"]


    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        used_modalities=args_train.used_modalities,
        used_resolutions=args_train.used_resolutions,
        batch_size=args_train.batch_size, 
        split="train",
    )

    val_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        used_modalities=args_train.used_modalities,
        used_resolutions=args_train.used_resolutions,
        batch_size=1,
        split="val",
        # split=None,
        max_subjects=args_train.max_val_subjects,
        dataset_filters=fc.args_to_dict(args_train.val_dataset_filters),
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
    timestamp = datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M"
    )  # Formato: Año-Mes-Día_Hora-Minuto
    _sum_writter_dir = os.path.join(_logs_dir_name, f"logs_{timestamp}")
    os.makedirs(_sum_writter_dir, exist_ok=True)
    writer = SummaryWriter(_sum_writter_dir)

    # ---- optimizer and lr_scheduler
    optimizer_g = torch.optim.AdamW(
        list(unet.parameters()),
        lr=args_train.lr_g,
        # betas=(0.9, 0.999),
        # eps=1e-06 if args_train.amp else 1e-08,
        # weight_decay=0.01,
    )

    optimizer_d = torch.optim.AdamW(
        list(discriminator.parameters()),
        lr=args_train.lr_d,
        # betas=(0.9, 0.999),
        # eps=1e-06 if args_train.amp else 1e-08,
        # weight_decay=0.01,
    )

    def create_lr_scheduler(optimizer, lr_scheduler_config):
        if lr_scheduler_config is None:
            return None
        if lr_scheduler_config.name == "PolynomialLR":
            return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args_train.max_train_steps, power=lr_scheduler_config.power)
        elif lr_scheduler_config.name == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_train.max_train_steps, eta_min=lr_scheduler_config.eta_min)
        elif lr_scheduler_config.name == "WarmupCosineLR":
            return create_warmup_cosine_scheduler(optimizer,
                                                warmup_start_factor=lr_scheduler_config.warmup_start_factor,
                                                warmup_steps=lr_scheduler_config.warmup_steps,
                                                max_train_steps=args_train.max_train_steps,
                                                eta_min=lr_scheduler_config.eta_min)
        else:
            return None
        
    lr_scheduler_g = create_lr_scheduler(optimizer_g, args_train.lr_scheduler_g)
    lr_scheduler_d = create_lr_scheduler(optimizer_d, args_train.lr_scheduler_d)

    # ---- loss function
    recon_loss_pt = torch.nn.MSELoss()
    adv_loss_pt = PatchAdversarialLoss(criterion="least_squares")
    class_loss_pt = torch.nn.CrossEntropyLoss()

    # ---- training loop
    first_epoch = 0
    global_step = 0
    max_epochs = (
        args_train.max_train_steps
        * args_train.gradient_accumulation_steps
        * args_train.batch_size
    ) // len(train_dataloader) + 1
    print(f"Max epochs: {max_epochs}")

    # ---- resume from checkpoint
    unet.to(device)
    discriminator.to(device)

    # Initilize ema
    if args_train.use_ema:
        ema = EMA(
            unet,
            decay=args_train.ema_params.decay,
            warm_up_steps=args_train.ema_params.warm_up_steps,
            warm_up_decay=args_train.ema_params.warm_up_decay,
            optimize_cpu=False,
        )
    else:
        ema = None

    # priority is to resume from check point
    if args_train.resume_from_checkpoint_path_name is not None:
        global_step, first_epoch = load_checkpoint(
            args_train.resume_from_checkpoint_path_name,
            unet,
            discriminator,
            device=device,
            train_dataloader_len=len(train_dataloader),
            gradient_accumulation_steps=args_train.gradient_accumulation_steps,
            batch_size=args_train.batch_size,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            lr_scheduler_g=lr_scheduler_g,
            lr_scheduler_d=lr_scheduler_d,
            ema=ema,
        )

    elif args_train.load_pretrained_model_from is not None:
        # checkpoint = torch.load(args_train.load_pretrained_model_from, weights_only=False, map_location=device_name)
        checkpoint = torch.load(
            args_train.load_pretrained_model_from, map_location=device_name
        )
        unet.load_state_dict(checkpoint["generator_state_dict"], strict=False)
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"], strict=False)
        if args_train.use_ema and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
            print("EMA state loaded from checkpoint")
        print(f"Pretrained model loaded from {args_train.load_pretrained_model_from}")

    unet.train()
    discriminator.train()


    # ---- memory reduction
    # -------- automatic mixed precision
    if args_train.amp:
        scaler_g = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
        scaler_d = GradScaler("cuda", init_scale=2.0**8, growth_factor=1.5)
    else:
        scaler_g = None
        scaler_d = None
    gradient_accumulation_count = 0

    # ---- training loop
    progress_bar = tqdm(
        range(0, args_train.max_train_steps), desc="Training", initial=global_step
    )


    def compute_mean_res_loss(res_list, res_weights):
        mean_res_loss = 0.0
        for res_loss, res_weight in zip(res_list, res_weights):
            mean_res_loss += res_loss * res_weight
        return mean_res_loss

    # early stopping variables
    best_val_loss = float("inf")
    for epoch in range(first_epoch, max_epochs):
        for batch in train_dataloader:

            # prepare inputs
            latents = batch["latent"].to(device)
            condition_resolution_idx = batch["resolution_idx"].to(device)
            
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

            # Forward pass
            with autocast("cuda", enabled=args_train.amp):

                # ------------ Train Generator
                model_output = unet(
                    latents, cond=condition_resolution_idx
                )

                # model_output is a list of tensors, convert it to a single tensor of size [n_resolutions, batch_size, channels, height, width, depth]
                model_output = torch.stack(model_output, dim=0)
                # exchange the first two dimensions to have [batch_size, n_resolutions, channels, height, width, depth]
                model_output = model_output.permute(1, 0, 2, 3, 4, 5)
                
                # rec_latents = []
                # for __i, _batch_res_idx in enumerate(condition_resolution_idx):
                #     rec_latents.append(model_output[_batch_res_idx][__i:__i+1])

                rec_latents = model_output[torch.arange(model_output.size(0)), condition_resolution_idx]
                recon_loss = recon_loss_pt(rec_latents.float(), latents.float())

                # ---------------- CYCLE CONSISTENCY ----------------
                num_res = len(args_train.used_resolutions)

                target_resolution_idx = torch.randint(
                    low=0,
                    high=num_res,
                    size=condition_resolution_idx.shape,
                    device=device
                )

                # x -> r_j
                fake_latents = model_output[torch.arange(model_output.size(0)), target_resolution_idx]

                # r_j -> x_hat
                cycle_output = unet(fake_latents, cond=target_resolution_idx)
                cycled_latents = cycle_output[torch.arange(model_output.size(0)), condition_resolution_idx]

                cycle_loss = recon_loss_pt(cycled_latents.float(), latents.float())
                # ---------------------------------------------------


                logits_fake_loss_list = []
                logits_class_loss_list = []
                ### PRONE TO ERRO IF IT IS NOT ACORDING TO THE RESOLUTION INDEX, NEEDS TO BE CHECKED
                for __res_idx in range(len(args_train.used_resolutions)):
                    disc_output = discriminator(model_output[:, __res_idx])
                    logits_fake_loss_list.append(adv_loss_pt(disc_output["patch_logits"], target_is_real=True, for_discriminator=False))

                    # repeat __res_idx to match the batch size and compute the class loss for each resolution
                    __res_idx_batch = torch.full((model_output.size(0),), __res_idx, dtype=torch.long, device=device)
                    logits_class_loss_list.append(class_loss_pt(disc_output["class_logits"], __res_idx_batch))

                logits_fake_loss = compute_mean_res_loss(logits_fake_loss_list, args_train.loss_weights.logits_fake_resolution_weights)
                logits_class_loss = compute_mean_res_loss(logits_class_loss_list, args_train.loss_weights.logits_class_resolution_weights)

                loss_g = (
                    recon_loss
                    + args_train.loss_weights.logits_fake_loss_weight * logits_fake_loss
                    + args_train.loss_weights.logits_class_loss_weight  * logits_class_loss
                    + args_train.loss_weights.cycle_loss_weight * cycle_loss
                    ) 


                if args_train.amp:
                    scaler_g.scale(loss_g).backward()
                    scaler_g.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(
                        list(unet.parameters()),
                        max_norm=1.0,
                    )
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                else:
                    loss_g.backward()
                    optimizer_g.step()

                if args_train.use_ema:
                    ema.update(step=global_step)
            
                if lr_scheduler_g is not None:
                    lr_scheduler_g.step()

                # ------------ Train Discriminator
                disc_output_fake = discriminator(rec_latents.contiguous().detach())
                loss_d_fake = adv_loss_pt(disc_output_fake["patch_logits"], target_is_real=False, for_discriminator=True)
                # loss_d_class_fake = class_loss_pt(disc_output_fake["class_logits"], condition_resolution_idx)

                disc_output_real = discriminator(latents.contiguous().detach())
                loss_d_real = adv_loss_pt(disc_output_real["patch_logits"], target_is_real=True, for_discriminator=True)
                # print(disc_output_real["class_logits"])
                # print(condition_resolution_idx)
                loss_d_class_real = class_loss_pt(disc_output_real["class_logits"], condition_resolution_idx)

                loss_d = (
                    0.5 * (loss_d_fake + loss_d_real)
                    + args_train.loss_weights.disc_class_loss_weight * (loss_d_class_real)
                )

                if args_train.amp:
                    scaler_d.scale(loss_d).backward()
                    # scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(
                        list(discriminator.parameters()),
                        max_norm=1.0,
                    )
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                else:
                    loss_d.backward()
                    optimizer_d.step()

                if lr_scheduler_d is not None:
                    lr_scheduler_d.step()

            # update writter
            if global_step % 10 == 0:
                writer.add_scalar("Loss_g/total", loss_g.detach().item(), global_step)
                writer.add_scalar("Loss_g/recon", recon_loss.detach().item(), global_step)
                writer.add_scalar("Loss_g/logits_fake", logits_fake_loss.detach().item(), global_step)
                writer.add_scalar("Loss_g/logits_class", logits_class_loss.detach().item(), global_step)
                writer.add_scalar("Loss_d/total", loss_d.detach().item(), global_step)
                writer.add_scalar("Loss_d/fake", loss_d_fake.detach().item(), global_step)
                writer.add_scalar("Loss_d/real", loss_d_real.detach().item(), global_step)
                writer.add_scalar("Loss_d/class_real", loss_d_class_real.detach().item(), global_step)
                ### learning rate
                writer.add_scalar("Lr_g", optimizer_g.param_groups[0]["lr"], global_step)
                writer.add_scalar("Lr_d", optimizer_d.param_groups[0]["lr"], global_step)


            # update progress bar
            progress_bar.update(1)

            logs = {
                "0lg": loss_g.detach().item(),
                # "0lgr": recon_loss.detach().item(),
                # "0lgf": logits_fake_loss.detach().item(),
                # "0lgc": logits_class_loss.detach().item(),
                "1ld": loss_d.detach().item(),
                "2res": [f"{__resolution}" for __resolution in batch["resolution"]][0],
                "2rid": [f"{__resolution_idx}" for __resolution_idx in batch["resolution_idx"]][0],
                "3id": [f"{__id[:4]}" for __id in batch["subject_id"]][0],
                # **logs_conditions,
            }

            progress_bar.set_postfix(**logs)

            # update global step
            global_step += 1

            # save the model in intervals
            if global_step % args_train.save_checkpoint_interval == 0:
                save_model(
                    unet=unet,
                    discriminator=discriminator,
                    optimizer_g=optimizer_g,
                    optimizer_d=optimizer_d,
                    lr_scheduler_g=lr_scheduler_g,
                    lr_scheduler_d=lr_scheduler_d,
                    global_step=global_step,
                    out_model_path=_checkpoint_dir_name,
                    ema=ema,
                )

            # Generar imágenes en intervalos
            if args_train.initial_val or global_step % args_train.val_interval == 0:
                unet.eval()
                discriminator.eval()

                if True:
                # try:
                    if args_train.use_ema:
                        ema.apply_shadow()

                    df_val_results =  validation(
                        unet=unet,
                        autoencoder=autoencoder,
                        val_dataloader=val_dataloader,
                        step=global_step,
                        args=args_train,
                    )
                    
                    # early stop and save best model
                    if not args_train.initial_val and df_val_results is not None:
                        # get ssi mean and rmse mean
                        ssim_mean = df_val_results["ssim"].mean()
                        rmse_mean = df_val_results["rmse"].mean()

                        writer.add_scalar("Val/ssim", ssim_mean, global_step)
                        writer.add_scalar("Val/rmse", rmse_mean, global_step)

                        ssim_loss = 1 - ssim_mean
                        if ssim_loss < best_val_loss:
                            best_val_loss = ssim_loss
                            # patience_counter = 0
                            print(f"Validation loss improved at step {global_step}: {ssim_loss}")
                            save_model(
                                unet=unet,
                                discriminator=discriminator,
                                optimizer_g=optimizer_g,
                                optimizer_d=optimizer_d,
                                lr_scheduler_g=lr_scheduler_g,
                                lr_scheduler_d=lr_scheduler_d,
                                global_step=global_step,
                                out_model_path=_checkpoint_dir_name,
                                ema=ema,
                                best=True
                            )
                try:
                    pass
                except Exception as e:
                    print(f"ERROR DURING VALIDATION STEP {global_step}: {e}")
                finally:
                    if args_train.use_ema:
                        ema.restore()

                args_train.initial_val = False
                unet.train()
                discriminator.train()

            if global_step >= args_train.max_train_steps:
                break

        if global_step >= args_train.max_train_steps:
            break

    # make sure the progress bar closes
    progress_bar.close()

    # # make  out_model_path dir if it does not exist
    save_model(
        unet=unet,
        discriminator=discriminator,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        global_step=global_step,
        out_model_path=_checkpoint_dir_name,
        ema=ema,
    )


args_train = {
    # directories
    "output_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test7_my_own_net/training/models/singlemod/t1w/singlemod_gen01_disc01_cycleloss",
    "checkpoints_dir_name": "check_points",
    "logs_dir_name": "logs",
    "val_imgs_dir_name": "val_imgs",
    # data
    "df_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv",
    # training configuration
    "max_train_steps": 25000,
    "save_checkpoint_interval": 10000,  # 10000,
    # ---- memory reduction
    "amp": True,
    # ---- Training stability
    "batch_size": 12,  # 3
    "gradient_accumulation_steps": 1,
    "use_ema": False,
    "ema_params": {
        "decay": 0.3,
        "warm_up_steps": 50000,
        "warm_up_decay": 0.3,
    },
    # ---- optimizer
    "lr_g": 1e-4,  # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    "lr_d": 1e-4,  # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    # "segmentation_encoder_lr": 5e-5, # 1e-4 for maisi, 1e-5 for blsmd
    # "lr":  1e-3, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    # "lr":  2.5e-5, # for maisi 1e-3 for maisi 1e-4 # for blsmd 2.5e-5
    # ---- lr_scheduler
    "lr_scheduler_g": None,
    "lr_scheduler_d": None,
    # "lr_scheduler": {"name": "PolynomialLR", "power": 2.0},
    # "lr_scheduler": {"name": "CosineAnnealingLR", "eta_min": 1e-6},
    # "lr_scheduler_g": {
    #     "name": "WarmupCosineLR",
    #     "warmup_start_factor": 1e-1,
    #     "warmup_steps": 500,
    #     "eta_min": 1e-4,
    # },
    # "lr_scheduler_d": {
    #     "name": "WarmupCosineLR",
    #     "warmup_start_factor": 1e-1,
    #     "warmup_steps": 500,
    #     "eta_min": 1e-4,
    # },
    # "lr_scheduler_segmentation_encoder": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-4, "warmup_steps": 500, "eta_min": 1e-6},
    # "lr_scheduler": {"name": "WarmupCosineLR", "warmup_start_factor": 1e-2, "warmup_steps": 25, "eta_min": 1e-6},
    # ---- pretrained_model
    # "load_pretrained_model_from": "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_training/no_synthsr/aaco5590_dataset_no_outliers_bfc/models/rflow/check_points/model_200000.pt",
    "load_pretrained_model_from": None,  
    # ---- resume from checkpoint
    "resume_from_checkpoint_path_name": None,  
    # "resume_from_checkpoint_path_name": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated_synthsr/test1_merged3_4res_probseg/check_points/model_100000.pt",  # not working
    # reproducibility
    "seed": 42,
    # validation
    "val_interval": 100,
    "initial_val": False,  # remember drop out
    # "validation_first": True, # if True, the model will be validated before the first training step, if False, the model will be validated after the first training step
    "val_seeds": [42],  # [0,12357], # seeds for the noise generation during validation
    "max_val_subjects": None,  # max number of subjects to be generated during validation, set to None to use all the subjects in the val dataloader
    "val_dataset_filters": {
        "sid": ["S0006"]  # Example filter for a specific subject ID
    },

    # specialied synthesis
    # "specialized_index": 1, # None for random, 0 for t1n, 1 for t1c, 2 for t2w, 3 for t2f
    "used_modalities": ["T1W"],  # "T1W", "T2W", "T2FLAIR"
    # "used_resolutions": [1.5, 3, 5, 7], #0.1, 1.5, 3, 5, 7
    # "used_resolutions": [0.1, 1.5, 3, 5, 7],  # 0.1, 1.5, 3, 5, 7
    "used_resolutions": [0.1, 7],  # 0.1, 1.5, 3, 5, 7
    # "used_resolutions": [3, 5, 7], #0.1, 1.5, 3, 5, 7
    # "identity_allowed": True, # if True, the model can learn the identity function, if False, the model has to learn the conversion
    "loss_weights": {
        # "mse": 1.0,
        # "charbonnier": 1.0,
        # "ssim": 0.1,
        "recon_loss_weight": 1.,  # weight for the reconstruction loss
        "logits_fake_loss_weight": 1.,  # weight for the fake logits loss
        "logits_class_loss_weight": 1.,  # weight for
        "cycle_loss_weight": 1.,  # weight for the cycle loss
        "logits_fake_resolution_weights": [0.25, 0.25, 0.25, 0.25],  # weights for the fake logits loss for each resolution
        "logits_class_resolution_weights": [0.25, 0.25, 0.25, 0.25],  # weights for the class logits loss for each
        "disc_class_loss_weight": 1.0,  # weight for the discriminator class loss
    },
    "resolution_emb_channels": None,  # number of resolution embeddings to use in the model
}


args_train = fc.dict_to_args(args_train, deep_conversion=True)
train(
    args_train,
    device,
)
