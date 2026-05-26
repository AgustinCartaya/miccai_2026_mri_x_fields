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
import sys

sys.path.append('/home/agustin/phd/synthesis')
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test2_direct_dummy_conversion/training/networks_declaration')
# sys.path.append('/home/agustin/phd/synthesis/tests/D3/preprocessing/brainst_preprocessing_pipeline/preprocessing')

sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')
import prep_image as prep_image

# pytorch
import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
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

# monai
from monai.bundle import ConfigParser
import diffusion_model_unet_maisi as diffusion_model_unet_maisi
from autoencoder_declaration import AutoencoderPrediction

# images
from PIL import Image



# device_name = f"cuda:{gpu_selector.get_least_used_gpu()}"
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





def instantiate_unconditioned_models(device, nb_resolutions):

    networks_config =  {
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

        "diffusion_unet_def": {
            "_target_": "monai.apps.generation.maisi.networks.diffusion_model_unet_maisi.DiffusionModelUNetMaisi",
            "spatial_dims": 3,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": 2,
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
                32,
                32
            ],
            "use_flash_attention": True,
            "with_conditioning": False,

            "nb_modalities": nb_resolutions,
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
                                                attention_levels = args.diffusion_unet_def.attention_levels,
                                                #    norm_num_groups = args.diffusion_unet_def.norm_num_groups,
                                                #    norm_eps = args.diffusion_unet_def.norm_eps,
                                                #    resblock_updown = args.diffusion_unet_def.resblock_updown,
                                                num_head_channels = args.diffusion_unet_def.num_head_channels,
                                                with_conditioning = args.diffusion_unet_def.with_conditioning,
                                                use_flash_attention = args.diffusion_unet_def.use_flash_attention,
                                                nb_modalities = args.diffusion_unet_def.nb_modalities,
                                                )
    

    # autoencoder (just for validation)
    # autoencoder = parser.get_parsed_content("autoencoder_def").to(device)
    autoencoder_chekpoint_path = "/home/agustin/phd/synthesis/tests/D3/maisi/understanding_vae/vae_weights/autoencoder_epoch273.pt"
    # checkpoint_autoencoder = torch.load(autoencoder_chekpoint_path, weights_only=True, map_location=device)
    # autoencoder.load_state_dict(checkpoint_autoencoder)
    # autoencoder.eval()
    
    autoencoder = AutoencoderPrediction(autoencoder_chekpoint_path, device, half=True)
    


    return {"unet": unet, 
              "autoencoder": autoencoder, 
              "networks_config": args,}










class LoadPaths:
    def __init__(self, dataset_path_name, modalities, resolutions, dataset_filters=None):
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
                
                

        # self.subject_ids = self.df["subject_id"].unique()
        self.modalities = modalities #self.df["modality"].unique()
        self.resolutions = resolutions # self.df["resolution"].unique()
        # self.latent_path = latent_path

        # order subjects by id and age
        # self.df = self.complete_dataset.df.sort_values(by=['subject_id', 'age'])

    # def create_latent_paths(self, row):
    #     latent_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/train_data/preprocessed/maisi_latents"
    #     subject_id = row["subject_id"]
    #     modality = row["modality"]
    #     resolution = row["resolution"]

    #     latent_name = f"{modality}_{resolution}T_{subject_id[1:]}_latent.npy"
    #     latent_path_name = os.path.join(latent_path, modality, f"{resolution}T", latent_name)
        
    #     # verify that the latent path exists, if not raise an error
    #     if not os.path.exists(latent_path_name):
    #         raise FileNotFoundError(f"Latent path {latent_path_name} does not exist.")
    #     return latent_path_name
    


    def get_data(self, split="train"):
        complete_df = self.df.copy()
        complete_df = complete_df[complete_df["split"] == split]
        
        self.subject_ids = complete_df["subject_id"].unique()
        

        # unique_ids = complete_df["subject_id"].unique()
        # unique_modalities = complete_df["modality"].unique()
        # unique_resolutions = complete_df["resolution"].unique()

        instances = {} 
        # for i, row in complete_df.iterrows():
        for subject_id in self.subject_ids:
            instances[subject_id] = {}
            for modality in self.modalities:
                instances[subject_id][modality] = {}
                for resolution in self.resolutions:
                    
                    row = complete_df[(complete_df["subject_id"] == subject_id) & (complete_df["modality"] == modality) & (complete_df["resolution"] == resolution)].iloc[0]
                    _instance = {}
                    _instance["img_path"] = row["path"]
                    _instance["latent_path"] = row["latent_path"]#self.create_latent_paths(row)

                    instances[subject_id][modality][resolution] = _instance
                    
        return instances




class PrepareDataset(Dataset):
    def __init__(self, 
                 dataset_path_name,
                 modalities,
                    resolutions,
                 dataset_filters=None,
                 split="train",
                 ):

        # load data
        data_loader = LoadPaths(dataset_path_name,
                                    modalities=modalities,
                                    resolutions=resolutions,
                                  dataset_filters=dataset_filters,
                                  )
        
        self.train_data = data_loader.get_data(split=split)
        self.subject_ids = data_loader.subject_ids
        self.modalities = modalities
        self.resolutions = resolutions
        
        
        print(f"Number of {split} subjects: {len(self.train_data)}")

        # number of latent in the folder
        self.num_instances = len(self.train_data) 
        self._length = self.num_instances
        self.ids_list = list(self.train_data.keys())


    def __len__(self):
        return self._length
    

    def load_t1w_7t_segmentation(self, path):
        seg, aff = prepare_img(path, normalize=False)
        return seg, aff
    

    def load_subject_data(self, instance):
        subject_data = {}
        subject_meta_data = {"latent_path": {}, "img_path": {}}
        for modality in self.modalities:
            for resolution in self.resolutions:
                latent_path = instance[modality][resolution]["latent_path"]
                latent = np.load(latent_path)
                subject_data[f'{modality}_{resolution}'] = latent
                subject_meta_data["latent_path"][f'{modality}_{resolution}'] = latent_path
                subject_meta_data["img_path"][f'{modality}_{resolution}'] = instance[modality][resolution]["img_path"]
        return subject_data, subject_meta_data
    
    # UNTIL HERE
    

    def __getitem__(self, index):

        subject_id = self.ids_list[index % len(self.ids_list)]
        instance = self.train_data[subject_id]

        example, example_meta = self.load_subject_data(instance)
        example["subject_id"] = subject_id
        example["meta_data"] = example_meta

        # load T1w segmentation


        return example
    

def collate_fn(examples, modalities, resolutions):
    res_dict = {}

    for modality in modalities:
        for resolution in resolutions:
            key = f'{modality}_{resolution}'
            res_dict[key] = torch.stack([torch.tensor(example[key]) for example in examples])
    
    res_dict["subject_id"] = [example["subject_id"] for example in examples]
    res_dict["meta_data"] = [example["meta_data"] for example in examples]
    
    return res_dict





def instantiate_dataset(dataset_path_name, modalities, resolutios,
                        batch_size, 
                        dataset_filters=None,
                        split="train",
                        ):
    # ---- Data set creation
    train_dataset = PrepareDataset(
        dataset_path_name=dataset_path_name,
        modalities=modalities,
        resolutions=resolutios,
        dataset_filters=dataset_filters,
        split=split,
    )

    # sampler = MaxPerSubjectSampler(train_dataset, max_per_subject=max_timepoints_per_epoch, shuffle=True, generator=gen_dataloader)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=split=="train",  # shuffle only in training
        # sampler=sampler,
        collate_fn=lambda examples: collate_fn(examples, modalities=modalities, resolutions=resolutios),
        # generator=gen_dataloader,
        num_workers=2, 
        persistent_workers=True,
    )
    return train_dataloader


# def prepare_img(img_path_name, normalize=True):
#     img, aff = nfc.load_nifti(img_path_name)
#     img, new_aff, aff = fc.resize_center_crop_pad(img, (384, 448, 384), aff)
#     if normalize:
#         img = util.robust_normalize(img, percentile=(0,100), strictly_positive=True)
#     return img, aff



@torch.no_grad()
def validation(
    unet,
    autoencoder,
    step,
    val_dataloader,
    args,
):

    print(f"Validation step {step}")
    metrics_list = []
    
    # ---- instantiate dataset
    # val_dataloader = instantiate_dataset(
    #     dataset_path_name=args.df_path,
    #     modalities=args_train.modalities,
    #     resolutios=args_train.resolutions,
    #     batch_size=1,
    #     split="val",
    # )
    
    step_path = f"{args.output_path}/{args.val_imgs_dir_name}/step_{step}"
    
    imgs_2D_output_path = f"{step_path}/imgs_2D"
    metrics_output_path = f"{step_path}/metrics"

    # create folders
    for path in [imgs_2D_output_path, metrics_output_path]:
        os.makedirs(path, exist_ok=True)
    
    for batch in tqdm(val_dataloader, desc="Validation"):
        for modality in args_train.modalities:
            y_org_path = batch["meta_data"][0]["img_path"][f'{modality}_{args_train.target_resolution}']
            # y_img, y_aff = prepare_img(y_org_path)
            y_img, y_aff = nfc.load_nifti(y_org_path)
            y_img = prep_image.prepare_img(y_img, normalize=True)
            # y_img, new_aff, aff = fc.resize_center_crop_pad(y_img, (384, 448, 384), y_aff)
            # y_img = util.robust_normalize(y_img, percentile=(0,100), strictly_positive=True)

            # prepare input and target
            x = []
            idx_resolutions = []
            x_org = []
            for resolution in args_train.x_valid_resolutions:
                key = f'{modality}_{resolution}'
                x.append(batch[key])
                idx_resolutions.append(args_train.x_valid_resolutions.index(resolution))

                x_org_path = batch["meta_data"][0]["img_path"][key]
                x_org.append(prep_image.prepare_img(nfc.load_nifti(x_org_path)[0], normalize=True))

            y = batch[f'{modality}_{args_train.target_resolution}']
            
            x = torch.cat(x, dim=0).to(device)
            y = y.to(device)
                  
            # create one hot encoding for the missing modality
            resolution_one_hot = torch.zeros((x.shape[0], len(args_train.x_valid_resolutions)))
            for i, idx in enumerate(idx_resolutions):
                resolution_one_hot[i, idx] = 1.0
            resolution_one_hot = resolution_one_hot.to(device)
            
            
            # predict
            subject_predictions = []
            subject_predictions_2D = []   
            subject_org_2D = []   
            
            with torch.no_grad(), torch.amp.autocast("cuda"):
                predictions = unet(x, modality_tensor=resolution_one_hot)
                # _pred_img = autoencoder.decode(predictions)
                for i in range(predictions.shape[0]):
                # for i in range(_pred_img.shape[0]):
                #     pred_img = _pred_img[i:i+1]
                    pred_img = autoencoder.decode(predictions[i:i+1])
                    pred_img = torch.clip(pred_img, 0.0, 1.0).cpu().squeeze().numpy()
                    subject_predictions.append(pred_img)
                    subject_predictions_2D.append(pred_img[:, :, pred_img.shape[2]//2])
                    subject_org_2D.append(x_org[i][:, :, x_org[i].shape[2]//2])
                
            # compute metrics

            # clean GPU
            del x, y, predictions
            torch.cuda.empty_cache()

            for res, pred_img in zip(args_train.x_valid_resolutions, subject_predictions):
                # compute metrics
                mse_value = util.compute_mse(y_img, pred_img)
                mae_value = util.compute_mae(y_img, pred_img)
                ssim_value = util.compute_ssim(y_img, pred_img)
                
                metrics_list.append({
                    "subject_id": batch["subject_id"][0],
                    "modality": modality,
                    "resolution": res,
                    "mse": mse_value,
                    "mae": mae_value,
                    "ssim": ssim_value,
                })
                
            # save images
            # save 2D images 
            subject_predictions_2D.append(y_img[:, :, y_img.shape[2]//2])
            subject_org_2D.append(y_img[:, :, y_img.shape[2]//2])
            complete_img_rec = np.concatenate(subject_predictions_2D, axis=1)
            complete_img_org = np.concatenate(subject_org_2D, axis=1)
            complete_img = np.concatenate([complete_img_rec, complete_img_org], axis=0)

            complete_img = Image.fromarray((complete_img*255).astype(np.uint8))
            complete_img.save(f"{imgs_2D_output_path}/step_{step}_{modality}_{batch['subject_id'][0]}.png" )
            
    val_df = pd.DataFrame(metrics_list)
    val_df.to_csv(f"{metrics_output_path}/step_{step}_metrics.csv", index=False)
                        
    torch.cuda.empty_cache()







def save_model(unet, optimizer, lr_scheduler, global_step, out_model_path, ema=None, best=False):  # MOD: se añade parámetro ema
    # Guardar el modelo
    unet_state_dict = unet.module.state_dict() if torch.distributed.is_initialized() else unet.state_dict()
    checkpoint = {
        "unet_state_dict": unet_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "num_train_timesteps": global_step,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
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



def load_checkpoint(checkpoint_path, unet, device, train_dataloader_len,
                    gradient_accumulation_steps, batch_size, optimizer=None, lr_scheduler=None, ema=None):
    # 1. Load checkpoint on CPU to avoid using VRAM
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Load weights into models
    unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)

    # 3. Move models to GPU
    unet.to(device)

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

from monai.losses import PerceptualLoss
from monai.losses import SSIMLoss
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps**2)
        return loss.mean()



def train(
    args_train,
    device,
):

    # ---- reproducibility
    set_seed(args_train.seed)
    # gen_dataloader = torch.Generator().manual_seed(args_train.seed)
    gen_modality = torch.Generator().manual_seed(args_train.seed)
    gen_resolution = torch.Generator().manual_seed(args_train.seed)

    # ---- instantiate models
    x_valid_resolutions = [res for res in args_train.resolutions if res != args_train.target_resolution]
    args_train.x_valid_resolutions = x_valid_resolutions


    models_dict = instantiate_unconditioned_models(device, nb_resolutions=len(x_valid_resolutions))
    unet = models_dict["unet"]
    autoencoder = models_dict["autoencoder"]
    networks_config = models_dict["networks_config"]
    
    # ---- instantiate dataset
    train_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        modalities=args_train.modalities,
        resolutios=args_train.resolutions,
        # latents_path=args_train.latents_path,
        batch_size=args_train.batch_size,
        split="train",
    )

    val_dataloader = instantiate_dataset(
        dataset_path_name=args_train.df_path,
        modalities=args_train.modalities,
        resolutios=args_train.resolutions,
        batch_size=1,
        split="val",
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

    # ---- create tensorboard writer and save configurations
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")  # Formato: Año-Mes-Día_Hora-Minuto
    _sum_writter_dir = os.path.join(_logs_dir_name, f"logs_{timestamp}")
    os.makedirs(_sum_writter_dir, exist_ok=True)
    writer = SummaryWriter(_sum_writter_dir)

    # ---- optimizer and lr_scheduler
    # optimizer = torch.optim.Adam(params=unet.parameters(), lr=args_train.lr) # for maisi  1e-4 # for blsmd 2.5e-5
    optimizer = torch.optim.Adam(
        list(unet.parameters()),
        lr=args_train.lr
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
    # loss_pt = torch.nn.MSELoss()
    loss_charbonnier_pt = CharbonnierLoss()
    # loss_perceptual_pt = PerceptualLoss(spatial_dims=3, network_type="vgg16", pretrained=True, requires_grad=False, device=device)
    # loss_ssim_pt = SSIMLoss(spatial_dims=3, data_range=1.0)


    # ---- training loop
    first_epoch = 0
    global_step = 0
    max_epochs = (args_train.max_train_steps * args_train.gradient_accumulation_steps * args_train.batch_size) // len(train_dataloader) + 1
    print(f"Max epochs: {max_epochs}")

    # ---- resume from checkpoint
    unet.to(device)

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
            device=device,
            train_dataloader_len=len(train_dataloader),
            gradient_accumulation_steps=args_train.gradient_accumulation_steps,
            batch_size=args_train.batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema
        )

    elif args_train.load_pretrained_model_from is not None:
        # checkpoint = torch.load(args_train.load_pretrained_model_from, weights_only=False, map_location=device_name)
        checkpoint = torch.load(args_train.load_pretrained_model_from, map_location=device_name)
        unet.load_state_dict(checkpoint["unet_state_dict"], strict=False)
        if args_train.use_ema and "ema_state_dict" in checkpoint:
            ema.shadow = checkpoint["ema_state_dict"]
            print("EMA state loaded from checkpoint")
        print(f"Pretrained model loaded from {args_train.load_pretrained_model_from}")
    
    unet.train()


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
        desc="Steps",
        initial=global_step
    )
    

    c = 0
    for epoch in range(first_epoch, max_epochs):
        for batch in train_dataloader:
            
            # number of subjects in the batch
            n_batch = len(batch["subject_id"])
            
            # create a tensor of index in 
            # n_batch = 1

            idx_modalities = torch.randint(low=0,high=len(args_train.modalities),size=(n_batch,),generator=gen_modality)
            selected_modalities = [args_train.modalities[i] for i in idx_modalities.tolist()] # this contains n_batch elements
            
            idx_resolutions = torch.randint(low=0,high=len(x_valid_resolutions),size=(n_batch,),generator=gen_resolution)
            selected_resolutions = [x_valid_resolutions[i] for i in idx_resolutions.tolist()] # this contains n_batch elements
            
            x_keys = [f"{mod}_{res}" for mod, res in zip(selected_modalities, selected_resolutions)]
            y_keys = [f"{mod}_{res}" for mod, res in zip(selected_modalities, [args_train.target_resolution]*n_batch)]
            
            x = []
            y = []
            for i, (x_key, y_key) in enumerate(zip(x_keys, y_keys)):
                x.append(batch[x_key][i].to(device).float())
                y.append(batch[y_key][i].to(device).float())
    
            x = torch.stack(x, dim=0) # this contains batch.suze * n_batch elements
            y = torch.stack(y, dim=0) # this contains batch.suze * n_batch elements
            
            # y = torch.cat([batch[key].to(device) for key in y_keys], dim=0).float() # this contains batch.suze * n_batch elements
            

            # create one hot encoding of the selected resolution
            resolution_one_hot = torch.zeros((x.shape[0], len(args_train.x_valid_resolutions)))
            # fill the one hot encoding
            for i, idx in enumerate(idx_resolutions):
                resolution_one_hot[i, idx] = 1.0
            resolution_one_hot = resolution_one_hot.to(device)


            # print(f"x.shape: {x.shape}, y.shape: {y.shape}, one_hot.shape: {resolution_one_hot.shape}")  
            # return          


            ### ---- DEBUGGING
            # print(f"x keys: {x_keys}, y keys: {y_keys}")
            # print(resolution_one_hot)
            
            # c += 1
            # if c == 10:
            #     return
            
            # # create image to visualize the inputs and outputs
            # if global_step % 1 == 0:
            #     input_img = autoencoder.decode(x[0].unsqueeze(0)).cpu().numpy().squeeze()
            #     target_img = autoencoder.decode(y[0].unsqueeze(0)).cpu().numpy().squeeze()
                
            #     input_img = input_img[:, :, input_img.shape[2]//2]
            #     target_img = target_img[:, :, target_img.shape[2]//2]
            #     # input_img = x[0, 1, :, :, x.shape[4]//2].cpu().numpy()
            #     # target_img = y[0, 1, :, :, y.shape[4]//2].cpu().numpy()
            #     complete_img = np.concatenate((input_img, target_img), axis=1)
            #     complete_img = Image.fromarray((complete_img*255).astype(np.uint8))
            #     complete_img.save(f"{args_train.output_path}/{args_train.val_imgs_dir_name}/input_target_step_{global_step}.png" )
            ### ---- DEBUGGING


            # concatenate the rest of the modalities
            # missing_modality = batch["missing_modality"].to(device)
            # missing_modality_one_hot = batch["missing_modality_one_hot"].to(device)

            # Forward pass
            with autocast("cuda", enabled=args_train.amp):
                missing_modality_pred = unet(x=x, 
                                            modality_tensor=resolution_one_hot,
                                            )
                # loss = loss_pt(missing_modality_pred.float(), y.float()) / args_train.gradient_accumulation_steps  # Dividir para escalar la pérdida
                loss_charbonnier = loss_charbonnier_pt(missing_modality_pred.float(), y.float())  # Dividir para escalar la pérdida
                # loss_ssim = loss_ssim_pt(missing_modality_pred.float(), y.float())
                # loss = (args_train.loss_weights.charbonnier * loss_charbonnier + args_train.loss_weights.ssim * loss_ssim) / args_train.gradient_accumulation_steps  # Dividir para escalar la pérdida
                loss = loss_charbonnier / args_train.gradient_accumulation_steps  # Dividir para escalar la pérdida
            # Acumulación de gradientes
            if args_train.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            gradient_accumulation_count += 1  # Contador de pasos acumulados

            # Solo se actualizan los pesos cada `gradient_accumulation_steps` pasos
            if gradient_accumulation_count % args_train.gradient_accumulation_steps == 0:
                if args_train.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if args_train.use_ema:
                    ema.update(step=global_step)

                if lr_scheduler is not None:
                    lr_scheduler.step()

                optimizer.zero_grad()  # Solo hacer zero_grad() después de actualizar los pesos
                gradient_accumulation_count = 0  # Reiniciar el contador

                # update writter 
                if global_step % 10 == 0:
                    writer.add_scalar("Loss/train", loss.item(), global_step)
                    writer.add_scalar("Loss_charbonnier/train", loss_charbonnier.item(), global_step)
                    # writer.add_scalar("Loss_ssim/train", loss_ssim.item(), global_step)
                    writer.add_scalar("Learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    # __mod_index = torch.argmax(batch["missing_modality_one_hot"][0].cpu(), dim=0)
                    # writer.add_scalar("missing modality index", __mod_index, global_step)

                # update progress bar 
                progress_bar.update(1)
                logs = {"0loss": loss.detach().item(), 
                        "0loss_cha": loss_charbonnier.detach().item(),
                        # "0loss_ssim": loss_ssim.detach().item(),
                        # "m_index": __mod_index
                        "1res": selected_resolutions,
                        "2id": batch["subject_id"],
                        # "mod": selected_modalities,
                        }
                progress_bar.set_postfix(**logs)

                # update global step
                global_step += 1

                # # Guardar modelo cada cierto intervalo
                if global_step % args_train.save_checkpoint_interval == 0:
                    save_model(unet, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema, )

                # Generar imágenes en intervalos
                if args_train.initial_val or global_step % args_train.val_interval == 0:
                    unet.eval()

                    # validation(unet, autoencoder, global_step, args_train)

                    try:
                        if args_train.use_ema:
                            ema.apply_shadow()
                        validation(unet, autoencoder, global_step, val_dataloader, args_train)
                    except Exception as e:
                        print(f"ERROR DURING VALIDATION STEP {global_step}: {e}")
                    finally:
                        if args_train.use_ema:
                            ema.restore()
                            
                    args_train.initial_val = False
                    unet.train()

                if global_step >= args_train.max_train_steps:
                    break

        if global_step >= args_train.max_train_steps:
            break

    # make sure the progress bar closes
    progress_bar.close()

    # # make  out_model_path dir if it does not exist
    save_model(unet, optimizer, lr_scheduler, global_step, _checkpoint_dir_name, ema=ema, )










args_train = {
    # directories 
    "output_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test2_direct_dummy_conversion/training/models/test1_T1W_5T_7T",
    "checkpoints_dir_name": "check_points",
    "logs_dir_name": "logs",
    "val_imgs_dir_name": "val_imgs",

    # data
    "df_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv",
    "latents_path": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/train_data/preprocessed/maisi_latents",

    # training configuration
    "max_train_steps": 100000,
    "save_checkpoint_interval": 5000,

    # ---- memory reduction
    "amp": True,

    # ---- Training stability
    "batch_size": 4, # possible 6 (no flash att), 8 (flash att)
    "gradient_accumulation_steps": 1,
    "use_ema": False,
    "ema_params": {
        "decay": 0.5,
        "warm_up_steps": 20000,
        "warm_up_decay": 0.1,
    },

    # ---- optimizer
    "lr": 1e-4, # for maisi  1e-4 # for blsmd 2.5e-5

    # ---- lr_scheduler
    "lr_scheduler": {"name": "PolynomialLR", "power": 2.0},
    # "lr_scheduler": None,

    # ---- pretrained_model
    # "load_pretrained_model_from": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test2_direct_dummy_conversion/training/models/test1_other_loses/check_points/model_5000.pt",
    "load_pretrained_model_from": None, # not working


    # ---- resume from checkpoint
    # "resume_from_checkpoint_path_name": "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test2_direct_dummy_conversion/training/models/test1_other_loses/check_points/model_5000.pt",
    "resume_from_checkpoint_path_name": None, # not working

    # reproducibility
    "seed": 42,

    # validation
    "val_interval": 1000,
    "initial_val": False,

    # specialied synthesis
    # "specialized_index": 1, # None for random, 0 for t1n, 1 for t1c, 2 for t2w, 3 for t2f
    
    "modalities": ["T1W"],
    # "resolutions": [0.1, 1.5, 3, 5, 7],
    "resolutions": [5, 7],
    "target_resolution": 7,

    "loss_weights": {
        # "mse": 1.0,
        "charbonnier": 1.0,
        # "ssim": 0.1,

    },

}


args_train = fc.dict_to_args(args_train, deep_conversion=True)
train(
    args_train,
    device,
)


# x.shape: torch.Size([4, 4, 96, 112, 96]), y.shape: torch.Size([4, 4, 96, 112, 96]), one_hot.shape: torch.Size([4, 1])
# x.shape: torch.Size([4, 4, 96, 112, 96]), y.shape: torch.Size([4, 4, 96, 112, 96]), one_hot.shape: torch.Size([4, 4])