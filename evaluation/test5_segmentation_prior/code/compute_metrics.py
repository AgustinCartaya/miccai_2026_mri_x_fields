import argparse
from asyncio import tasks
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




# mine
import utils.nifti_functions as nfc
import utils.util as util
import utils.functions as fc
import utils.util_freesurfer_segmentation as ufs
import utils.gpu_selector as gpu_selector
import data_loaders.load_dataset as load_dataset
import utils.data_normalization as data_normalization

import prep_image as prep_image


from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from functools import partial

# images
from PIL import Image

sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')
# pd.set_option('display.max_columns', None)
# do not constrain the length of the printed dataframe
pd.set_option('display.max_colwidth', None)



SEG_MAPPING_DICT_8 = {
    1: ufs.SURROUNDING_CSF,    
    2: ufs.CEREBRAL_CORTEX_36,
    3: ufs.CEREBRAL_WM + ufs.EXTRA_CEREBRAL_WM,
    4: ufs.INTERNAL_CSF,
    5: ufs.CEREBRAL_SUB_CORTICAL_GM + ufs.EXTRA_CEREBRAL_SUB_CORTICAL_GM,
    6: ufs.CEREBELLUM_GM,
    7: ufs.CEREBELLUM_WM,
    8: ufs.BRAINSTEM
}


SEG_MAPPING_NAME = {
    1: "surrounding_csf",    
    2: "cortex",
    3: "cerebral_wm",
    4: "internal_csf",
    5: "cerebral_sub_cortical_gm",
    6: "cerebellum_gm",
    7: "cerebellum_wm",
    8: "brainstem"
}




def create_df_task2(pred_path_name, modalitites, resolutions):
    data = []
    for modality in modalitites:
        for resolution in resolutions:
            pred_path = os.path.join(pred_path_name, modality, f"0.1T_to_{resolution}T", "pred")
            pred_files = glob.glob(os.path.join(pred_path, "*.nii.gz"))
            for pred_file in pred_files:
                iid = os.path.basename(pred_file).replace(".nii.gz", "")
                sid = "S" + iid.split("_")[-1]
                # org_img_path = pred_file.replace("pred", "org").replace(f"{resolution}T_to_7T", f"{resolution}T")
                # latent_seg_supersynth_path = org_img_path.replace("org", "latent_seg_supersynth").replace(".nii.gz", ".npy")
                
                predseg_file = pred_file.replace("/pred/", "/seg/").replace(".nii.gz", "_seg.nii.gz")
                predseg_supersynth_file = pred_file.replace("/pred/", "/seg/").replace(".nii.gz", "_seg_supersynth.nii.gz")
                if not os.path.exists(predseg_file):
                    predseg_supersynth_file = None
                data.append({
                    "sid": sid,
                    "iid": iid,
                    "modality": modality,
                    "resolution": resolution,
                    "pred_img_path": pred_file,
                    "predseg_img_path": predseg_file,
                    "predseg_supersynth_img_path": predseg_supersynth_file,
                })
    df = pd.DataFrame(data)
    return df



def create_df_task1(pred_path_name, modalitites, resolutions):
    data = []
    for modality in modalitites:
        for resolution in resolutions:
            pred_path = os.path.join(pred_path_name, modality, f"{resolution}T_to_7T", "pred")
            pred_files = glob.glob(os.path.join(pred_path, "*.nii.gz"))
            for pred_file in pred_files:
                iid = os.path.basename(pred_file).replace(".nii.gz", "")
                sid = "S" + iid.split("_")[-1]
                # org_img_path = pred_file.replace("pred", "org").replace(f"{resolution}T_to_7T", f"{resolution}T")
                # latent_seg_supersynth_path = org_img_path.replace("org", "latent_seg_supersynth").replace(".nii.gz", ".npy")
                predseg_file = pred_file.replace("/pred/", "/seg/").replace(".nii.gz", "_seg.nii.gz")
                predseg_supersynth_file = pred_file.replace("/pred/", "/seg/").replace(".nii.gz", "_seg_supersynth.nii.gz")
                if not os.path.exists(predseg_file):
                    predseg_supersynth_file = None
                data.append({
                    "sid": sid,
                    "iid": iid,
                    "modality": modality,
                    "resolution": resolution,
                    "pred_img_path": pred_file,
                    "predseg_img_path": predseg_file,
                    "predseg_supersynth_img_path": predseg_supersynth_file,
                })
    df = pd.DataFrame(data)
    return df




def compute_image_metrics(synthetic_image, org_image):
    ssim = util.compute_ssim(synthetic_image, org_image)
    rmse = util.compute_RMS(synthetic_image, org_image)
    return {"ssim": ssim, "rmse": rmse}


def compute_segmentation_metrics(synthetic_segmentation, org_segmentation, values_to_evaluate=None):
    if values_to_evaluate is None:
        values_to_evaluate = np.unique(org_segmentation)
    dices = {}
    volumes = {}
    for value in values_to_evaluate:
        mask_syn = synthetic_segmentation == value
        mask_org = org_segmentation == value
        dices[value] = util.compute_dice_coefficient(mask_syn, mask_org)
        vol_gt = np.sum(mask_org)
        vol_pred = np.sum(mask_syn)
        volumes[value] = util.compute_normalized_volume(vol_gt, vol_pred)
        
    return {"dice": dices, "volumes": volumes}



Z_CLIP_RANGE = (150, 180)
def evaluate_row(args, test_segmentation_prior=False, crop_z=False):
    i, row = args

    pred_img_path = row["pred_img_path"]
    org_img_path = row["org_img_path"]
    pred_seg_path = row["predseg_img_path"]
    org_seg_path = row["seg_synthseg_path"]

    synthetic_image = nfc.load_nifti(pred_img_path)[0]
    org_image = nfc.load_nifti(org_img_path)[0]
    synthetic_segmentation = nfc.load_nifti(pred_seg_path)[0]
    org_segmentation = nfc.load_nifti(org_seg_path)[0]

    if crop_z:
        synthetic_image = synthetic_image[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]
        org_image = org_image[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]
        synthetic_segmentation = synthetic_segmentation[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]
        org_segmentation = org_segmentation[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]

    synthetic_segmentation = ufs.merge_segmentation(
        synthetic_segmentation,
        SEG_MAPPING_DICT_8
    )
    org_segmentation = ufs.merge_segmentation(
        org_segmentation,
        SEG_MAPPING_DICT_8
    )

    image_metrics = compute_image_metrics(
        synthetic_image,
        org_image
    )

    segmentation_metrics = compute_segmentation_metrics(
        synthetic_segmentation,
        org_segmentation,
        values_to_evaluate=SEG_MAPPING_DICT_8.keys()
    )

    result = {
        "index": i,
        "ssim": image_metrics["ssim"],
        "rmse": image_metrics["rmse"],
    }

    for value in segmentation_metrics["dice"]:
        result[f"dice_{SEG_MAPPING_NAME[value]}"] = \
            segmentation_metrics["dice"][value]

        result[f"vol_{SEG_MAPPING_NAME[value]}"] = \
            segmentation_metrics["volumes"][value]
            
    if test_segmentation_prior:
        pred_supersynth_seg_path = row["predseg_supersynth_img_path"]
        src_supersynth_seg_path = row["src_seg_supersynth_path"]
        if pred_supersynth_seg_path is not None and os.path.exists(pred_supersynth_seg_path):
            src_supersynth_seg_path = nfc.load_nifti(src_supersynth_seg_path)[0]
            pred_supersynth_seg_path = nfc.load_nifti(pred_supersynth_seg_path)[0]
            pred_supersynth_seg_path = ufs.merge_segmentation(
                pred_supersynth_seg_path,
                SEG_MAPPING_DICT_8
            )
            if crop_z:
                pred_supersynth_seg_path = pred_supersynth_seg_path[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]
                src_supersynth_seg_path = src_supersynth_seg_path[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]
            src_supersynth_seg_path = ufs.merge_segmentation(
                src_supersynth_seg_path,
                SEG_MAPPING_DICT_8
            )
            supersynth_segmentation_metrics = compute_segmentation_metrics(
                pred_supersynth_seg_path,
                src_supersynth_seg_path,
                values_to_evaluate=SEG_MAPPING_DICT_8.keys()
            )
            
            for value in supersynth_segmentation_metrics["dice"]:
                result[f"seg_prior_dice_{SEG_MAPPING_NAME[value]}"] = \
                    supersynth_segmentation_metrics["dice"][value]
                result[f"seg_prior_vol_{SEG_MAPPING_NAME[value]}"] = \
                    supersynth_segmentation_metrics["volumes"][value]

    return result







# def evaluate_task_fast_res(df_val, pred_path_name, task):
    
#     if task == 1:
#         pred_path_name = os.path.join(pred_path_name, "task1")

#         modalitites = ["T1W", "T2W", "T2FLAIR"]
#         resolutions = [0.1, 1.5, 3, 5]
#         df_val_task = create_df_task1(pred_path_name, modalitites, resolutions)
        
#         # remove resolution col from df_val
#         merged_df = pd.merge(df_val.drop(columns=["resolution"]), df_val_task[["iid", "resolution", "pred_img_path", "predseg_img_path"]], on=["iid"], how="inner")
        
#         # add src img and seg
#         df_val_src = df_val[["sid", "modality", "resolution", "org_img_path", "seg_synthseg_path"]].copy()
#         rename = {
#             "org_img_path": "src_org_img_path",
#             "seg_synthseg_path": "src_seg_synthseg_path",
#             "seg_supersynth_path": "src_seg_supersynth_path"
#         }
#         df_val_src = df_val_src.rename(columns=rename)
#         merged_df = pd.merge(merged_df, df_val_src, on=["sid", "modality", "resolution"], how="inner")
#         # merged_df = merged_df.head(1)
#         # print(df_val.head())
#         # print(merged_df[["sid", "iid", "modality", "resolution","pred_img_path"]].head(40))
#     elif task == 2:
#         pred_path_name = os.path.join(pred_path_name, "task2")

#         modalitites = ["T1W", "T2W", "T2FLAIR"]
#         resolutions = [1.5, 3, 5, 7]
#         df_val_task = create_df_task2(pred_path_name, modalitites, resolutions)
#         merged_df = pd.merge(df_val, df_val_task, on=["sid", "iid", "modality", "resolution"], how="inner")
        
#         df_val_src = df_val[df_val["resolution"] == 0.1][["sid", "modality", "org_img_path", "seg_synthseg_path"]].copy()
        
#         rename = {
#             "org_img_path": "src_org_img_path",
#             "seg_synthseg_path": "src_seg_synthseg_path",
#             "seg_supersynth_path": "src_seg_supersynth_path"
#         }
#         df_val_src = df_val_src.rename(columns=rename)
#         merged_df = pd.merge(merged_df, df_val_src, on=["sid", "modality"], how="inner")
#         # merged_df = merged_df.head(1)
        
#     # results_df = merged_df[["sid", "iid", "modality", "resolution"]].copy()
    

    
#     tasks = list(merged_df.iterrows())

#     with ProcessPoolExecutor(max_workers=8) as executor:
#         results = list(
#             tqdm(
#                 executor.map(evaluate_row, tasks),
#                 total=len(tasks)
#             )
#         )

#     for result in results:
#         idx = result.pop("index")
#         for key, value in result.items():
#             merged_df.at[idx, key] = value
            
#     output_df_path_name = os.path.join(pred_path_name, "evaluation_metrics.csv")
#     merged_df.to_csv(output_df_path_name, index=False)
    



def evaluate_task_fast(df_original, pred_path_name, task, test_segmentation_prior=False, crop_z=False):
    
    if task == 1:
        pred_path_name = os.path.join(pred_path_name, "task1")

        modalitites = ["T1W", "T2W", "T2FLAIR"]
        resolutions = [0.1, 1.5, 3, 5]
        df_task = create_df_task1(pred_path_name, modalitites, resolutions)
        
        # remove resolution col from df_original to save the source resolution and not always the target (7)
        merged_df = pd.merge(df_original.drop(columns=["resolution"]), df_task[["iid", "resolution", "pred_img_path", "predseg_img_path", "predseg_supersynth_img_path"]], on=["iid"], how="inner")
        
        # add src img and seg
        df_original_src = df_original[["sid", "modality", "resolution", "org_img_path", "seg_synthseg_path", "seg_supersynth_path"]].copy()
        rename = {
            "org_img_path": "src_org_img_path",
            "seg_synthseg_path": "src_seg_synthseg_path",
            "seg_supersynth_path": "src_seg_supersynth_path"
        }
        df_original_src = df_original_src.rename(columns=rename)
        merged_df = pd.merge(merged_df, df_original_src, on=["sid", "modality", "resolution"], how="inner")

        
    elif task == 2:
        pred_path_name = os.path.join(pred_path_name, "task2")

        modalitites = ["T1W", "T2W", "T2FLAIR"]
        resolutions = [1.5, 3, 5, 7]
        df_task = create_df_task2(pred_path_name, modalitites, resolutions)
        
        
        merged_df = pd.merge(df_original, df_task, on=["sid", "iid", "modality", "resolution"], how="inner")
        
        df_original_src = df_original[df_original["resolution"] == 0.1][["sid", "modality", "org_img_path", "seg_synthseg_path", "seg_supersynth_path"]].copy()
        
        rename = {
            "org_img_path": "src_org_img_path",
            "seg_synthseg_path": "src_seg_synthseg_path",
            "seg_supersynth_path": "src_seg_supersynth_path"
        }
        df_original_src = df_original_src.rename(columns=rename)
        merged_df = pd.merge(merged_df, df_original_src, on=["sid", "modality"], how="inner")

    
    tasks = list(merged_df.iterrows())
    
    evaluate_row_partial = partial(
        evaluate_row,
        test_segmentation_prior=test_segmentation_prior,
        crop_z=crop_z
    )
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(evaluate_row_partial, tasks),
                total=len(tasks)
            )
        )

    for result in results:
        idx = result.pop("index")
        for key, value in result.items():
            merged_df.at[idx, key] = value
            
    name_df = "evaluation_metrics.csv" if not crop_z else "evaluation_metrics_cropped.csv"
    output_df_path_name = os.path.join(pred_path_name, name_df)
    merged_df.to_csv(output_df_path_name, index=False)
    

# def evaluate_task2(df_val, pred_path_name):
    
#     pred_path_name = os.path.join(pred_path_name, "task2")

#     modalitites = ["T1W", "T2W", "T2FLAIR"]
#     resolutions = [1.5, 3, 5, 7]

#     df_val_task2 = create_df_task2(pred_path_name, modalitites, resolutions)
    

#     merged_df = pd.merge(df_val, df_val_task2, on=["sid", "iid", "modality", "resolution"], how="inner")
    
#     results_df = merged_df[["sid", "iid", "modality", "resolution"]].copy()
#     results_df["ssim"] = np.nan
#     results_df["rmse"] = np.nan
    
#     for value in SEG_MAPPING_DICT_8.keys():
#         results_df[f"dice_{SEG_MAPPING_NAME[value]}"] = np.nan
#         results_df[f"vol_{SEG_MAPPING_NAME[value]}"] = np.nan


#     for i, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
#         pred_img_path = row["pred_img_path"]
#         org_img_path = row["org_img_path"]
#         pred_seg_path = row["predseg_img_path"]
#         org_seg_path = row["seg_synthseg_path"]

#         synthetic_image = nfc.load_nifti(pred_img_path)[0]
#         org_image = nfc.load_nifti(org_img_path)[0]
#         synthetic_segmentation = nfc.load_nifti(pred_seg_path)[0]
#         org_segmentation = nfc.load_nifti(org_seg_path)[0]
        
#         synthetic_segmentation = ufs.merge_segmentation(synthetic_segmentation, SEG_MAPPING_DICT_8)
#         org_segmentation = ufs.merge_segmentation(org_segmentation, SEG_MAPPING_DICT_8)

#         image_metrics = compute_image_metrics(synthetic_image, org_image)
#         segmentation_metrics = compute_segmentation_metrics(synthetic_segmentation, org_segmentation, values_to_evaluate=SEG_MAPPING_DICT_8.keys())

#         results_df.at[i, "ssim"] = image_metrics["ssim"]
#         results_df.at[i, "rmse"] = image_metrics["rmse"]
        
#         for value in segmentation_metrics["dice"].keys():
#             results_df.at[i, f"dice_{SEG_MAPPING_NAME[value]}"] = segmentation_metrics["dice"][value]
#             results_df.at[i, f"vol_{SEG_MAPPING_NAME[value]}"] = segmentation_metrics["volumes"][value]    
    
#     output_df_path_name = os.path.join(pred_path_name, "evaluation_metrics.csv")
#     results_df.to_csv(output_df_path_name, index=False)



if __name__ == "__main__":
    # df_val = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv")
    # output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/val"
    
    
    df_val = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv")
    df_val = df_val[df_val["split"] == "val"]
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/train"
    
    use_controlnet = False
    n_inference_steps = 30
    crop_z = True

    dm_chk_number = 145000
    dm_seg_channels = 8
    used_mask = f"merged_{dm_seg_channels}"

    # dm_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated/test1/check_points/model_{dm_chk_number}.pt"
    dm_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated/test3_merged8_4res/check_points/model_{dm_chk_number}.pt"
    # dm_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated/test3_merged8_4res_probseg/check_points/model_{dm_chk_number}.pt"
 
    if not use_controlnet:
        output_path = os.path.join(output_path, f"basic", used_mask, f"chk_{dm_chk_number}_steps_{n_inference_steps}")
    else:
        # dm_chk_number = 210000
        cnet_chk_number = 100000
        controlnet_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated_controlnet/test1/check_points/model_{cnet_chk_number}.pt"
        output_path = os.path.join(output_path, f"controlnet", f"chk_{dm_chk_number}_cnchk_{cnet_chk_number}_steps_{n_inference_steps}")
        # raise NotImplementedError("ControlNet is not implemented yet for the evaluation, but it will be in the future. For now, just set use_controlnet to False if you want to run the evaluation.")

    # evaluate_task2(df_val, output_path)
    evaluate_task_fast(df_val, output_path, task=1, test_segmentation_prior=True, crop_z=crop_z)
    # evaluate_task_fast(df_val, output_path, task=2)


# /home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/train/basic/merged_3/chk_200000_steps_30/task2/T1W/0.1T_to_3T/seg/SP_T1W_3T_0006.nii.gz