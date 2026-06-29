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

# mine
import utils.nifti_functions as nfc
import utils.util as util
import utils.functions as fc
import utils.util_freesurfer_segmentation as ufs
import utils.gpu_selector as gpu_selector
import data_loaders.load_dataset as load_dataset
import utils.data_normalization as data_normalization

import prep_image as prep_image

# images
from PIL import Image





modalitites = ["T1W", "T2W", "T2FLAIR"]
resolutions = [0.1, 1.5, 3, 5, 7]

def create_val_set_task_3():

    
    output_gen_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_synthsr_prior/results/val/basic/merged_3/chk_280000_steps_30/with_synthsr/task3"
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/validation_set/results/task3"            
    df_gt_val = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/paired_train_data.csv")
    df_val = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv")


    df_gt_val["_sid"] = df_gt_val["sid"] 
    df_gt_val["sid"] = df_gt_val["iid"].apply(lambda x: f"S0{x[1:4]}")
    # order by sid, modality, resolution
    df_gt_val = df_gt_val.sort_values(by=["sid", "modality", "resolution"])
    # display(df_gt_val.head(5))

    for modality in modalitites:
        for src_resolution in resolutions:
            for tar_resolution in resolutions:
                
                if src_resolution == tar_resolution:
                    continue

                possible_src_rows = df_val[(df_val["modality"] == modality) & (df_val["resolution"] == src_resolution)]

                for index, src_row in possible_src_rows.iterrows():
                    sid = src_row["sid"]
                    iid = src_row["iid"]
                    tar_row = df_gt_val[(df_gt_val["sid"] == sid) & (df_gt_val["modality"] == modality) & (df_gt_val["resolution"] == tar_resolution)].iloc[0]
                
                    # src_img = nfc.load_nifti(src_row["org_img_path"])[0]
                    tar_path = tar_row["org_img_path"]
                    # tra_img, aff = nfc.load_nifti(tar_row["org_img_path"])[0]

                    save_path = os.path.join(output_path, modality, f"{src_resolution}T_to_{tar_resolution}T", "pred", f"{iid.replace(str(src_resolution), str(tar_resolution))}.nii.gz")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    # copy the target image to the save path
                    shutil.copy(tar_path, save_path)

            #         break
            #     break
            # break


    # show_val_results(df_val, df_gt_val, output_gen_path)
        




create_val_set_task_3()