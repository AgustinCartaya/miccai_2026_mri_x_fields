import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import sys


# pytorch
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')


import prep_supersynth as prep_supersynth
import prep_vol2vol as prep_vol2vol 


def preprocess_supersynth(split="train", filter_modalities=None, filter_resolutions=None, folder_column='iid'):
    # read the csv file created by create_csv.py
    # csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/validation_data.csv"
    csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"

    df = pd.read_csv(csv_path)

    if filter_modalities is not None:
        df = df[df["modality"].isin(filter_modalities)]
    if filter_resolutions is not None:
        df = df[df["resolution"].isin(filter_resolutions)]
    # df = df.head(1)

    base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/supersynth"
    # base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/pr_{split}_data/preprocessed/supersynth"
    bar = tqdm(total=df.shape[0], desc="Preprocessing images with mri_super_synth")

    seg_paths = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        subject_id = row[folder_column]
        resolution = row['resolution']
        modality = row['modality']
        # split = row['split']
        img_path = row['org_img_path']


        # print(type(subject_id), type(resolution), type(modality), type(split), type(img_path))
        output_path = os.path.join(base_output_path, modality, f"{str(resolution)}T", str(subject_id))
        os.makedirs(output_path, exist_ok=True)

# def save_supersynth(img_path_name, out_path, verify=False, verbose=False, convert_to_nifti=True, remove_mgz=True):

        prep_supersynth.save_supersynth(img_path, output_path, verify=True, verbose=False, convert_to_nifti=True, remove_mgz=True, keep_only_segmentation=True)

        # vol2vol the segmentation to the original space of the input image, using the original image as fixed and the segmentation as moving
        fixed = img_path
        moving = os.path.join(output_path, "segmentation.nii.gz")
        output = os.path.join(output_path, "segmentation_resampled.nii.gz")
        prep_vol2vol.apply_vol2vol(fixed, moving, output, verify=True, verbose=False, nearest=True)
        seg_paths.append(output)

        bar.update(1)
    bar.close()

    # save the dataframe to a new csv file
    df["seg_supersynth_path"] = seg_paths
    output_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data_with_supersynth_segmentation.csv"
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    # preprocess_supersynth("pr_train", filter_modalities=None, filter_resolutions=[0.1, 1.5])
    # preprocess_supersynth("train", filter_modalities=None, filter_resolutions=None)
    preprocess_supersynth("val", filter_modalities=None, filter_resolutions=None, folder_column='sid')
    #REMEMBER TO CHANGE THE FOLDER NAME OF PR_TRAIN_DATA TO TRAIN_DATA