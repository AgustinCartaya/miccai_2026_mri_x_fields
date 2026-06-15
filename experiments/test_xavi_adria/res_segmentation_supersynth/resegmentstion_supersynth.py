import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import sys


# pytorch
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')


# import prep_supersynth as prep_supersynth
# import prep_vol2vol as prep_vol2vol 
import perp_segmentation_and_resampling as perp_segmentation_and_resampling

def preprocess_supersynth(split="train", filter_modalities=None, filter_resolutions=None):
    # read the csv file created by create_csv.py
    # csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/validation_data.csv"
    csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"

    df = pd.read_csv(csv_path)

    if filter_modalities is not None:
        df = df[df["modality"].isin(filter_modalities)]
    if filter_resolutions is not None:
        df = df[df["resolution"].isin(filter_resolutions)]
    df = df.head(1)

    base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/resegmented_supersynth"
    # base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/pr_{split}_data/preprocessed/supersynth"
    bar = tqdm(total=df.shape[0], desc="Preprocessing images with mri_super_synth")

    seg_paths = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # subject_id = row['sid']
        iid = row['iid']
        resolution = row['resolution']
        modality = row['modality']
        # split = row['split']
        img_path = row['seg_supersynth_path']
        img_path = img_path.replace("segmentation_resampled", "SynthT1")


        # print(type(subject_id), type(resolution), type(modality), type(split), type(img_path))
        output_path = os.path.join(base_output_path, modality, f"{str(resolution)}T")
        output_name = f"{iid}_seg.nii.gz"
        output_path_name = os.path.join(output_path, output_name)
        os.makedirs(output_path, exist_ok=True)

        perp_segmentation_and_resampling.segment_and_resample(img_path, output_path_name, verify=True, algorithm="supersynth")

        bar.update(1)
    bar.close()

    # save the dataframe to a new csv file
    df["reseg_supersynth_path"] = seg_paths
    output_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test_xavi_adria/res_segmentation_supersynth/csv/{split}_data_with_supersynth_segmentation.csv"
    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    # preprocess_supersynth("pr_train", filter_modalities=None, filter_resolutions=[0.1, 1.5])
    # preprocess_supersynth("train", filter_modalities=None, filter_resolutions=None)
    filter_modalities = ["T1W"]
    filter_resolutions = [0.1]
    preprocess_supersynth("val", filter_modalities=filter_modalities, filter_resolutions=filter_resolutions)
    #REMEMBER TO CHANGE THE FOLDER NAME OF PR_TRAIN_DATA TO TRAIN_DATA