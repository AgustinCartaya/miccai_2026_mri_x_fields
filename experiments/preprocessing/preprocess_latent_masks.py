import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import sys


# pytorch
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')
sys.path.append('/home/agustin/phd/synthesis')

import utils.nifti_functions as nfc
import utils.util_freesurfer_segmentation as ufs

import prep_image as prep_image
from scipy.ndimage import zoom

# def preprocess_supersynth(split="train", unique_modality=None, unique_resolution=None, merge_seg_3=False):
#     # read the csv file created by create_csv.py
#     csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"
#     df = pd.read_csv(csv_path)#.head(1)

#     base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/latent_masks"
#     if merge_seg_3:
#         base_output_path = os.path.join(base_output_path, "merged_seg_3")
#     bar = tqdm(total=df.shape[0], desc="Preprocessing images with mri_super_synth")

#     seg_paths = []

#     for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#         subject_id = row['subject_id']
#         resolution = row['resolution']
#         modality = row['modality']
#         seg_path = row['segmentation_path']

#         if not os.path.exists(seg_path):
#             seg_paths.append(None)
#             bar.update(1)
#             continue



#         # print(type(subject_id), type(resolution), type(modality), type(split), type(img_path))
#         output_path = os.path.join(base_output_path, modality, f"{str(resolution)}T", str(subject_id))
#         if unique_modality is not None or unique_resolution is not None:
#             if unique_modality is not None and modality != unique_modality:
#                 if unique_resolution is not None and resolution != unique_resolution:
#                     output_path = os.path.join(base_output_path, unique_modality, f"{str(unique_resolution)}T", str(subject_id)) 
#                 else:
#                     output_path = os.path.join(base_output_path, unique_modality, f"{str(resolution)}T", str(subject_id))
#             elif unique_resolution is not None and resolution != unique_resolution:
#                 output_path = os.path.join(base_output_path, modality, f"{str(unique_resolution)}T", str(subject_id)) 

#         output_name = f"segmentation.npy"
#         output_path_name = os.path.join(output_path, output_name)

#         # verify if the output already exists, if it does, skip the processing
#         if not os.path.exists(output_path_name):
#             print(f"Processing subject {row['subject_id']} modality {row['modality']} resolution {row['resolution']} \n with segmentation path: {seg_path}")
#             seg, aff = nfc.load_nifti(seg_path)

#             if merge_seg_3:
#                 seg = ufs.merge_seg36_to_seg3(seg)

#             seg = prep_image.prepare_img(seg, normalize=False)
#             seg_small = zoom(seg, (0.25, 0.25, 0.25), order=0)

#             os.makedirs(output_path, exist_ok=True)
#             np.save(output_path_name, seg_small)

#         seg_paths.append(output_path_name)
#         bar.update(1)
#     bar.close()

#     # save the dataframe to a new csv file
#     df["latent_seg_mask"] = seg_paths
#     output_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data_with_latent_masks.csv"
#     df.to_csv(output_csv_path, index=False)


# if __name__ == "__main__":
#     preprocess_supersynth(split="pr_train", unique_modality='T1W', unique_resolution=None, merge_seg_3=True)











def process_row(index, row, base_output_path, unique_modality, unique_resolution, merge_seg_3):
    subject_id = row['subject_id']
    resolution = row['resolution']
    modality = row['modality']
    seg_path = row['segmentation_path']

    if (not os.path.exists(seg_path)
        or (unique_modality is not None and modality != unique_modality)
        or (unique_resolution is not None and resolution != unique_resolution)
        ):
        return index, None

    output_path = os.path.join(base_output_path, modality, f"{str(resolution)}T", str(subject_id))
    output_name = "segmentation.npy"
    output_path_name = os.path.join(output_path, output_name)

    if not os.path.exists(output_path_name):
        seg, aff = nfc.load_nifti(seg_path)

        if merge_seg_3:
            seg = ufs.merge_seg36_to_seg3(seg)

        seg = prep_image.prepare_img(seg, normalize=False)
        seg_small = zoom(seg, (0.25, 0.25, 0.25), order=0)

        os.makedirs(output_path, exist_ok=True)
        np.save(output_path_name, seg_small)

    return index, output_path_name


from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_supersynth(split="train", unique_modality=None, unique_resolution=None, merge_seg_3=False):
    csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"
    df = pd.read_csv(csv_path)

    base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/latent_masks"
    if merge_seg_3:
        base_output_path = os.path.join(base_output_path, "merged_seg_3")

    seg_paths = [None] * len(df)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                process_row,
                idx,
                row,
                base_output_path,
                unique_modality,
                unique_resolution,
                merge_seg_3
            )
            for idx, row in df.iterrows()
        ]

        for f in tqdm(as_completed(futures), total=len(futures)):
            idx, result = f.result()
            seg_paths[idx] = result

    df["latent_seg_mask"] = seg_paths

    output_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data_with_latent_masks.csv"
    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    preprocess_supersynth(split="pr_train", unique_modality=None, unique_resolution=None, merge_seg_3=True)