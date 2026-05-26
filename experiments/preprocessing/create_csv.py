import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import sys


# pytorch
sys.path.append('/home/agustin/phd/synthesis')


# import utils.nifti_functions as nfc
# import utils.util as util
# import utils.functions as fc
# import utils.util_freesurfer_segmentation as ufs
# import utils.gpu_selector as gpu_selector
# import data_loaders.load_dataset as load_dataset
# import utils.data_normalization as data_normalization

# import preprocessing.prep_registration as prep_registration
# from scipy.ndimage import affine_transform
# from scipy.ndimage import map_coordinates
# from scipy.ndimage import gaussian_filter
# from scipy.ndimage import zoom



# table columns: subject_id, resolution, modlity, split, path


def create_validation_csv():
    modalitites = ['T1W', 'T2W', 'T2FLAIR']
    resolutions = [0.1, 1.5, 3, 5, 7]
    splits = ['train', 'val']

    base_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/val_data/rawdata"
    csv_list = []

    for modality in modalitites:
        mod_path = os.path.join(base_path, modality)
        for resolution in resolutions:
            res_path = os.path.join(mod_path, f"{resolution}T")
            # read all .nii.gz files in res_path
            files = [f for f in os.listdir(res_path) if f.endswith('.nii.gz')]
            
            for file in tqdm(files):
                # the id is the last 4 characters before the .nii.gz
                subject_id = file.replace('.nii.gz', '')[-4:]
                # print(type(subject_id))
                img_data = {
                    'subject_id': f"S{subject_id}",
                    'resolution': resolution,
                    'modality': str(modality),
                    'split': 'val',
                    'path': os.path.join(res_path, file)
                }
                csv_list.append(img_data)
    df = pd.DataFrame(csv_list)
    # order by subject_id, modality, resolution
    df = df.sort_values(by=['subject_id', 'modality', 'resolution'])
    df.to_csv(os.path.join(base_path, '/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv'), index=False)

def create_train_csv():
    modalitites = ['T1W', 'T2W', 'T2FLAIR']
    resolutions = [0.1, 1.5, 3, 5, 7]
    splits = ['train', 'val']

    base_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/train_data/rawdata"
    csv_list = []

    for modality in modalitites:
        mod_path = os.path.join(base_path, modality)
        for resolution in resolutions:
            res_path = os.path.join(mod_path, f"{resolution}T")
            # read all .nii.gz files in res_path
            files = [f for f in os.listdir(res_path) if f.endswith('.nii.gz')]
            
            for file in tqdm(files):
                # the id is the last 4 characters before the .nii.gz
                subject_id = file.replace('.nii.gz', '')
                # print(type(subject_id))
                img_data = {
                    'subject_id': f"S{subject_id}",
                    'resolution': resolution,
                    'modality': str(modality),
                    'split': 'train',
                    'path': os.path.join(res_path, file)
                }
                csv_list.append(img_data)
    df = pd.DataFrame(csv_list)
    # order by subject_id, modality, resolution
    df = df.sort_values(by=['subject_id', 'modality', 'resolution'])
    df.to_csv(os.path.join(base_path, '/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv'), index=False)




def create_csv(base_path, output_csv_path=None):
    modalitites = ['T1W', 'T2W', 'T2FLAIR']
    resolutions = [0.1, 1.5, 3, 5, 7]
    splits = ['train', 'val']

    csv_list = []

    for modality in modalitites:
        mod_path = os.path.join(base_path, modality)
        for resolution in resolutions:
            res_path = os.path.join(mod_path, f"{resolution}T")
            # read all .nii.gz files in res_path
            files = [f for f in os.listdir(res_path) if f.endswith('.nii.gz')]
            
            for file in tqdm(files):
                # the id is the last 4 characters before the .nii.gz
                subject_id = file.replace('.nii.gz', '')
                # print(type(subject_id))
                img_data = {
                    'subject_id': f"S{subject_id}",
                    'resolution': resolution,
                    'modality': str(modality),
                    'split': 'train',
                    'path': os.path.join(res_path, file)
                }
                csv_list.append(img_data)
    df = pd.DataFrame(csv_list)
    # order by subject_id, modality, resolution
    df = df.sort_values(by=['subject_id', 'modality', 'resolution'])
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
    return df



# create_validation_csv()
# create_train_csv()

p_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/zip/training/unzip/release_20260414/Training_prospective"
r_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/zip/training/unzip/release_20260414/Training_retrospective"
p_df = create_csv(p_path)
r_df = create_csv(r_path)

output_csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/pr_train_data.csv"
final_df = pd.concat([p_df, r_df], ignore_index=True)

# remove duplicates subject_id, modality, resolution, keep the first one
final_df = final_df.drop_duplicates(subset=['subject_id', 'modality', 'resolution'], keep='first')
final_df.to_csv(output_csv_path, index=False)