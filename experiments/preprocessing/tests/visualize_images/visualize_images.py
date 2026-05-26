import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('/home/agustin/phd/synthesis')
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')

import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util as util
import utils.util_freesurfer_segmentation as ufs

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from autoencoder_declaration import AutoencoderPrediction
import prep_image as prep_image


device_name = "cuda:2"
device = torch.device(device_name)
from PIL import Image

def visualize_images(csv_path, output_path):
    # read the csv file created by create_csv.py
    df = pd.read_csv(csv_path)

    os.makedirs(output_path, exist_ok=True)

    bar = tqdm(total=df.shape[0], desc="Cresating 2D images")
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        subject_id = row['subject_id']
        resolution = row['resolution']
        modality = row['modality']
        img_path = row['path']
        
        


    bar.close()

if __name__ == "__main__":
    output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/preprocessing/tests/visualize_images/outputs/org"
    csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"
    visualize_images(csv_path, output_path)