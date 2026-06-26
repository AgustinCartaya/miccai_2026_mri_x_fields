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

# images
from PIL import Image
import matplotlib.pyplot as plt
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')

# Do not limit the number of columns
pd.set_option('display.max_columns', None)





import numpy as np


def compute_reference_quantiles(
    images,
    masks=None,
    quantiles=np.linspace(0, 100, 1001),
    lower_clip=0,
    upper_clip=100,
):
    """
    Compute a mean reference distribution from a dataset.

    Parameters
    ----------
    images : list[np.ndarray]
    masks : list[np.ndarray]
    quantiles : np.ndarray
    lower_clip : float
    upper_clip : float

    Returns
    -------
    reference_quantiles : np.ndarray
    """

    all_quantiles = []

    if masks is None:
        masks = [np.ones_like(img, dtype=bool) for img in images]

    for img, mask in zip(images, masks):

        voxels = img[mask > 0]

        if len(voxels) == 0:
            continue

        low = np.percentile(voxels, lower_clip)
        high = np.percentile(voxels, upper_clip)

        voxels = voxels[
            (voxels >= low) &
            (voxels <= high)
        ]

        q = np.percentile(voxels, quantiles)
        all_quantiles.append(q)

    reference_quantiles = np.mean(
        np.stack(all_quantiles, axis=0),
        axis=0
    )

    return reference_quantiles


def compute_reference_quantiles_tissuewise(
    images,
    segs,
    quantiles=np.linspace(0, 100, 101),
    lower_clip=1,
    upper_clip=99,
    tissues = {
    "csf": 1,
    "gm": 2,
    "wm": 3,
    }
):
    """
    Compute reference landmarks for each tissue.
    """

    reference = {}

    for tissue, labels in tissues.items():

        all_subject_quantiles = []

        for img, seg in zip(images, segs):

            mask = np.isin(seg, labels)

            voxels = img[mask]

            if len(voxels) < 100:
                continue

            low = np.percentile(voxels, lower_clip)
            high = np.percentile(voxels, upper_clip)

            voxels = voxels[
                (voxels >= low) &
                (voxels <= high)
            ]

            all_subject_quantiles.append(
                np.percentile(voxels, quantiles)
            )

        reference[tissue] = np.mean(
            np.stack(all_subject_quantiles),
            axis=0
        )

    return reference


def load_mean_histogram(training_df, 
                        modalities=["T1W", "T2W", "T2FLAIR"],
                        resolutions=[0.1, 1.5, 3, 5, 7], 
                        max_images=None, 
                        per_tissue=False, 
                        tissues = {
                            "csf": 1,
                            "gm": 2,
                            "wm": 3,
                        }
                        ):
    # max_images = 4
   
    # resolutions = [0.1, 1.5]
    # resolutions = [0.1, 1.5, 3, 5, 7]

    # training_df =  pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv")
    # training_df = training_df[training_df["split"] == "train"]

    # results = {}
    mean_histograms = {}
    
    for modality in modalities:
        # results[modality] = {}
        mean_histograms[modality] = {}

        for resolution in resolutions:
            # results[modality][resolution] = []
            mean_histograms[modality][resolution] = []

            possible_rows = training_df[(training_df["modality"] == modality) & (training_df["resolution"] == resolution)]
            
            available = min(len(possible_rows), max_images) if max_images is not None else len(possible_rows)
            if available == 0:
                print(f"No images available for modality {modality} and resolution {resolution}")
                continue
            selected_rows = possible_rows.sample(n=available, random_state=42)
            # load images
            
            _selected_images = []
            _selected_segs = []
            _selected_masks = []
            bar = tqdm(total=available, desc=f"Loading images for modality {modality} and resolution {resolution}")
            for i, row in selected_rows.iterrows():
                img_path = row["org_img_path"]
                img = nfc.load_nifti(img_path)[0]
                seg = nfc.load_nifti(row["seg_synthseg_path"])[0]

                if per_tissue:
                    seg = ufs.merge_seg96_to_seg3(seg)

                # normalize image to 0-1
                img = util.robust_normalize(img, strictly_positive=True, mask=seg > 0)
                # results[modality][resolution].append(img) 
                _selected_images.append(img)  
                _selected_masks.append(seg > 0)
                _selected_segs.append(seg)
                bar.update(1)
            bar.close()
            # mean_histograms[modality][resolution] = compute_mean_histogram(_selected_images, n_percentiles=1000)[1]
            if not per_tissue:
                mean_histograms[modality][resolution] = compute_reference_quantiles(_selected_images, masks=_selected_masks, quantiles=np.linspace(0, 100, 1001))
            else:
                mean_histograms[modality][resolution] = compute_reference_quantiles_tissuewise(_selected_images, _selected_segs, quantiles=np.linspace(0, 100, 101))
    return mean_histograms
