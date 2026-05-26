import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('/home/agustin/phd/synthesis')
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')

import utils.nifti_functions as nfc
import utils.functions as fc
import utils.util as util

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from autoencoder_declaration import AutoencoderPrediction
import prep_image as prep_image


device_name = "cuda:0"
device = torch.device(device_name)
from PIL import Image


# [364, 436, 364] # current shape
# (320, 384, 320) # lower target shape
# (384, 448, 384) # upper target shape   
# (320, 448, 320) # medium target shape   


# def solve_visualization(img):
#     img = prep_image.robust_normalize(img, percentile=(0, 100), strictly_positive=True)
#     print(f"Image shape: {img.shape}, min: {img.min()}, max: {img.max()}")
#     # Extract middle slices
#     mid_sagittal = img[img.shape[0] // 2, :, :]
#     mid_coronal  = img[:, img.shape[1] // 2, :]
#     mid_axial    = img[:, :, img.shape[2] // 2]
    
#     canvas = np.zeros((max(img.shape)*3, max(img.shape), 3), dtype=np.float32)

#     for i in range(3):
#         x_start = 0
#         x_end = mid_sagittal.shape[0]
#         canvas[x_start:x_end, 0:mid_sagittal.shape[1], i]     = mid_sagittal
#         x_start += x_end
#         x_end += mid_coronal.shape[0]
#         canvas[x_start:x_end, 0:mid_coronal.shape[1], i]   = mid_coronal
#         x_start += x_end
#         x_end += mid_axial.shape[0]
#         canvas[x_start:x_end, 0:mid_axial.shape[1], i] = mid_axial

        
#     # convert to uint8
#     canvas = (canvas * 255).astype(np.uint8)
#     return canvas
    
def solve_visualization(img):
    img = prep_image.robust_normalize(img, percentile=(0, 100), strictly_positive=True)

    mid_sagittal = img[img.shape[0] // 2, :, :]
    mid_coronal  = img[:, img.shape[1] // 2, :]
    mid_axial    = img[:, :, img.shape[2] // 2]

    sag = fc.gray_to_rgb(mid_sagittal, to_uint8=False)
    cor = fc.gray_to_rgb(mid_coronal, to_uint8=False)
    axi = fc.gray_to_rgb(mid_axial, to_uint8=False)

    # compute canvas size (H fixed = max height, W = sum widths)
    h1, w1 = sag.shape[:2]
    h2, w2 = cor.shape[:2]
    h3, w3 = axi.shape[:2]

    H = max(h1, h2, h3)
    W = w1 + w2 + w3

    canvas = np.zeros((H, W, 3), dtype=np.float32)

    # red background
    canvas[..., 0] = 1.0

    def place(img, x0):
        h, w = img.shape[:2]
        y0 = (H - h) // 2  # vertical centering
        canvas[y0:y0+h, x0:x0+w] = img

    place(sag, 0)
    place(cor, w1)
    place(axi, w1 + w2)

    return (canvas * 255).astype(np.uint8)

# def solve_visualization(img):
#     img = prep_image.robust_normalize(img, percentile=(0, 100), strictly_positive=True)

#     # Extract middle slices
#     mid_sagittal = img[img.shape[0] // 2, :, :]
#     mid_coronal  = img[:, img.shape[1] // 2, :]
#     mid_axial    = img[:, :, img.shape[2] // 2]

#     # Normalize each slice to same size canvas
#     h, w = mid_sagittal.shape
#     canvas = np.zeros((h * 3, w, 3), dtype=np.float32)

#     # Red background
#     canvas[..., 0] = 1.0  # red channel = 1

#     # Place slices (normalize intensity into grayscale RGB)
#     def to_rgb(slice_2d):
#         s = (slice_2d - slice_2d.min()) / (slice_2d.ptp() + 1e-8)
#         return np.stack([s, s, s], axis=-1)

#     canvas[0:h, :, :]     = to_rgb(mid_sagittal)
#     canvas[h:2*h, :, :]   = to_rgb(mid_coronal)
#     canvas[2*h:3*h, :, :] = to_rgb(mid_axial)

#     return (canvas * 255).astype(np.uint8)

def process_row(row):
    """
    Process a single dataframe row.
    """
    try:
        subject_id = row['subject_id']
        resolution = row['resolution']
        modality = row['modality']
        img_path = row['path']
        # img_path = row['segmentation_path']

        img, aff = nfc.load_nifti(img_path, transpose=False)
        # img = np.where(img != 0, 1.,0.) # set negative values to 0
        # img = np.where(img > 0, 1, 0) # set negative values to 0
        
        img = prep_image.robust_normalize(img, percentile=(0,100), strictly_positive=True)

        org_shape = img.shape
        new_shape = (320, 448, 320) 
        img_cropped = prep_image.resize_center_crop_pad(
            img,
            new_shape=new_shape
        )[0].copy()

        img_back = prep_image.resize_center_crop_pad(
            img_cropped,
            new_shape=org_shape
        )[0].copy()

        is_similar = np.allclose(img, img_back)


        if not is_similar:
            new_shape_str = f"{new_shape[0]}_{new_shape[1]}_{new_shape[2]}"

            base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/preprocessing/tests/verify_cropping/verify_cropping_{new_shape_str}"
            comparison_output_path = os.path.join(base_output_path, "comparison")
            individual_output_path = os.path.join(base_output_path, "individual")
            os.makedirs(comparison_output_path, exist_ok=True)
            os.makedirs(individual_output_path, exist_ok=True)
            # imgs_list_2D = fc.cat_n_views_different_layers([img,img_back], 
            #                                             view_layersoffset_list=[(2, 0), (2, -15), (1, 0), (0, 10)], 
            #                                             axis=0, 
            #                                             img_cropping=0,
            #                                             to_rgb=True)
            
            # imgs_list_2D = [solve_visualization(img), solve_visualization(img_cropped)]
            # complete_img = np.concatenate(imgs_list_2D, axis=1)
            # complete_img = Image.fromarray(complete_img)
            # complete_img.save(f"{comparison_output_path}/{subject_id}.png" )

            # imgs_list_2D = fc.cat_n_views_different_layers([img_cropped], 
            #                                             view_layersoffset_list=[(2, 0), (2, -15), (1, 0), (0, 10)], 
            #                                             axis=0, 
            #                                             img_cropping=0,
            #                                             to_rgb=True)
            
            imgs_list_2D = [solve_visualization(img_cropped)]
            complete_img = np.concatenate(imgs_list_2D, axis=1)
            complete_img = Image.fromarray(complete_img)
            complete_img.save(f"{individual_output_path}/{subject_id}_cropped.png" )


            return (
                f"Warning: the original and the back-transformed images "
                f"are not similar for subject {subject_id}, modality {modality}, "
                f"resolution {resolution}T. "
                f"Max difference: {np.sum(np.abs(img - img_back))}"
            )
            
        return None

    except Exception as e:
        return f"Error processing {row.get('path', 'unknown')}: {e}"


def preprocess_latents(split="train", num_workers=8):
    csv_path = (
        f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/"
        f"{split}_data.csv"
    )

    df = pd.read_csv(csv_path)

    rows = [row for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = [executor.submit(process_row, row) for row in rows]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preprocessing images with mri_super_synth"
        ):

            result = future.result()

            if result is not None:
                print(result)


if __name__ == "__main__":

    preprocess_latents(
        split="pr_train",
        num_workers=8
    )
