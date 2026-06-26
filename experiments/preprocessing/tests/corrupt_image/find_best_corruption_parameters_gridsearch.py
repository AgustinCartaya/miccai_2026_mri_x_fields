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
import utils.functions as fc
import utils.util_freesurfer_segmentation as ufs
import utils.util as util

import prep_image as prep_image
import prep_histogram as prep_histogram
# from scipy.ndimage import zoom

from scipy.ndimage import gaussian_filter


from scipy.optimize import differential_evolution
from tqdm import tqdm
# ---------------------------------------------------------
# Transformation pipeline
# ---------------------------------------------------------


def add_gaussian_noise(image, mask, sigma=0.05):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    # clip to 0-1 range
    noisy_image = np.clip(noisy_image, 0, 1)
    noisy_image[mask == 0] = 0
    return noisy_image


def load_mean_histograms_global_tissuewise(output_dir, modalitites=["T1W", "T2W", "T2FLAIR"], resolutions=[0.1, 1.5, 3, 5, 7]):
    mean_histograms_global_tw = {}
    for _modality in modalitites:
        mean_histograms_global_tw[_modality] = {}
        for _resolution in resolutions:
            mean_histograms_global_tw[_modality][_resolution] = np.load(os.path.join(output_dir, f"{_modality}_{_resolution}.npy"))
    return mean_histograms_global_tw


def preprocess_image(src_img, src_seg):
    src_seg = ufs.merge_seg96_to_seg3(src_seg)
    src_img[src_seg == 0] = 0
    src_img = util.robust_normalize(src_img, percentile=(0, 100), strictly_positive=True, mask=src_seg > 0)
    # src_img = util.robust_normalize(src_img, percentile=(0.5, 99.5), strictly_positive=True, mask=src_seg > 0)
    return src_img, src_seg

def load_train_data(src_modality, src_res, tar_res, num_subjects=None, apply_histogram_matching=False, mean_histograms_global=None):
    df = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv")
    df = df[df["paired"] == 1]

    df = df[(df["modality"] == src_modality) & ((df["resolution"] == src_res) | (df["resolution"] == tar_res))]

    if num_subjects is not None:
        unique_sids = df["sid"].unique()
        if num_subjects < len(unique_sids):
            selected_sids = random.sample(list(unique_sids), num_subjects)
            df = df[df["sid"].isin(selected_sids)]


    train_data = []
    for sid in df["sid"].unique():
        src_row = df[(df["sid"] == sid) & (df["modality"] == src_modality) & (df["resolution"] == src_res)].iloc[0]
        tar_row = df[(df["sid"] == sid) & (df["modality"] == src_modality) & (df["resolution"] == tar_res)].iloc[0]

        src_img, _ = nfc.load_nifti(src_row["org_img_path"])
        src_seg, _ = nfc.load_nifti(src_row["seg_supersynth_path"])
        tar_img, _ = nfc.load_nifti(tar_row["org_img_path"])
        tar_seg, _ = nfc.load_nifti(tar_row["seg_supersynth_path"])

        src_img, src_seg = preprocess_image(src_img, src_seg)
        tar_img, tar_seg = preprocess_image(tar_img, tar_seg)

        if apply_histogram_matching:
            src_img = prep_histogram.match_hist_with_reference(src_img, mean_histograms_global[src_modality][tar_res], mask=src_seg)

        train_data.append({
            "src_img": src_img,
            "src_seg": src_seg,
            "tar_img": tar_img,
            "tar_seg": tar_seg
        })

    return train_data


# def transform_image(
#     src_img,
#     src_seg,
#     x_noise,
#     x_sigma,
#     x_power,
#     histogram_reference,
# ):
#     """
#     Apply the complete transformation pipeline.
#     """

#     # Noise
#     c_img_noisy = add_gaussian_noise(
#         src_img,
#         src_seg,
#         x_noise
#     )

#     # Gaussian smoothing
#     c_img_gaussian = gaussian_filter(
#         c_img_noisy,
#         sigma=x_sigma
#     )

#     # Histogram matching
#     c_img_histmatched = (
#         prep_histogram.match_hist_with_reference(
#             c_img_gaussian,
#             histogram_reference,
#             mask=src_seg
#         )
#     )

#     # Power transform
#     c_img_final = np.power(
#         np.clip(c_img_histmatched, 1e-8, None),
#         x_power
#     )

#     return c_img_final



def transform_image(
    src_img,
    src_seg,
    x_noise,
    x_sigma,
    x_power,
    histogram_reference=None,
):
    """
    Apply the complete transformation pipeline.
    """

    # Noise
    c_img_noisy = add_gaussian_noise(
        src_img,
        src_seg,
        x_noise
    )

    # Gaussian smoothing
    c_img_gaussian = gaussian_filter(
        c_img_noisy,
        sigma=x_sigma
    )

    # Power transform
    c_img_final = np.power(
        np.clip(c_img_gaussian, 1e-8, None),
        x_power
    )

    return c_img_final








# ---------------------------------------------------------
# Dataset loss
# ---------------------------------------------------------
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
Z_CLIP_RANGE = (150, 180)

def dataset_mse(params, train_data, histogram_reference):
    x_noise, x_sigma, x_power = params

    total_losses = []

    for sample in train_data:
        src_img = sample["src_img"]
        src_seg = sample["src_seg"]
        tar_img = sample["tar_img"]

        pred_img = transform_image(
            src_img,
            src_seg,
            x_noise,
            x_sigma,
            x_power,
            histogram_reference
        )

        _tar_img = tar_img[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]
        _pred_img = pred_img[:, :, Z_CLIP_RANGE[0]:Z_CLIP_RANGE[1]]

        mse = np.mean(
            (_pred_img.astype(np.float32) -
             _tar_img.astype(np.float32)) ** 2
        )
        ssim = util.compute_ssim(_pred_img, _tar_img)
        psnr = util.compute_psnr(_pred_img, _tar_img)

        total_losses.append({"mse": mse, "ssim": ssim, "psnr": psnr})

    mean_loses = {
        "mse": float(np.mean([loss["mse"] for loss in total_losses])),
        "ssim": float(np.mean([loss["ssim"] for loss in total_losses])),
        "psnr": float(np.mean([loss["psnr"] for loss in total_losses]))
    }

    mean_loses["ssim_loss"] = 1.0 - mean_loses["ssim"]
    return mean_loses


# worker function (must be top-level for multiprocessing)
def _evaluate_params(args):
    params, train_data, histogram_reference = args
    loss = dataset_mse(params, train_data, histogram_reference)
    return params, loss


def grid_search_parallel(param_grid, train_data, histogram_reference, max_workers=None):
    best_params = None
    best_loss = None
    grid_table = []

    param_list = list(product(
        param_grid["x_noise"],
        param_grid["x_sigma"],
        param_grid["x_power"]
    ))

    tasks = [(p, train_data, histogram_reference) for p in param_list]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_evaluate_params, t) for t in tasks]

        with tqdm(total=len(futures), desc="Grid search") as pbar:
            for fut in as_completed(futures):
                params, loss = fut.result()

                grid_table.append({
                    "x_noise": params[0],
                    "x_sigma": params[1],
                    "x_power": params[2],
                    "ssim": loss["ssim"],
                    "mse": loss["mse"],
                    "psnr": loss["psnr"]
                })

                if best_loss is None or loss["ssim_loss"] < best_loss["ssim_loss"]:
                    best_loss = loss
                    best_params = params

                pbar.set_postfix({
                    "best_loss": f"{best_loss['ssim']:.6f}",
                    "noise": f"{best_params[0]:.4f}",
                    "sigma": f"{best_params[1]:.4f}",
                    "power": f"{best_params[2]:.4f}",
                })

                pbar.update(1)

    grid_df = pd.DataFrame(grid_table)
    return best_params, best_loss, grid_df

# ---------------------------------------------------------
# Optional callback to monitor optimization
# ---------------------------------------------------------

if __name__ == "__main__":
    used_modality = "T1W"
    src_res = 7
    tar_res = 1.5

    histograms_global_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/postprocessing/histogram_matching/mean_histograms/global"
    mean_histograms_global  = load_mean_histograms_global_tissuewise(histograms_global_path, modalitites=[used_modality], resolutions=[src_res, tar_res]) 

    histogram_reference = mean_histograms_global[used_modality][tar_res]

    train_data = load_train_data(src_modality=used_modality, 
                                 src_res=src_res, 
                                 tar_res=tar_res, 
                                 num_subjects=None, 
                                 apply_histogram_matching=True, 
                                 mean_histograms_global=mean_histograms_global)

    print("Finding best corruption parameters for {} at {}T -> {}T".format(used_modality, src_res, tar_res))
    # print(f"Initial MSE: {dataset_mse([0.0, 0.0, 1.0], train_data, histogram_reference):.8f}")


    aa = 10
    search_space = {
        "x_noise": np.linspace(0.0, 0.1, aa),
        "x_sigma": np.linspace(0.0, 2.5, aa),
        "x_power": np.linspace(0.5, 1.5, 10),
    }
    best_params, best_loss, grid_df = grid_search_parallel(
        param_grid=search_space,
        train_data=train_data,
        histogram_reference=histogram_reference,
        max_workers=32
    )

    print("\nGrid search finished")
    print("----------------------")
    print("Best parameters:")
    print(f"x_noise = {best_params[0]:.6f}")
    print(f"x_sigma = {best_params[1]:.6f}")
    print(f"x_power = {best_params[2]:.6f}")

    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/preprocessing/tests/corrupt_image/best_corruption_parameters_gridsearch"
    os.makedirs(output_path, exist_ok=True)
    # save the best parameters to a json file
    import json
    best_params_dict = {
        "x_noise": best_params[0],
        "x_sigma": best_params[1],
        "x_power": best_params[2],
        "final_mse": best_loss["mse"],
        "final_ssim": best_loss["ssim"],
        "final_psnr": best_loss["psnr"],
        "used_modality": used_modality,
        "src_res": src_res,
        "tar_res": tar_res,
    }
    with open(os.path.join(output_path, f"best_corruption_parameters_{used_modality}_{src_res}T_{tar_res}T.json"), "w") as f:
        json.dump(best_params_dict, f, indent=4)

    # save the grid search results to a csv file
    grid_df.to_csv(os.path.join(output_path, f"grid_search_results_{used_modality}_{src_res}T_{tar_res}T.csv"), index=False)


