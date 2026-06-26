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
    src_img = util.robust_normalize(src_img, strictly_positive=True, mask=src_seg > 0)

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

def transform_image(
    src_img,
    src_seg,
    x_noise,
    x_sigma,
    x_power,
    histogram_reference,
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

    # # Histogram matching
    # c_img_histmatched = (
    #     prep_histogram.match_hist_with_reference(
    #         c_img_gaussian,
    #         histogram_reference,
    #         mask=src_seg
    #     )
    # )

    # Power transform
    c_img_final = np.power(
        c_img_gaussian,
        x_power
    )

    return c_img_final









# ---------------------------------------------------------
# Dataset loss
# ---------------------------------------------------------
Z_CLIP_RANGE = (150, 180)

def dataset_mse(
    params,
    train_data,
    histogram_reference,
):
    """
    Average MSE across all training pairs.
    """

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

    mean_losses = {
        "mse": float(np.mean([loss["mse"] for loss in total_losses])),
        "ssim": float(np.mean([loss["ssim"] for loss in total_losses])),
        "psnr": float(np.mean([loss["psnr"] for loss in total_losses]))
    }
    mean_losses["ssim_loss"] = 1.0 - mean_losses["ssim"]

    return mean_losses

def objective_function(params, train_data):
    """
    Objective function for optimization.
    """
    loss = dataset_mse(
        params,
        train_data,
        None
    )
    return loss["ssim_loss"]  # We want to minimize the ssim_loss











def optimize_corruption_params(
    train_data,
    used_modality,
    src_res,
    tar_res,
    bounds=None,
    maxiter=10,
    popsize=5,
    save_dir="/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/preprocessing/tests/corrupt_image/best_corruption_parameters",
):
    import os
    import json
    from tqdm import tqdm
    from scipy.optimize import differential_evolution

    # ---------------------------------------------------------
    # Load histogram + dataset
    # ---------------------------------------------------------

    print(f"\n=== Optimizing {used_modality} ({src_res}T → {tar_res}T) ===")
    print(f"Initial SSIM loss: {objective_function([0.0, 0.0, 1.0], train_data):.8f}")

    # ---------------------------------------------------------
    # Progress tracking (local state, no globals)
    # ---------------------------------------------------------
    pbar = tqdm(total=maxiter, desc=f"{used_modality} DE", unit="gen")
    best_loss = {"ssim_loss": float("inf")}

    # def callback(xk, convergence):
    #     nonlocal best_loss

    #     loss = dataset_mse(xk, train_data, histogram_reference)

    #     pbar.update(1)

    #     if loss["ssim_loss"] < best_loss["ssim_loss"]:
    #         best_loss = loss

    #         pbar.set_postfix({
    #             "best_ssim": f"{loss['ssim']:.6f}",
    #             "noise": f"{xk[0]:.4f}",
    #             "sigma": f"{xk[1]:.4f}",
    #             "power": f"{xk[2]:.4f}",
    #         })

    def callback(xk, convergence):
        nonlocal best_loss

        pbar.update(1)

        # Only track parameters (NO dataset recomputation)
        pbar.set_postfix({
            "noise": f"{xk[0]:.4f}",
            "sigma": f"{xk[1]:.4f}",
            "power": f"{xk[2]:.4f}",
            "conv": f"{convergence:.2e}",
        })


    # ---------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------
    result = differential_evolution(
        func=objective_function,
        bounds=bounds,
        args=(train_data,),
        strategy="best1bin",
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        workers=8,
        updating="deferred",
        callback=callback,
        polish=False,
    )

    pbar.close()

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    best_noise, best_sigma, best_power = result.x

    result_loss = dataset_mse(result.x, train_data, None)

    print("\nOptimization finished")
    print(f"Modality: {used_modality}")
    print(f"x_noise = {best_noise:.6f}")
    print(f"x_sigma = {best_sigma:.6f}")
    print(f"x_power = {best_power:.6f}")
    print(f"Final SSIM = {result.fun:.8f}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)

    best_params = {
        "x_noise": float(best_noise),
        "x_sigma": float(best_sigma),
        "x_power": float(best_power),
        "final_mse": float(result_loss["mse"]),
        "final_ssim": float(result_loss["ssim"]),
        "final_psnr": float(result_loss["psnr"]),
        "used_modality": used_modality,
        "src_res": src_res,
        "tar_res": tar_res,
    }

    out_file = os.path.join(
        save_dir,
        f"best_corruption_parameters_{used_modality}_{src_res}T_{tar_res}T.json"
    )

    with open(out_file, "w") as f:
        json.dump(best_params, f, indent=4)

    return best_params



if __name__ == "__main__":
    # ---------------------------------------------------------
    # Bounds
    # ---------------------------------------------------------
    bounds = [
        (0.0, 0.1),   # noise
        (0.0, 2.5),   # sigma
        (0.5, 1.5),   # power
    ]

    histograms_global_path="/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/postprocessing/histogram_matching/mean_histograms/global"


    for used_modality in ["T1W", "T2W", "T2FLAIR"]:
        for src_res, tar_res in [(7, 0.1), (7, 1.5), (7, 3), (7, 5), 
                                 (5, 0.1), (5, 1.5), (5, 3),
                                 (3, 0.1), (3, 1.5),
                                 (1.5, 0.1)]:

            if os.path.exists(f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/preprocessing/tests/corrupt_image/best_corruption_parameters/best_corruption_parameters_{used_modality}_{src_res}T_{tar_res}T.json"):

                print(f"\nFinding best corruption parameters for {used_modality} at {src_res}T -> {tar_res}T")
                continue
            mean_histograms_global = load_mean_histograms_global_tissuewise(
                histograms_global_path,
                modalitites=[used_modality],
                resolutions=[src_res, tar_res]
            )

            histogram_reference = mean_histograms_global[used_modality][tar_res]

            train_data = load_train_data(
                src_modality=used_modality,
                src_res=src_res,
                tar_res=tar_res,
                num_subjects=None,
                apply_histogram_matching=True,
                mean_histograms_global=mean_histograms_global
            )


            best_params = optimize_corruption_params(
                train_data,
                used_modality,
                src_res,
                tar_res,
                bounds=bounds,
                maxiter=10,
                popsize=5,
                save_dir="/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/preprocessing/tests/corrupt_image/best_corruption_parameters",
            )


# # ---------------------------------------------------------
# # Optional callback to monitor optimization
# # ---------------------------------------------------------
# used_modality = "T1W"
# src_res = 7
# tar_res = 1.5

# output_dir = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/postprocessing/histogram_matching/mean_histograms/global"
# mean_histograms_global  = load_mean_histograms_global_tissuewise(output_dir, modalitites=[used_modality], resolutions=[src_res, tar_res]) 

# histogram_reference = mean_histograms_global[used_modality][tar_res]

# train_data = load_train_data(src_modality=used_modality, 
#                                 src_res=src_res, 
#                                 tar_res=tar_res, 
#                                 num_subjects=None, 
#                                 apply_histogram_matching=True, 
#                                 mean_histograms_global=mean_histograms_global)
# print("Finding best corruption parameters for {} at {}T -> {}T".format(used_modality, src_res, tar_res))
# print(f"Initial SSIM loss: {objective_function([0.0, 0.0, 1.0], train_data, histogram_reference):.8f}")

# maxiter = 3

# pbar = tqdm(
#     total=maxiter,
#     desc="Differential Evolution",
#     unit="gen"
# )
# best_loss = None


# def callback(xk, convergence):

#     global best_loss

#     loss = dataset_mse(
#         xk,
#         train_data,
#         histogram_reference
#     )
#     pbar.update(1)
#     if best_loss is None or loss["ssim_loss"] < best_loss["ssim_loss"]:
#         best_loss = loss

#         # print(
#         #     f"\nNew best:"
#         #     f"\n noise={xk[0]:.5f}"
#         #     f"\n sigma={xk[1]:.5f}"
#         #     f"\n power={xk[2]:.5f}"
#         #     f"\n mse={loss:.8f}"
#         # )

#         pbar.set_postfix({
#             "best_ssim": f"{loss['ssim']:.6f}",
#             "noise": f"{xk[0]:.4f}",
#             "sigma": f"{xk[1]:.4f}",
#             "power": f"{xk[2]:.4f}",
#         })

# # ---------------------------------------------------------
# # Parameter ranges
# # ---------------------------------------------------------

# bounds = [

#     # x_noise
#     (0.0, 0.1,),

#     # x_sigma
#     (0.0, 2.5,),

#     # x_power
#     (0.5, 1.5,),
# ]

# # ---------------------------------------------------------
# # Optimization
# # ---------------------------------------------------------

# result = differential_evolution(
#     func=objective_function,
#     bounds=bounds,
#     args=(
#         train_data,
#         histogram_reference,
#     ),
#     strategy="best1bin",
#     popsize=1,
#     maxiter=maxiter,
#     tol=1e-6,
#     mutation=(0.5, 1.0),
#     recombination=0.7,
#     seed=42,
#     workers=8,        # all CPU cores
#     updating="deferred",
#     callback=callback,
#     polish=True,
# )

# # ---------------------------------------------------------
# # Results
# # ---------------------------------------------------------
# pbar.close()
# best_noise, best_sigma, best_power = result.x
# result_loss = dataset_mse(
#     result.x,
#     train_data,
#     histogram_reference
# )

# print("\nOptimization finished")
# print("----------------------")
# print("Best parameters:")
# print(f"x_noise = {best_noise:.6f}")
# print(f"x_sigma = {best_sigma:.6f}")
# print(f"x_power = {best_power:.6f}")
# print(f"Final SSIM = {result.fun:.8f}")


# output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/preprocessing/tests/corrupt_image/best_corruption_parameters"

# # save the best parameters to a json file
# import json
# best_params = {
#     "x_noise": best_noise,
#     "x_sigma": best_sigma,
#     "x_power": best_power,
#     "final_mse": result_loss["mse"],
#     "final_ssim": result_loss["ssim"],
#     "final_psnr": result_loss["psnr"],
#     "used_modality": used_modality,
#     "src_res": src_res,
#     "tar_res": tar_res,
# }
# os.makedirs(output_path, exist_ok=True)
# with open(os.path.join(output_path, f"best_corruption_parameters_{used_modality}_{src_res}T_{tar_res}T.json"), "w") as f:
#     json.dump(best_params, f, indent=4)



