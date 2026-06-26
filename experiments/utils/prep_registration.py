import sys
import os
import numpy as np

import SimpleITK as sitk


import subprocess


def save_rigid_registration(path_name_img_fixed, path_name_img_moving, path_name_img_registered, path_name_out_reg_matrix=None, verify=False, verbose=False, affine=False):
    # mri_robust_register --mov moving_img.nii --dst MNI_img.nii --mapmov output_img.nii --lta output_reg_stats.lta  --weights v1to2-weights.mgz --iscale --satit

    if verify:
        exist = os.path.exists(path_name_img_registered)
        if exist:
            print(f"Registration already done for: {path_name_img_moving}\nfile in: {path_name_img_registered}")
            return True
        
    command = ["mri_robust_register", 
               "--mov", path_name_img_moving, 
               "--dst", path_name_img_fixed, 
               "--mapmov", path_name_img_registered, 
               "--iscale", 
               "--satit", 
            #    "--affine", 
            #    "--lta", path_name_out_reg_matrix
               ]
    if path_name_out_reg_matrix is not None:
        command.append("--lta")
        command.append(path_name_out_reg_matrix)
    if affine:
        command.append("--affine")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        if verbose:
            print(f"MNI registration done saved in: {path_name_img_registered}")
    else:
        # Print error message
        print("Error:", result.stderr)


def _aply_precomputed_rigid_registration_lta(path_name_img_fixed, path_name_img_moving, path_name_img_registered, path_name_lta, is_label=False, verify=False, verbose=False ):
    # mri_vol2vol --mov other.mgz --targ dst.mgz --lta m2d.lta --o other_reg.mgz --no-resample
    # mri_vol2vol --mov other.mgz --targ dst.mgz --lta m2d.lta --o other_reg.mgz --interp trilinear --regheader

    if verify:
        exist = os.path.exists(path_name_img_registered)
        if exist:
            print(f"Registration already done for: {path_name_img_moving}\nfile in: {path_name_img_registered}")
            return True
    command = ["mri_vol2vol",
               "--mov", path_name_img_moving, 
               "--targ", path_name_img_fixed, 
               "--lta", path_name_lta, 
               "--o", path_name_img_registered,
               "--precision", "float"]
    
    if is_label:
        command.extend(["--interp", "nearest"])  # Use nearest neighbor interpolation for label images
    else:
        command.extend(["--interp", "trilinear"])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        if verbose:
            print(f"Precomputed registration done saved in: {path_name_img_registered}")
    else:
        # Print error message
        print("Error:", result.stderr)


def _aply_precomputed_rigid_registration_affine_matrix(path_name_img_fixed, path_name_img_moving, path_name_img_registered, affine_matrix, is_label=False, verify=False, verbose=False):
    """
    Applies a 4x4 affine matrix to a 3D image and saves the registered image in the space of the fixed image.

    Parameters:
        fixed_image_path (str): Path to the fixed reference image.
        moving_image_path (str): Path to the moving image.
        affine_matrix (np.ndarray): 4x4 affine matrix in physical coordinates.
        output_path (str): Path to save the registered image.
        is_label (bool): If True, use nearest neighbor interpolation (for label images).
    """
    if not isinstance(affine_matrix, np.ndarray) or affine_matrix.shape != (4, 4):
        raise ValueError("Affine matrix must be a numpy array of shape (4, 4).")

    if verify:
        # verify if the output path exists then dont apply the registration
        if os.path.exists(path_name_img_registered):
            if verbose:
                print(f"Output path {path_name_img_registered} already exists. Skipping registration.")
            return
        
    # Load images
    fixed = sitk.ReadImage(path_name_img_fixed, sitk.sitkFloat32)
    moving = sitk.ReadImage(path_name_img_moving)

    # Create the affine transform
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(affine_matrix[:3, :3].flatten().tolist())
    transform.SetTranslation(affine_matrix[:3, 3].tolist())

    # Choose interpolation mode
    interpolation_mode = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear

    # Apply the transformation
    resampled = sitk.Resample(
        moving,                # Moving image
        fixed,                 # Reference space
        transform,             # Affine transform
        interpolation_mode,    # Use nearest neighbor for labels
        0,                     # Default pixel value
        moving.GetPixelID()    # Preserve original pixel type
    )

    # Save the registered image
    sitk.WriteImage(resampled, path_name_img_registered)
    

def save_precomputed_rigid_registration(path_name_img_fixed, path_name_img_moving, path_name_img_registered, prec_registration, is_label=False, verify=False, verbose=False):
    """
    Applies a precomputed rigid registration to a moving image and saves the registered image in the space of the fixed image.

    Parameters:
        path_name_img_fixed (str): Path to the fixed reference image.
        path_name_img_moving (str): Path to the moving image.
        path_name_img_registered (str): Path to save the registered image.
        prec_registration (str or np.ndarray): Path to the precomputed registration file (LTA) or a 4x4 affine matrix.
        is_label (bool): If True, use nearest neighbor interpolation (for label images).
        verify (bool): If True, check if the output file already exists and skip registration if it does.
        verbose (bool): If True, print additional information.
    """
    if isinstance(prec_registration, str):
        # Assume it's an LTA file
        _aply_precomputed_rigid_registration_lta(path_name_img_fixed, path_name_img_moving, path_name_img_registered, prec_registration, is_label, verify, verbose)
    elif isinstance(prec_registration, np.ndarray):
        # Assume it's a 4x4 affine matrix
        _aply_precomputed_rigid_registration_affine_matrix(path_name_img_fixed, path_name_img_moving, path_name_img_registered, prec_registration, is_label, verify, verbose)
    else:
        raise ValueError("prec_registration must be either a string (path to LTA file) or a numpy array (4x4 affine matrix).")


def ras_to_lps_affine(matrix):
    flip = np.diag([-1, -1, 1, 1])
    return flip @ matrix @ flip


# MNI_T1W_PATH_NAME = "/home/agustin/phd/synthesis/data/MNI_template/oiginal/mni_icbm152_t1_tal_nlin_sym_09c.nii"
# MNI_T1W_SYNTHSR_PATH_NAME = "/home/agustin/phd/synthesis/data/MNI_template/synthsr/mni_icbm152_t1_tal_nlin_sym_09c.nii"



# def save_registration_to_MNI_freesurfer(img_path_name, out_path_name, reg_lta, verify=False, verbose=False, affine=False, mni_synthrs=False):
#     # mri_robust_register --mov moving_img.nii --dst MNI_img.nii --mapmov output_img.nii --lta output_reg_stats.lta  --weights v1to2-weights.mgz --iscale --satit
#     if mni_synthrs:
#         mni_template = MNI_T1W_SYNTHSR_PATH_NAME
#     else:
#         mni_template = MNI_T1W_PATH_NAME

#     if verify:
#         exist = os.path.exists(out_path_name)
#         if exist:
#             print(f"SynthSR already done for: {img_path_name}\nfile in: {out_path_name}")
#             return True
        
#     command = ["mri_robust_register", 
#                "--mov", img_path_name, 
#                "--dst", mni_template, 
#                "--mapmov", out_path_name, 
#                "--iscale", 
#                "--satit", 
#             #    "--affine", 
#                "--lta", reg_lta]
#     if affine:
#         command.append("--affine")
#     result = subprocess.run(command, capture_output=True, text=True)

#     if result.returncode == 0:
#         if verbose:
#             print(f"MNI registration done saved in: {out_path_name}")
#     else:
#         # Print error message
#         print("Error:", result.stderr)


# def save_registration_to_MNI_freesurfer_easyreg(img_path_name, out_path_name, ref_reg, df_forw, df_back, verify=False, verbose=False, mni_synthrs=False):
#     """mri_easyreg --ref <reference_image> --flo <floating_image>  \
#                 --ref_seg <ref_image_segmentation> --flo_seg <flo_image_segmentation>  \
#                 --ref_reg [deformed_ref_image] --flo_reg <deformed_flo_image>  \
#                 --fwd_field [forward_field] --bak_field <backward_field>  \
#                 --threads <number_of_threads> --affine_only"""

#     if mni_synthrs:
#         mni_template = MNI_T1W_SYNTHSR_PATH_NAME
#     else:
#         mni_template = MNI_T1W_PATH_NAME

#     if verify:
#         exist = os.path.exists(out_path_name)
#         if exist:
#             print(f"SynthSR already done for: {img_path_name}\nfile in: {out_path_name}")
#             return True
        
#     command = ["mri_easyreg", 
#                "--ref", mni_template, 
#                "--flo", img_path_name, 
#                "--flo_reg", out_path_name, 
#                "--ref_reg", ref_reg, 
#                "--fwd_field", df_forw, 
#                "--bak_field", df_back, 
#                "--threads", "84", 
#                "--affine_only"]

#     result = subprocess.run(command, capture_output=True, text=True)

#     if result.returncode == 0:
#         if verbose:
#             print(f"MNI registration done saved in: {out_path_name}")
#     else:
#         # Print error message
#         print("Error:", result.stderr)


