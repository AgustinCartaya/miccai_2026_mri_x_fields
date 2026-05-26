
import sys
import os
import subprocess

import nibabel as nib
import numpy as np

# import sys
# sys.path.append('/home/agustin/phd/synthesis')
# import utils.nifti_functions as nfc

# normal = 31seg
# not robust = 28seg 
# no robust no cortical_parcelation = 23seg


def load_nifti(path_name, transpose=False):
    imag_nifti = nib.load(path_name)
    img_data = imag_nifti.get_fdata()
    if transpose:
        img_data = np.transpose(img_data, (1, 0, 2))
    return img_data, (imag_nifti.affine, imag_nifti.header)

def save_nifti(image_np, affine, img_path_name, transpose=False):
    if transpose:
        image_np = np.transpose(image_np, (1, 0, 2)) 

    img_nifti = nib.Nifti1Image(image_np, affine=affine[0], header=affine[1])
    nib.save(img_nifti, img_path_name)



def save_supersynth(img_path_name, out_path, verify=False, verbose=False, convert_to_nifti=True, remove_mgz=True, keep_only_segmentation=True):
    """Run mri_synthseg to get the segmentation of the input image.
    Args:
        img_path_name (str): Path to the input image.
        out_path_name (str): Path to save the output segmentation.
        verify (bool): If True, check if the output file already exists.
        verbose (bool): If True, print additional information.

    Returns:
        None: The function saves the segmentation to the specified output path.
    Example:
        prep_segmentation.save_synthseg_segmentation(
                                img_path_name=path_name_raw_t1w,
                                out_path_name=path_name_seg,
                                verify=True,
                                verbose=True,
                            )
    """
    
    # mri_super_synth --i <input> --o <output> --mode invivo --threads 16
    if verify:
        # verify if there are .nii.gz or .mgz files in the folder out_path_name
        # find files in the folder out_path_name that end with .nii.gz or .mgz
        if os.path.exists(out_path):
            files_in_out_path = os.listdir(out_path)
            synthsr_files = [f for f in files_in_out_path if f.endswith(".nii.gz") or f.endswith(".mgz")]
            if len(synthsr_files) > 0:
                print("#" * 50)
                print(synthsr_files)
                print(f"Supersynth already done for: {img_path_name}\nfile in: {out_path}")
                return True

 
    command = [#"/home/Code/freesurfer-supersynth/bin/mri_super_synth",
                 "mri_super_synth", 
                "--i", img_path_name, 
                "--o", out_path, 
                "--mode", "invivo",
                # "--robust",
                # "--cpu",
                "--threads", "16"
                ]
    
    command_str = " ".join(command)
    if verbose:
        print(f"Running command: {command_str}")
#     export FREESURFER_HOME=/home/Code/freesurfer-supersynth
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
    # verify if output path exists and if not, create it

    os.makedirs(out_path, exist_ok=True)
    # command_1 = "export FREESURFER_HOME=/home/Code/freesurfer-supersynth && source $FREESURFER_HOME/SetUpFreeSurfer.sh"
    # result = subprocess.run(command_1, capture_output=True, text=True)
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        if verbose:
            print(f"synthseg done saved in: {out_path}")

        if keep_only_segmentation:
            # remove all files in the output folder except the segmentation.nii.gz file
            for f in os.listdir(out_path):
                if f != "segmentation.mgz":
                    os.remove(os.path.join(out_path, f))
            
        if convert_to_nifti:
            # find all .mgz files in the output folder
            files = [f for f in os.listdir(out_path) if f.endswith(".mgz")]
            # load each .mgz file and save as nifti
            for f in files:
                img, aff = load_nifti(os.path.join(out_path, f))
                save_nifti(img, aff, os.path.join(out_path, f.replace(".mgz", ".nii.gz")))
            
            if remove_mgz:
                # remove all .mgz files
                for f in files:
                    os.remove(os.path.join(out_path, f))

    else:
        # Print error message
        print("Error:", result.stderr)


