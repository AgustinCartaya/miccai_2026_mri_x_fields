
import sys
import os
import subprocess


def save_synthSR(img_path_name, out_path_name, verify=False, verbose=False):
    if verify:
        exist = os.path.exists(out_path_name)
        if exist:
            print(f"SynthSR already done for: {img_path_name}\nfile in: {out_path_name}")
            return True
 
    command = ["mri_synthsr", 
               "--i", img_path_name, 
               "--o", out_path_name, 
               "--threads", "84"]
    
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        # Print the output
        if verbose:
            print(f"SynthSR done saved in: {out_path_name}")
        # print(result.stdout)
    else:
        # Print error message
        print("Error:", result.stderr)

# img_path_name = "/home/agustin/phd/synthesis/data/raw/sub-MIRIAD188_ses-01_acq-orig_run-01_T1w.nii.gz"
# out_path_name = "/home/agustin/phd/synthesis/data/preprocess/sub-MIRIAD188_ses-01_acq-orig_run-01_T1w_SynthSR_freesurfer.nii.gz"
# save_synthSR(img_path_name, out_path_name, verify=True)