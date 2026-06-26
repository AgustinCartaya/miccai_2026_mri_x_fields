import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import sys


# pytorch
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')


# import prep_synthsr as prep_synthsr
import prep_vol2vol as prep_vol2vol 
import prep_registration as prep_registration 



from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import os


USLR_PATH = "/home/agustin/phd/synthesis/tests/D3/preprocessing/brainst_preprocessing_pipeline/preprocessing/USLR"
os.environ["PYTHONPATH"] = os.path.join(USLR_PATH)
sys.path.append(os.path.dirname(USLR_PATH))

from USLR.utils.fn_utils import compute_centroids_ras
from USLR.utils import synthmorph_utils


TARGET_IMG_PATH = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/mni/P_T1W_3T_0006.nii.gz"
TARGET_SEG_PATH = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/mni/P_T1W_3T_0006_seg.nii.gz"
CENTROID_ATLAS, _ = compute_centroids_ras(TARGET_SEG_PATH, synthmorph_utils.labels_registration)


import nibabel as nib

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


def process_row_uslr(index, row, base_img_output_path, base_seg_output_path):

    iid = row['iid']
    resolution = row['resolution']
    modality = row['modality']
    img_moving_path = row['raw_org_img_path']
    seg_moving_path = row['raw_seg_synthseg_path']

    try:
        # T1W processing
        img_output_path = os.path.join(
            base_img_output_path,
            modality,
            f"{resolution}T"
        )

        seg_output_path = os.path.join(
            base_seg_output_path,
            modality,
            f"{resolution}T"
        )

        os.makedirs(img_output_path, exist_ok=True)
        os.makedirs(seg_output_path, exist_ok=True)

        img_output_name = f"{iid}_interp.nii.gz"
        img_output_path_name = os.path.join(img_output_path,img_output_name)

        seg_output_name = f"{iid}_seg.nii.gz"
        seg_output_path_name = os.path.join(seg_output_path,seg_output_name)


        centroid_sbj, ok = compute_centroids_ras(seg_moving_path, synthmorph_utils.labels_registration)
        M_ref = synthmorph_utils.getM(CENTROID_ATLAS[:, ok > 0], centroid_sbj[:, ok > 0], use_L1=False)

        if not os.path.exists(img_output_path_name):

            # perform skull stripping before registration to MNI space
            img_output_path_name_skullstrip = img_output_path_name.replace(".nii.gz", "_skullstrip.nii.gz")
            org_img, aff = load_nifti(img_moving_path)
            org_seg, _ = load_nifti(seg_moving_path)
            org_img[org_seg == 0] = 0
            save_nifti(org_img, aff, img_output_path_name_skullstrip)


            transform_matrix_temp_to_mni = prep_registration.ras_to_lps_affine(M_ref)
            prep_registration.save_precomputed_rigid_registration(
                TARGET_IMG_PATH,
                img_output_path_name_skullstrip, #row["t1w_raw_path"],
                img_output_path_name,
                transform_matrix_temp_to_mni,
                verify=True
            )

            # remove the skull stripped image
            try:
                os.remove(img_output_path_name_skullstrip)
            except:
                pass


            prep_registration.save_precomputed_rigid_registration(
                TARGET_SEG_PATH,
                seg_moving_path, #row["t1w_raw_path"],
                seg_output_path_name,
                transform_matrix_temp_to_mni,
                verify=True,
                is_label=True
            )

            

        return index, img_output_path_name, seg_output_path_name

    except Exception as e:

        print(f"Failed {iid}: {e}")

        return index, None, None



def process_row(index, row, base_img_output_path, base_seg_output_path):

    iid = row['iid']
    resolution = row['resolution']
    modality = row['modality']
    img_moving_path = row['raw_org_img_path']
    seg_moving_path = row['raw_seg_synthseg_path']

    try:
        # T1W processing
        img_output_path = os.path.join(
            base_img_output_path,
            modality,
            f"{resolution}T"
        )

        seg_output_path = os.path.join(
            base_seg_output_path,
            modality,
            f"{resolution}T"
        )

        os.makedirs(img_output_path, exist_ok=True)
        os.makedirs(seg_output_path, exist_ok=True)

        img_output_name = f"{iid}_interp.nii.gz"
        img_output_path_name = os.path.join(img_output_path,img_output_name)

        seg_output_name = f"{iid}_seg.nii.gz"
        seg_output_path_name = os.path.join(seg_output_path,seg_output_name)

        # Run SynthSeg only if needed
        if not os.path.exists(img_output_path_name):

            prep_vol2vol.apply_vol2vol(
                TARGET_IMG_PATH,
                img_moving_path,
                img_output_path_name,
                verify=True,
                verbose=False,
                # nearest=True
            )

            prep_vol2vol.apply_vol2vol(
                TARGET_IMG_PATH,
                seg_moving_path,
                seg_output_path_name,
                verify=True,
                verbose=False,
                nearest=True
            )


        return index, img_output_path_name, seg_output_path_name

    except Exception as e:

        print(f"Failed {iid}: {e}")

        return index, None, None
    
def preprocess_supersynth(split="claras", num_workers=4):

    csv_path = (
        f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"
    )

    df = pd.read_csv(csv_path)
    # df = df.head(1)

    base_img_output_path = (f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/interpolated_imgs")
    base_seg_output_path = (f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/interpolated_synthseg")

    # Preallocate to preserve order
    interpolated_img_paths = [None] * len(df)
    interpolated_seg_paths = [None] * len(df)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = []

        for index, row in df.iterrows():

            futures.append(
                executor.submit(
                    process_row_uslr,
                    index,
                    row,
                    base_img_output_path,
                    base_seg_output_path
                )
            )

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preprocessing"
        ):

            index, img_path, seg_path = future.result()

            interpolated_img_paths[index] = img_path
            interpolated_seg_paths[index] = seg_path

    df["org_img_path"] = interpolated_img_paths
    df["seg_synthseg_path"] = interpolated_seg_paths
    df["seg_supersynth_path"] = interpolated_seg_paths

    output_csv_path = (
        f"/home/agustin/phd/miccai/miccai_2026/"
        f"mri_x_fields/data/csv/"
        f"{split}_data_interpolated.csv"
    )

    df.to_csv(output_csv_path, index=False)



    

if __name__ == "__main__":
    preprocess_supersynth(split="claras", num_workers=4)