
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import sys
import os
import numpy as np
import pandas as pd

sys.path.append('/home/agustin/phd/synthesis')
import utils.nifti_functions as nfc
import utils.util as util


# pytorch
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')

import prep_image as prep_image
import prep_histogram as prep_histogram
# from scipy.ndimage import zoom


Z_CLIP_RANGE = (150, 180)


def load_mean_histograms_global_tissuewise(output_dir, modalitites=["T1W", "T2W", "T2FLAIR"], resolutions=[0.1, 1.5, 3, 5, 7]):
    mean_histograms_global_tw = {}
    for _modality in modalitites:
        mean_histograms_global_tw[_modality] = {}
        for _resolution in resolutions:
            mean_histograms_global_tw[_modality][_resolution] = np.load(os.path.join(output_dir, f"{_modality}_{_resolution}.npy"))
    return mean_histograms_global_tw

output_dir = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/postprocessing/histogram_matching/mean_histograms/global"
mean_histograms_global  = load_mean_histograms_global_tissuewise(output_dir, modalitites=["T1W", "T2W", "T2FLAIR"], resolutions=[0.1,1.5, 3, 5, 7]) 


df_val = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv")



def process_file(src_file: Path, src_root: Path, dst_root: Path,
                 z_clip_range: tuple[int, int]) -> None:

    rel_path = src_file.relative_to(src_root)
    dst_file = dst_root / rel_path
    # verify if file already exists
    if dst_file.exists():
        print(f"Skipping existing file: {dst_file}")
        return str(dst_file)
    dst_file.parent.mkdir(parents=True, exist_ok=True)

    img, aff = nfc.load_nifti(str(src_file))
    # print(f"min: {img.min()}, max: {img.max()}, mean: {img.mean()}, std: {img.std()}")
    iid = os.path.basename(src_file).replace(".nii.gz", "")  
    modality = iid.split("_")[1]
    # resolution = float(iid.split("_")[2].replace("T", ""))
    # print(os.path.basename(os.path.dirname(os.path.dirname(src_file))))

    from_to = os.path.basename(os.path.dirname(os.path.dirname(src_file)))
    # resolution = float().split("_")[0].replace("T", ""))
    src_resolution = float(from_to.split("_")[0].replace("T", ""))
    tar_resolution = float(from_to.split("_")[-1].replace("T", ""))
    sid = f"S{str(iid.split('_')[-1])}"

    if src_resolution not in [0.1, 1.5]:
        src_resolution = int(src_resolution)

    if tar_resolution not in [0.1, 1.5]:
        tar_resolution = int(tar_resolution)

    seg_path = df_val[(df_val["sid"] == sid) & (df_val["modality"] == modality) & (df_val["resolution"] == src_resolution)]["seg_supersynth_path"]
    # verify that seg_path is not empty
    if seg_path.empty:
        raise ValueError(f"No segmentation found for sid: {sid}, modality: {modality}, resolution: {src_resolution}")
    else:
        seg_path = seg_path.values[0]
    
    # print(f"SEG PATH: {seg_path}")
    seg, _ = nfc.load_nifti(seg_path)
    # print("performing histogram matching...")
    histogram_reference = mean_histograms_global[modality][tar_resolution]
    img_post = prep_histogram.match_hist_with_reference(
        img,
        histogram_reference,
        mask=seg > 0,  # Only consider the brain region for histogram matching
    )
    # z_start, z_end = z_clip_range
    # cropped_data = img[:, :, z_start:z_end]

    # print(f"Saving post-processed image to: {dst_file}")
    nfc.save_nifti(img_post, aff, str(dst_file))

    return str(dst_file)


def crop_z_slices(src_root: Path,
                  dst_root: Path,
                  z_clip_range: tuple[int, int],
                  num_workers: int | None = None):

    nifti_files = list(src_root.rglob("*.nii.gz"))

    print(f"Found {len(nifti_files)} files")

    if num_workers is None:
        # Typically good for I/O-heavy workloads
        num_workers = min(32, multiprocessing.cpu_count() * 2)
        # num_workers = 1

    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = {
            executor.submit(
                process_file,
                src_file,
                src_root,
                dst_root,
                z_clip_range,
            ): src_file
            for src_file in nifti_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                dst_file = future.result()
                print(f"[{i}/{len(nifti_files)}] Saved: {dst_file}")
            except Exception as e:
                print(f"ERROR processing {futures[future]}: {e}")


if __name__ == "__main__":
    input_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_synthsr_prior/results/val/basic/merged_3/chk_280000_steps_30/with_synthsr/task3"
    output_path = input_path + "_postprocessed"
    src_root = Path(input_path)
    dst_root = Path(output_path)

    crop_z_slices(
        src_root=src_root,
        dst_root=dst_root,
        z_clip_range=Z_CLIP_RANGE,
        num_workers=4,  # tune as needed
    )




# Z_CLIP_RANGE = (150, 180)

# from pathlib import Path
# import sys
# import nibabel as nib
# import numpy as np


# sys.path.append('/home/agustin/phd/synthesis')
# import utils.nifti_functions as nfc


# def crop_z_slices(src_root: Path, dst_root: Path, z_clip_range: tuple[int, int]):
#     # Find all NIfTI files
#     for src_file in src_root.rglob("*.nii.gz"):

#         # Compute corresponding destination path
#         rel_path = src_file.relative_to(src_root)
#         dst_file = dst_root / rel_path

#         # Create destination directories
#         dst_file.parent.mkdir(parents=True, exist_ok=True)

#         # Load image
#         img, aff = nfc.load_nifti(str(src_file))

#         # Example: crop 10 slices from beginning and end of Z axis
#         # cropped_data = data[:, :, 10:-10]
#         z_start, z_end = z_clip_range
#         cropped_data = img[:, :, z_start:z_end]

#         # nib.save(cropped_img, str(dst_file))
#         nfc.save_nifti(cropped_data, aff, str(dst_file))
#         print(f"Saved: {dst_file}")


# if __name__ == "__main__":
#     src_root = Path("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/basic/chk_200000/task1_res")
#     dst_root = Path("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/basic/chk_200000/task1_zclipped")
#     crop_z_slices(src_root, dst_root, Z_CLIP_RANGE)