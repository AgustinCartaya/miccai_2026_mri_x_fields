
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import sys

sys.path.append('/home/agustin/phd/synthesis')
import utils.nifti_functions as nfc

Z_CLIP_RANGE = (150, 180)


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

    z_start, z_end = z_clip_range
    cropped_data = img[:, :, z_start:z_end]

    nfc.save_nifti(cropped_data, aff, str(dst_file))

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
    input_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_prior/results/val_ema/basic/merged_8/chk_145000_steps_30/task3"
    output_path = input_path + "_zclipped"
    src_root = Path(input_path)
    dst_root = Path(output_path)

    crop_z_slices(
        src_root=src_root,
        dst_root=dst_root,
        z_clip_range=Z_CLIP_RANGE,
        num_workers=16,  # tune as needed
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