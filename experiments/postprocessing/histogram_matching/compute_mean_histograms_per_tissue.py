
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

import sys

sys.path.append('/home/agustin/phd/synthesis')

# mine
import utils.nifti_functions as nfc
import utils.util as util
import utils.util_freesurfer_segmentation as ufs

# import prep_image as prep_image


from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


import numpy as np


def compute_image_reference_quantiles_tissuewise(
    img,
    seg,
    quantiles=np.linspace(0, 100, 1001),
    lower_clip=0,
    upper_clip=100,
    tissues_idx=None,
    min_voxels=10,
):

    if tissues_idx is None:
        tissues_idx = np.unique(seg)

    tissue_quantiles = []

    for tissue in tissues_idx:

        voxels = img[seg == tissue]

        if len(voxels) < min_voxels:
            print(f"Tissue {tissue} contains only {len(voxels)} voxels.")
            # raise ValueError(
            #     f"Tissue {tissue} contains only "
            #     f"{len(voxels)} voxels."
            # )

        low = np.percentile(voxels, lower_clip)
        high = np.percentile(voxels, upper_clip)

        voxels = voxels[
            (voxels >= low) &
            (voxels <= high)
        ]

        tissue_quantiles.append(
            np.percentile(voxels, quantiles)
        )

    return np.asarray(tissue_quantiles)



def process_single_image(row, quantiles):
    img = nfc.load_nifti(row["org_img_path"])[0]
    seg = nfc.load_nifti(row["seg_synthseg_path"])[0]
    seg = ufs.merge_seg96_to_seg3(seg)
    img[seg == 0] = 0
    img = util.robust_normalize(img, strictly_positive=True, mask=seg > 0)

    return compute_image_reference_quantiles_tissuewise(
        img,
        seg,
        quantiles=quantiles,
        lower_clip=0,
        upper_clip=100,
        tissues_idx=[1, 2, 3]
    )

def load_mean_histogram(
    training_df,
    modalities=["T1W", "T2W", "T2FLAIR"],
    resolutions=[0.1, 1.5, 3, 5, 7],
    max_images=None,
    num_workers=32,
):

    mean_histograms = {}
    quantiles = np.linspace(0, 100, 1001)

    if num_workers is None:
        num_workers = os.cpu_count()

    for modality in modalities:
        mean_histograms[modality] = {}

        for resolution in resolutions:

            possible_rows = training_df[
                (training_df["modality"] == modality) &
                (training_df["resolution"] == resolution)
            ]

            available = (
                min(len(possible_rows), max_images)
                if max_images is not None
                else len(possible_rows)
            )

            if available == 0:
                print(f"No images for {modality} {resolution}")
                mean_histograms[modality][resolution] = None
                continue

            if available < len(possible_rows):
                selected_rows = possible_rows.sample(n=available, random_state=42)
            else:
                selected_rows = possible_rows
                print(f"Using all {available} images for {modality} {resolution}")

            q_list = []

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(process_single_image, row, quantiles)
                    for _, row in selected_rows.iterrows()
                ]

                for f in tqdm(as_completed(futures),
                               total=available,
                               desc=f"{modality}-{resolution}"):

                    q = f.result()
                    if q is not None:
                        q_list.append(q)

            mean_histograms[modality][resolution] = (
                np.mean(q_list, axis=0) if len(q_list) > 0 else None
            )

    return mean_histograms


if __name__ == "__main__":
    training_df = pd.read_csv("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv")
    modalities =["T2W", "T2FLAIR"]
    resolutions =[0.1, 1.5, 3, 5, 7]
    max_images = None
    mean_histograms = load_mean_histogram(training_df, 
                                          modalities=modalities, 
                                          resolutions=resolutions, 
                                          max_images=max_images,
                                          num_workers=32)

    # save the histograms as np arrays
    output_dir = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/postprocessing/histogram_matching/mean_histograms/tissuewise_3"
    os.makedirs(output_dir, exist_ok=True)
    for modality in modalities:
        for resolution in resolutions:
            if mean_histograms[modality][resolution] is not None:
                np.save(os.path.join(output_dir, f"{modality}_{resolution}.npy"), mean_histograms[modality][resolution])


    # with open("/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/json/mean_histograms.json", "w") as f:
    #     json.dump(mean_histograms, f)


# def compute_image_reference_quantiles_tissuewise(
#     img,
#     seg,
#     quantiles=np.linspace(0, 100, 101),
#     lower_clip=1,
#     upper_clip=99,
#     tissues = {
#     "csf": 1,
#     "gm": 2,
#     "wm": 3,
#     }
# ):
#     """
#     Compute reference landmarks for each tissue.
#     """

#     reference = {}

#     for tissue, labels in tissues.items():
#         mask = np.isin(seg, labels)

#         voxels = img[mask]

#         if len(voxels) < 100:
#             continue

#         low = np.percentile(voxels, lower_clip)
#         high = np.percentile(voxels, upper_clip)

#         voxels = voxels[
#             (voxels >= low) &
#             (voxels <= high)
#         ]

#         q = np.percentile(voxels, quantiles)
#         reference[tissue] = q

#     return reference
