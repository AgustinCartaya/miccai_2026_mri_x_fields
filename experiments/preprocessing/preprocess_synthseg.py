import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import sys


# pytorch
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')


import prep_segmentation as prep_segmentation
import prep_vol2vol as prep_vol2vol 


# def preprocess_supersynth(split="train"):
#     csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"
#     df = pd.read_csv(csv_path)#.head(10)

#     base_output_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/{split}_data/preprocessed/synthseg"
#     bar = tqdm(total=df.shape[0], desc="Preprocessing images with mri_synthseg")
#     seg_paths = []
#     for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#         subject_id = row['subject_id']
#         resolution = row['resolution']
#         modality = row['modality']
#         # split = row['split']
#         img_path = row['path']

#         print("init")
#         # for the moment we will segment just the T1w images, but we can easily extend this to other modalities in the future
#         if modality != "T1W":
#             output_path = os.path.join(base_output_path, "T1W", f"{str(resolution)}T")
#             output_name = f"{subject_id}_synthseg_resampled.nii.gz"
#             output_path_name = os.path.join(output_path, output_name)
#             seg_paths.append(output_path_name)
#             bar.update(1)
#             continue




#         # print(type(subject_id), type(resolution), type(modality), type(split), type(img_path))
#         output_path = os.path.join(base_output_path, modality, f"{str(resolution)}T")
#         # output_name = f"{modality}_{resolution}T_{subject_id[1:]}_synthseg.nii.gz"
#         output_name = f"{subject_id}_synthseg.nii.gz"
#         output_path_name = os.path.join(output_path, output_name)
#         resampled_path_name = output_path_name.replace(".nii.gz", "_resampled.nii.gz")
#         os.makedirs(output_path, exist_ok=True)

#         if not os.path.exists(resampled_path_name):
#             prep_segmentation.save_synthseg_segmentation(img_path, output_path_name, verify=True, verbose=False, cortical_parcelation=False)
        
#         prep_vol2vol.apply_vol2vol(img_path, output_path_name, resampled_path_name, verify=True, verbose=False, nearest=True)
        
#         # remove the original segmentation file
#         try:
#             os.remove(output_path_name)
#         except Exception as e:
#             pass
#         seg_paths.append(resampled_path_name)

#         bar.update(1)


#     bar.close()

#     # save the dataframe to a new csv file
#     df["segmentation_path"] = seg_paths
#     output_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data_with_synthseg_segmentation.csv"
#     df.to_csv(output_csv_path, index=False)



from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import os


def process_row(index, row, base_output_path):

    subject_id = row['subject_id']
    resolution = row['resolution']
    modality = row['modality']
    img_path = row['path']

    try:

        # # Reuse T1W segmentation
        # if modality != "T1W":

        #     output_path = os.path.join(
        #         base_output_path,
        #         "T1W",
        #         f"{resolution}T"
        #     )

        #     # dirty trick
        #     subject_id = subject_id.replace(f"_{modality}_", "_T1W_")

        #     output_name = f"{subject_id}_synthseg_resampled.nii.gz"

        #     output_path_name = os.path.join(
        #         output_path,
        #         output_name
        #     )

        #     return index, output_path_name

        # T1W processing
        output_path = os.path.join(
            base_output_path,
            modality,
            f"{resolution}T"
        )

        os.makedirs(output_path, exist_ok=True)

        output_name = f"{subject_id}_synthseg.nii.gz"

        output_path_name = os.path.join(
            output_path,
            output_name
        )

        resampled_path_name = output_path_name.replace(
            ".nii.gz",
            "_resampled.nii.gz"
        )

        # Run SynthSeg only if needed
        if not os.path.exists(resampled_path_name):

            prep_segmentation.save_synthseg_segmentation(
                img_path,
                output_path_name,
                verify=True,
                verbose=False,
                cortical_parcelation=False
            )

            prep_vol2vol.apply_vol2vol(
                img_path,
                output_path_name,
                resampled_path_name,
                verify=True,
                verbose=False,
                nearest=True
            )

            try:
                os.remove(output_path_name)
            except:
                pass

        return index, resampled_path_name

    except Exception as e:

        print(f"Failed {subject_id}: {e}")

        return index, None


def preprocess_supersynth(split="train", num_workers=4):

    csv_path = (
        f"/home/agustin/phd/miccai/miccai_2026/"
        f"mri_x_fields/data/csv/{split}_data.csv"
    )

    df = pd.read_csv(csv_path)

    base_output_path = (
        f"/home/agustin/phd/miccai/miccai_2026/"
        f"mri_x_fields/data/{split}_data/preprocessed/synthseg"
    )

    # Preallocate to preserve order
    seg_paths = [None] * len(df)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:

        futures = []

        for index, row in df.iterrows():

            futures.append(
                executor.submit(
                    process_row,
                    index,
                    row,
                    base_output_path
                )
            )

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Preprocessing"
        ):

            index, seg_path = future.result()

            seg_paths[index] = seg_path

    df["segmentation_path"] = seg_paths

    output_csv_path = (
        f"/home/agustin/phd/miccai/miccai_2026/"
        f"mri_x_fields/data/csv/"
        f"{split}_data_with_synthseg_segmentation.csv"
    )

    df.to_csv(output_csv_path, index=False)



    

if __name__ == "__main__":
    preprocess_supersynth(split="pr_train")