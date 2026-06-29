import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import random
import sys


# pytorch
sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')


import prep_synthsr as prep_synthsr
import prep_vol2vol as prep_vol2vol 


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import os


def process_row(index, row, base_output_path):

    iid = row['iid']
    resolution = row['resolution']
    modality = row['modality']
    img_path = row['org_img_path']

    try:

        # T1W processing
        output_path = os.path.join(
            base_output_path,
            modality,
            f"{resolution}T"
        )

        os.makedirs(output_path, exist_ok=True)

        output_name = f"{iid}_synthsr.nii.gz"

        output_path_name = os.path.join(
            output_path,
            output_name
        )

        no_resampled_path_name = output_path_name.replace(
            ".nii.gz",
            "_noresampled.nii.gz"
        )

        # Run SynthSeg only if needed
        if not os.path.exists(output_path_name):

            prep_synthsr.save_synthSR(
                img_path,
                no_resampled_path_name,
                verify=True,
                verbose=False,
            )

            prep_vol2vol.apply_vol2vol(
                img_path,
                no_resampled_path_name,
                output_path_name,
                verify=True,
                verbose=False,
                # nearest=True
            )

            try:
                os.remove(no_resampled_path_name)
            except:
                pass

        return index, output_path_name

    except Exception as e:

        print(f"Failed {iid}: {e}")

        return index, None


def preprocess_supersynth(split="train", num_workers=4):

    csv_path = (
        f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/{split}_data.csv"
    )

    df = pd.read_csv(csv_path)
    # df = df.head(1)

    base_output_path = (
        f"/home/agustin/phd/miccai/miccai_2026/"
        f"mri_x_fields/data/{split}_data/preprocessed/synthsr"
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

    df["synthsr_path"] = seg_paths

    output_csv_path = (
        f"/home/agustin/phd/miccai/miccai_2026/"
        f"mri_x_fields/data/csv/"
        f"{split}_data_with_synthsr.csv"
    )

    df.to_csv(output_csv_path, index=False)



    

if __name__ == "__main__":
    preprocess_supersynth(split="paired_train")