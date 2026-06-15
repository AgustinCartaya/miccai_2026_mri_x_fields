
import os
from tqdm import tqdm
import pandas as pd
import sys
import glob


sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')

import perp_segmentation_and_resampling as perp_segmentation_and_resampling


def create_df(pred_path_name, modalitites, resolutions):
    data = []
    for modality in modalitites:
        for resolution in resolutions:
            pred_path = os.path.join(pred_path_name, modality, str(resolution), "pred")
            pred_files = glob.glob(os.path.join(pred_path, "*.nii.gz"))
            for pred_file in pred_files:
                iid = os.path.basename(pred_file).replace(".nii.gz", "")
                sid = "S" + iid.split("_")[-1]
                # org_img_path = pred_file.replace("pred", "org").replace(f"{resolution}T_to_7T", f"{resolution}T")
                # latent_seg_supersynth_path = org_img_path.replace("org", "latent_seg_supersynth").replace(".nii.gz", ".npy")
                data.append({
                    "sid": sid,
                    "iid": iid,
                    "modality": modality,
                    "resolution": resolution,
                    "pred_img_path": pred_file,
                    
                })
    df = pd.DataFrame(data)
    return df



def segment(pred_path_name, segment_supersynth=False):
    modalitites = ["T1W", "T2W", "T2FLAIR"]
    resolutions = [0.1, 1.5, 3, 5, 7]

    df_val_task1 = create_df(pred_path_name, modalitites, resolutions)

    bar = tqdm(df_val_task1.iterrows(), total=len(df_val_task1), desc="Segmenting reconstructed images")
    # filter 
    for i, row in df_val_task1.iterrows():    

        modality = row["modality"]
        resolution = row["resolution"]
        iid = row["iid"]
        pred_img_path = row["pred_img_path"]

        if resolution not in [0.1, 1.5]:
            resolution = int(resolution)
        save_path = os.path.join(pred_path_name, modality, str(resolution), "seg")
        save_name = iid + "_seg" + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        perp_segmentation_and_resampling.segment_and_resample(pred_img_path, save_path_name)
        
        if segment_supersynth:
            # also create the segmentation using supersynth and save it in the same folder with the name save_name.replace("_seg.nii.gz", "_seg_supersynth.nii.gz")
            supersynth_save_path_name = save_path_name.replace("_seg.nii.gz", "_seg_supersynth.nii.gz")
            perp_segmentation_and_resampling.segment_and_resample(pred_img_path, supersynth_save_path_name, algorithm="supersynth")
    
        bar.update(1)  

    bar.close()
    
    


if __name__ == "__main__":
    df_val = pd.read_csv(
        "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"
    )
    df_val = df_val[df_val["split"] == "val"]
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test1_vae/results/rec"


    segment(
        pred_path_name=output_path,
        segment_supersynth=False
    )
