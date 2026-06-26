
import os
from tqdm import tqdm
import pandas as pd
import sys
import glob


sys.path.append('/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/utils')

import perp_segmentation_and_resampling as perp_segmentation_and_resampling


# def segment_and_resample(pred_img_path, save_path_name, verify=True, algorithm="synthseg"):
    
#     if verify and os.path.exists(save_path_name):
#         return

#     non_resampled_path_name = save_path_name.replace(".nii.gz", "_non_resampled.nii.gz")

#     if algorithm == "synthseg":
#         prep_segmentation.save_synthseg_segmentation(
#             pred_img_path,
#             non_resampled_path_name,
#             verify=verify,
#             verbose=False,
#             cortical_parcelation=False
#         )
#     elif algorithm == "supersynth":
#         supersynth_seg_path = non_resampled_path_name.replace(".nii.gz", "_supersynth")
#         prep_supersynth.save_supersynth(
#             pred_img_path,
#             supersynth_seg_path,
#             verify=verify,
#             verbose=False,
#             convert_to_nifti=True,
#             remove_mgz=True,
#             keep_only_segmentation=True
#         )
#         # copy the file called segmentation.nii.gz created by save_supersynth to its parent folder with the name non_resampled_path_name
#         shutil.copyfile(os.path.join(supersynth_seg_path, "segmentation.nii.gz"), non_resampled_path_name)
#         # remove the supersynth_seg_path folder
#         shutil.rmtree(supersynth_seg_path)
        
#     # resampled_path_name = save_path_name.replace(".nii.gz", "_seg.nii.gz")
#     prep_vol2vol.apply_vol2vol(
#         pred_img_path,
#         non_resampled_path_name,
#         save_path_name,
#         verify=verify,
#         verbose=False,
#         nearest=True
#     )

#     # remove the intermediate segmentation file
#     try:
#         os.remove(non_resampled_path_name)
#     except Exception as e:            
#         pass


def create_df_task1(pred_path_name, modalitites, resolutions):
    data = []
    for modality in modalitites:
        for resolution in resolutions:
            pred_path = os.path.join(pred_path_name, modality, f"{resolution}T_to_7T", "pred")
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



def segment_task1(pred_path_name, segment_supersynth=False):
    pred_path_name = os.path.join(pred_path_name, "task1")

    modalitites = ["T1W", "T2W", "T2FLAIR"]
    resolutions = [0.1, 1.5, 3, 5]

    df_val_task1 = create_df_task1(pred_path_name, modalitites, resolutions)

    bar = tqdm(df_val_task1.iterrows(), total=len(df_val_task1), desc="Segmenting Task 1")
    # filter 
    for i, row in df_val_task1.iterrows():    

        modality = row["modality"]
        resolution = row["resolution"]
        iid = row["iid"]
        pred_img_path = row["pred_img_path"]

        if resolution not in [0.1, 1.5]:
            resolution = int(resolution)
        save_path = os.path.join(pred_path_name, modality, f"{resolution}T_to_7T", "seg")
        save_name = iid.replace(f"{resolution}T", f"{7}T") + "_seg" + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        bar.set_description(f"Segmenting Task 1 - sid: {row['sid']} Modality {modality} = Resolution {resolution}T")


        # if os.path.exists(save_path_name):
        #     bar.update(1)  
        #     continue

        perp_segmentation_and_resampling.segment_and_resample(pred_img_path, save_path_name)
        
        if segment_supersynth:
            # also create the segmentation using supersynth and save it in the same folder with the name save_name.replace("_seg.nii.gz", "_seg_supersynth.nii.gz")
            supersynth_save_path_name = save_path_name.replace("_seg.nii.gz", "_seg_supersynth.nii.gz")
            perp_segmentation_and_resampling.segment_and_resample(pred_img_path, supersynth_save_path_name, algorithm="supersynth")
    
        bar.update(1)  

    bar.close()
    
    





def create_df_task2(pred_path_name, modalitites, resolutions):
    data = []
    for modality in modalitites:
        for resolution in resolutions:
            pred_path = os.path.join(pred_path_name, modality, f"0.1T_to_{resolution}T", "pred")
            pred_files = glob.glob(os.path.join(pred_path, "*.nii.gz"))
            for pred_file in pred_files:
                iid = os.path.basename(pred_file).replace(".nii.gz", "")
                sid = iid.split("_")[-1]
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




    
def segment_task2(pred_path_name, segment_supersynth=False):
    pred_path_name = os.path.join(pred_path_name, "task2")

    modalitites = ["T1W", "T2W", "T2FLAIR"]
    resolutions = [1.5, 3, 5, 7]

    df_val_task2 = create_df_task2(pred_path_name, modalitites, resolutions)

    bar = tqdm(df_val_task2.iterrows(), total=len(df_val_task2), desc="Segmenting Task 2")
    # filter 
    for i, row in df_val_task2.iterrows():    

        modality = row["modality"]
        resolution = row["resolution"]
        iid = row["iid"]
        pred_img_path = row["pred_img_path"]

        if resolution not in [0.1, 1.5]:
            resolution = int(resolution)
        save_path = os.path.join(pred_path_name, modality, f"0.1T_to_{resolution}T", "seg")
        save_name = iid.replace(f"0.1T", f"{resolution}T") + "_seg" + ".nii.gz"
        save_path_name = os.path.join(save_path, save_name)

        bar.set_description(f"Segmenting Task 2 - sid: {row['sid']} Modality {modality} = Resolution {resolution}T")

        perp_segmentation_and_resampling.segment_and_resample(pred_img_path, save_path_name)
        
        if segment_supersynth:
            # also create the segmentation using supersynth and save it in the same folder with the name save_name.replace("_seg.nii.gz", "_seg_supersynth.nii.gz")
            supersynth_save_path_name = save_path_name.replace("_seg.nii.gz", "_seg_supersynth.nii.gz")
            perp_segmentation_and_resampling.segment_and_resample(pred_img_path, supersynth_save_path_name, algorithm="supersynth")

        bar.update(1)  

    bar.close()
    


def segment_task2_test(pred_path_name):
    pred_path_name = os.path.join(pred_path_name, "task2")

    modalitites = ["T1W", "T2W", "T2FLAIR"]
    # resolutions = [1.5, 3, 5, 7]
    resolutions = [3, 5, 7]

    df_val_task2 = create_df_task2(pred_path_name, modalitites, resolutions)

    # filter 
    def create_input_output_txt(temp_path):
        input_list = []
        output_list = []
        for i, row in df_val_task2.iterrows():    

            modality = row["modality"]
            resolution = row["resolution"]
            iid = row["iid"]
            pred_img_path = row["pred_img_path"]

            if resolution not in [0.1, 1.5]:
                resolution = int(resolution)
            save_path = os.path.join(pred_path_name, modality, f"0.1T_to_{resolution}T", "seg")
            save_name = iid.replace(f"0.1T", f"{resolution}T") + "_seg_non_resampled" + ".nii.gz"
            save_path_name = os.path.join(save_path, save_name)
            
            input_list.append(pred_img_path)
            output_list.append(save_path_name)
        
        # save lists as txt files
        input_path = os.path.join(temp_path, "input_list.txt")
        output_path = os.path.join(temp_path, "output_list.txt")
        with open(input_path, "w") as f:
            for item in input_list:
                f.write("%s\n" % item)
        with open(output_path, "w") as f:
            for item in output_list:
                f.write("%s\n" % item)
        return input_path, output_path
                
    ### SEGMENT
    temp_folder = os.path.join(pred_path_name, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    input_path, output_path = create_input_output_txt(temp_folder)
    
    prep_segmentation.save_synthseg_segmentation(
            input_path,
            output_path,
            verify=False,
            verbose=True,
            cortical_parcelation=False
    )


    ### RESAMPLE

    with open(input_path, "r") as f:
        input_paths = f.read().splitlines()
    with open(output_path, "r") as f:
        output_paths = f.read().splitlines()   

            
    bar = tqdm(df_val_task2.iterrows(), total=len(df_val_task2), desc="Segmenting Task 2")
    
    for input_path, non_resampled_path_name in zip(input_paths, output_paths):
        # bar.set_description(f"Segmenting Task 2 - sid: {row['sid']} Modality {modality} = Resolution {resolution}T")

        save_path_name = non_resampled_path_name.replace("_seg_non_resampled.nii.gz", "_seg.nii.gz")
        if os.path.exists(save_path_name):
            bar.update(1)  
            continue

        non_resampled_path_name = save_path_name.replace(".nii.gz", "_non_resampled.nii.gz")

        # resampled_path_name = save_path_name.replace(".nii.gz", "_seg.nii.gz")
        prep_vol2vol.apply_vol2vol(
            input_path,
            non_resampled_path_name,
            save_path_name,
            verify=True,
            verbose=False,
            nearest=True
        )

        # remove the intermediate segmentation file
        try:
            os.remove(non_resampled_path_name)
        except Exception as e:            
            pass
    
        bar.update(1)  
    # reemove temp folder
    try:
        # shutil.rmtree(temp_folder)
        print("removing temp folder")
    except Exception as e:
        pass
    bar.close()
    


if __name__ == "__main__":
    # output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_synthsr_prior/results/val"
    output_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/evaluation/test5_segmentation_synthsr_prior/results/train"

    use_controlnet = False
    use_synthsr = True

    dm_chk_number = 140000
    n_inference_steps = 30
    cut_step = None

    dm_seg_channels = 3
    used_mask = f"merged_{dm_seg_channels}"

    if not use_controlnet:
        output_path = os.path.join(output_path, f"basic", used_mask, f"chk_{dm_chk_number}_steps_{n_inference_steps}")
    else:
        # dm_chk_number = 210000
        cnet_chk_number = 100000
        controlnet_chk_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/experiments/test5_segmentation_prior/training/models/all_357t/segconcatenated_controlnet/test1/check_points/model_{cnet_chk_number}.pt"
        output_path = os.path.join(output_path, f"controlnet", f"chk_{dm_chk_number}_cnchk_{cnet_chk_number}_steps_{n_inference_steps}")
        # raise NotImplementedError("ControlNet is not implemented yet for the evaluation, but it will be in the future. For now, just set use_controlnet to False if you want to run the evaluation.")

    if cut_step is not None:
        output_path = os.path.join(output_path, "tests", f"cut_step_{cut_step}")

    if use_synthsr:
        output_path = os.path.join(output_path, "with_synthsr")
    else:
        output_path = os.path.join(output_path, "without_synthsr")



    segment_task1(output_path, segment_supersynth=True)
    segment_task2(output_path, segment_supersynth=True)
    # segment_task2_test(output_path)



