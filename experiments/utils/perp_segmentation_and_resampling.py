import os
import shutil
import prep_segmentation as prep_segmentation
import prep_supersynth as prep_supersynth
import prep_vol2vol as prep_vol2vol 


def segment_and_resample(pred_img_path, save_path_name, verify=True, algorithm="synthseg"):
    
    if verify and os.path.exists(save_path_name):
        return

    non_resampled_path_name = save_path_name.replace(".nii.gz", "_non_resampled.nii.gz")

    if algorithm == "synthseg":
        prep_segmentation.save_synthseg_segmentation(
            pred_img_path,
            non_resampled_path_name,
            verify=verify,
            verbose=False,
            cortical_parcelation=False
        )
    elif algorithm == "supersynth":
        supersynth_seg_path = non_resampled_path_name.replace(".nii.gz", "_supersynth")
        prep_supersynth.save_supersynth(
            pred_img_path,
            supersynth_seg_path,
            verify=verify,
            verbose=False,
            convert_to_nifti=True,
            remove_mgz=True,
            keep_only_segmentation=True
        )
        # copy the file called segmentation.nii.gz created by save_supersynth to its parent folder with the name non_resampled_path_name
        shutil.copyfile(os.path.join(supersynth_seg_path, "segmentation.nii.gz"), non_resampled_path_name)
        # remove the supersynth_seg_path folder
        shutil.rmtree(supersynth_seg_path)
        
    # resampled_path_name = save_path_name.replace(".nii.gz", "_seg.nii.gz")
    prep_vol2vol.apply_vol2vol(
        pred_img_path,
        non_resampled_path_name,
        save_path_name,
        verify=verify,
        verbose=False,
        nearest=True
    )

    # remove the intermediate segmentation file
    try:
        os.remove(non_resampled_path_name)
    except Exception as e:            
        pass