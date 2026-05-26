import pandas as pd

import os


# # --------------------- this is to solve the ID problem ---------------------


def load_df(csv_path):
    if type(csv_path) == str:
        df = pd.read_csv(csv_path)
    else: # we assume that csv_path is already a dataframe
        df = csv_path
    return df
    
def add_subject_global_id(csv_path, save=True):

    df = load_df(csv_path)
    
    # verify that the subject_id column exists
    if 'subject_id' not in df.columns:
        raise ValueError("subject_id column not found in the csv file")
    df['sid'] = df['subject_id'].apply(lambda x: x.split("_")[-1])

    # the paired colum is 1 if df['subject_id'].apply(lambda x: x.split("_")[0] starts with "SP") and 0 otherwise
    df['paired'] = df['subject_id'].apply(lambda x: 1 if x.split("_")[0].startswith("SP") else 0)

    # rename subject id column to iid (image id) and move it to the end of the dataframe
    df = df.rename(columns={'subject_id': 'iid'})

    # important columsn order = [sid,paired,resolution,modality,split,subject_id]
    columns_order = ['sid', 'paired', 'resolution', 'modality', 'split', 'iid']
    df = df[columns_order + [col for col in df.columns if col not in columns_order]]    

    # sort by sid, resolution, modality
    df = df.sort_values(by=['paired', 'sid', 'resolution', 'modality'])
    
    # add a S before the sid to make it clear that it is a subject id and not an image id
    df['sid'] = df['sid'].apply(lambda x: "S" + x)


    # print(df.head())
    if save:
        if type(csv_path) == str:
            output_csv_path = csv_path.replace(".csv", "_sid.csv")
            df.to_csv(output_csv_path, index=False)
        else:
            raise ValueError("csv_path must be strings if save is True")
    
    return df


def order_columns_and_rename(csv_path, save=True):
    expected_columns_order_and_names = ['sid', 'paired', 'resolution', 'modality', 'split', 'iid', 'org_img_path', 'seg_synthseg_path', 'seg_supersynth_path', 'latent_path', 'latent_norm_wm_path', 'latent_seg_synthseg_path', 'latent_seg_supersynth_path']
    mapping_rename =  {
        "path": "org_img_path",
        "latent_path": "latent_path",
        "segmentation_path": "seg_synthseg_path",
        "segmentation_supersynth_path": "seg_supersynth_path",
        "latent_seg_supersynth_mask": "latent_seg_supersynth_path",
        "latent_seg_mask": "latent_seg_synthseg_path",
        "latent_normalized_wm_path": "latent_norm_wm_path",
        
    }
    
    
    df = load_df(csv_path)
    
    # rename columns according to the mapping
    df = df.rename(columns=mapping_rename)
    
    # order the available columns according to the expected columns order and names, if some of the expected columns are not in the dataframe, remove them from the expected columns list and order the remaining columns, if some of the columns in the dataframe are not in the expected columns list, keep them at the end of the dataframe
    available_columns = df.columns.tolist()
    expected_columns_order_and_names = [col for col in expected_columns_order_and_names if col in available_columns]
    # add at the end the columns in available_columns that are not in expected_columns_order_and_names
    expected_columns_order_and_names += [col for col in available_columns if col not in expected_columns_order_and_names]
    df = df[expected_columns_order_and_names]
    
    
    # sort by sid, resolution, modality
    df = df.sort_values(by=['paired', 'sid', 'resolution', 'modality'])
    
    if save: 
        if type(csv_path) == str:
            csv_path = csv_path.replace(".csv", "_ordered.csv")
            df.to_csv(csv_path, index=False)
        else:
            raise ValueError("csv_path must be strings if save is True")
    return df


def merge_base_dataset_with_other_dataset_column(base_csv_path, other_csv_path, key_columns, merge_columns, save=True):
    
    df_base = load_df(base_csv_path)
    df_other = load_df(other_csv_path)
    
    df_merged = pd.merge(df_base, df_other[key_columns + merge_columns], on=key_columns, how='left')
    

    if save: 
        if type(base_csv_path) == str and type(other_csv_path) == str:
            base_csv_name = os.path.basename(base_csv_path).replace(".csv", "")
            other_csv_name = os.path.basename(other_csv_path).replace(".csv", "")
            csv_name = base_csv_name + "_MERGED_" + other_csv_name + ".csv"
            
            csv_path = os.path.join(os.path.dirname(base_csv_path), csv_name)
            
            df_merged.to_csv(csv_path, index=False)
        else:
            raise ValueError("base_csv_path and other_csv_path must be strings if save is True")
    return df_merged
    
    
    
if __name__ == "__main__":
    
    # # ADD SID AND COLUMN
    # # csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data_with_supersynth_segmentation.csv"
    # # csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/paired_train_data.csv"
    # csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"
    # # csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/val_data.csv"
    # add_subject_global_id(csv_path)
    
    # # RENAME AND ORDER COLUMNS
    # csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"
    # csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data_with_supersynth_segmentation.csv"
    # csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/paired_train_data.csv"
    # csv_path = "/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data_MERGED_train_data_with_supersynth_segmentation.csv"
    # order_columns_and_rename(csv_path)
    
    
    # base_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"
    # other_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data_with_supersynth_segmentation.csv"
    # key_columns = ["org_img_path"]
    # merge_columns = ["seg_supersynth_path"]
    # merge_base_dataset_with_other_dataset_column(base_csv_path, other_csv_path, key_columns, merge_columns)
    
    
    
    
    
    # to merge segmentation with train 
    column_to_merge = "seg_supersynth_path"
    csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/res/train_data_with_supersynth_segmentation copy.csv"
    df_seg = pd.read_csv(csv_path)
    df_seg = add_subject_global_id(df_seg, save=False)
    df_seg = order_columns_and_rename(df_seg, save=False)
    # print(df_seg.columns)
    
    base_csv_path = f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data.csv"
    df_base = pd.read_csv(base_csv_path)
    # remove "seg_supersynth_path"
    if column_to_merge in df_base.columns:
        df_base = df_base.drop(columns=[column_to_merge])
    # print(df_base.columns)
    
    df_base = merge_base_dataset_with_other_dataset_column(df_base, df_seg, key_columns=["org_img_path"], merge_columns=[column_to_merge], save=False)
    df_base = order_columns_and_rename(df_base, save=False)
    
    df_base.to_csv(f"/home/agustin/phd/miccai/miccai_2026/mri_x_fields/data/csv/train_data_supersynth_seg_merged.csv", index=False)
    

    
    # df_base2 = pd.read_csv(base_csv_path)
    # # verify if df_base == df_base2
    # print(df_base.equals(df_base2)) # should be False because df_base has the new column "seg_supersynth_path" and df_base2 does not have it
    
    
    
# # --------------------- this is to solve the ID problem ---------------------





