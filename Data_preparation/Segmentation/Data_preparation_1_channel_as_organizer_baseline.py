'''
Prepare the breast DCE dataset for MedNeXt
'''

import pandas as pd
import os
import json
import argparse
import SimpleITK as sitk
import numpy as np
from natsort import natsorted
from utils import resample_image_to_isotropic, reconcile_metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', help='specifiy the tast name', default='Task012_breastDCE', type=str)
    args = parser.parse_args()

    root = 'root directory path'  # Replace with your actual root directory path
    data_partition_path = 'cv_fold_splits.xlsx'
    image_path = os.path.join(root, 'dataset', 'images')
    mask_path = os.path.join(root,  'dataset','segmentations', 'expert')
    patient_info_path = os.path.join(root, 'dataset', 'patient_info_files')
    convert_path = os.path.join(root, 'MedNext_dataset_python310', 'nnUNet_raw_data_base', 'nnUNet_raw_data', args.task_name)
    if not os.path.exists(convert_path):
        os.makedirs(convert_path, exist_ok=True)

    train_case_names = pd.read_excel(data_partition_path, sheet_name='train_fold_0')['patient_id'].tolist()
    val_case_names = pd.read_excel(data_partition_path, sheet_name='val_fold_0')['patient_id'].tolist()
    case_names = natsorted(train_case_names + val_case_names)

    image_save_path = os.path.join(convert_path, 'imagesTr')
    mask_save_path = os.path.join(convert_path, 'labelsTr')  

    if not os.path.exists(image_save_path):
            os.makedirs(image_save_path, exist_ok=True)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path, exist_ok=True)

    for case_name in case_names:
        print(f"Processing case: {case_name}")
        patient_json_file_path = os.path.join(patient_info_path, case_name.lower() + '.json')
        with open(patient_json_file_path, 'r') as f:
            patient_data = json.load(f)

        patient_id = patient_data['patient_id']
        if case_name.upper() != patient_id.upper():
            raise ValueError(f"Mismatch ID in {case_name}")

        # breast_coords = patient_data['primary_lesion']['breast_coordinates']
        dce_folder = os.path.join(image_path, case_name)
        dce_files = [f for f in natsorted(os.listdir(dce_folder)) if f.endswith('.nii.gz')]

        full_stack= []

        # Read DCE1 (index 1) first as reference
        ref_dce_file = f"{case_name.lower()}_{1:04d}.nii.gz"
        ref_image = sitk.ReadImage(os.path.join(dce_folder, ref_dce_file), sitk.sitkFloat32)

        # Add tumor mask 
        tumor_mask_path = os.path.join(mask_path, case_name.lower() + '.nii.gz')
        tumor_mask = sitk.ReadImage(tumor_mask_path)
        tumor_mask = reconcile_metadata(tumor_mask, ref_image, interpolator=sitk.sitkNearestNeighbor)

        # if sitk.GetArrayViewFromImage(tumor_mask_cropped).max()<=0:
        #     raise ValueError(f'The cropped mask {tumor_mask_path} has no positive values, which means no tumor in this breast!')

        for i, dce_file in enumerate(dce_files):
            expected_name = f"{case_name.lower()}_{i:04d}.nii.gz"
            if dce_file != expected_name:
                raise ValueError(f"File name mismatch in {case_name}: expected {expected_name}, got {dce_file}")
            
            full_path = os.path.join(dce_folder, dce_file)
            image = sitk.ReadImage(full_path, sitk.sitkFloat32)

            if i != 1:# Skip the first DCE as it is the reference
                image = reconcile_metadata(image, ref_image, interpolator=sitk.sitkLinear)

            full_stack.append(sitk.GetArrayFromImage(image))

        full_stack_np = np.stack(full_stack, axis=0)

        # Normalize with mean and std of all dce volumes
        mean_intensity = np.mean(full_stack_np)
        std_intensity = np.std(full_stack_np)

        if std_intensity == 0:
            print(f"Standard deviation is zero for case {case_name}. Cannot normalize intensities.")
            full_stack_normalized = full_stack_np - mean_intensity
        
        else:
            full_stack_normalized = (full_stack_np - mean_intensity) / std_intensity

        # resample to isotropic resolution
        tumor_mask_resampled = resample_image_to_isotropic(tumor_mask, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkNearestNeighbor)

        if len(full_stack)>99:
            raise ValueError("Too many DCE volumes!")

        if len(full_stack) > 1:
            for j in range(1,len(dce_files)):
                dce_image = full_stack_normalized[j]
                dce_image_sitk = sitk.GetImageFromArray(dce_image)
                dce_image_sitk.CopyInformation(ref_image)
                dce_image_sitk = resample_image_to_isotropic(dce_image_sitk, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
                # print(dce_image_sitk.GetPixelIDTypeAsString())
                DCE_n_convert_name = os.path.join(image_save_path,f"{case_name}_{str(j).zfill(2)}_0000.nii.gz")
                # save preprocessed post contrast images
                sitk.WriteImage(dce_image_sitk, DCE_n_convert_name)
                mask_convert_name = os.path.join(mask_save_path, f"{case_name}_{str(j).zfill(2)}.nii.gz")
                # save the tumor mask
                sitk.WriteImage(tumor_mask_resampled, mask_convert_name)

            # save the first post-contrast-precontrast image
            subtraction_10 = full_stack_normalized[1]-full_stack_normalized[0]
            subtraction_10_sitk = sitk.GetImageFromArray(subtraction_10)
            subtraction_10_sitk.CopyInformation(ref_image)
            subtraction_10_sitk = resample_image_to_isotropic(subtraction_10_sitk, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
            # print(subtraction_10_sitk.GetPixelIDTypeAsString())
            subtraction_convert_name = os.path.join(image_save_path,f"{case_name}_{str(len(dce_files)).zfill(2)}_0000.nii.gz")
            sitk.WriteImage(subtraction_10_sitk, subtraction_convert_name)

            mask_convert_name = os.path.join(mask_save_path, f"{case_name}_{str(len(dce_files)).zfill(2)}.nii.gz")
            # save the tumor mask
            sitk.WriteImage(tumor_mask_resampled, mask_convert_name)
            

