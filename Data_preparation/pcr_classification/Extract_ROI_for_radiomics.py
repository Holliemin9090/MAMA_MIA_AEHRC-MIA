"""
Extract breast region on all dce sequences and reample to isotropic resolution for radiomics analysis.
"""
import pandas as pd
import os
import sys
import json
import SimpleITK as sitk
import numpy as np
import shutil
from natsort import natsorted
from utils import (reconcile_metadata, resample_image_to_isotropic, crop_breast)

if __name__ == '__main__':

    root = 'root path'
    data_partition_path = os.path.join(root,'Data_preparation', 'Segmentation', 'cv_fold_splits.xlsx')
    image_path = os.path.join(root, 'dataset', 'images')
    mask_path = os.path.join(root,  'dataset','segmentations', 'expert')
    patient_info_path = os.path.join(root, 'dataset', 'patient_info_files')
    data_save_path = os.path.join(root, 'MedNext_dataset_python310', 'Radiomics_classification_inputs_v1')
    
    train_case_names = pd.read_excel(data_partition_path, sheet_name='train_fold_0')['patient_id'].tolist()
    val_case_names = pd.read_excel(data_partition_path, sheet_name='val_fold_0')['patient_id'].tolist()
    case_names = natsorted(train_case_names + val_case_names)

    if os.path.exists(data_save_path):
        shutil.rmtree(data_save_path)
    os.makedirs(data_save_path, exist_ok=True)

    image_save_folder = os.path.join(data_save_path, "imagesTr")
    mask_save_folder = os.path.join(data_save_path, "labelsTr")
    os.makedirs(image_save_folder, exist_ok=True)
    os.makedirs(mask_save_folder, exist_ok=True)
    
    for case_name in case_names:
        print(f"Processing case: {case_name}")
        patient_json_file_path = os.path.join(patient_info_path, case_name.lower() + '.json')
        with open(patient_json_file_path, 'r') as f:
            patient_data = json.load(f)

        patient_id = patient_data['patient_id']
        if case_name.upper() != patient_id.upper():
            raise ValueError(f"Mismatch ID in {case_name}")

        breast_coords = patient_data['primary_lesion']['breast_coordinates']
        dce_folder = os.path.join(image_path, case_name)
        DCE_image_names = [f for f in natsorted(os.listdir(dce_folder)) if f.endswith('.nii.gz')]

        DCE_1_path = os.path.join(dce_folder, DCE_image_names[1])
        DCE_1 = sitk.ReadImage(DCE_1_path)  

        for i, dce_file in enumerate(DCE_image_names):
            if dce_file != f"{case_name.lower()}_00{i:02d}.nii.gz":
                raise ValueError(f"File name mismatch: {dce_file} does not match expected name {case_name.lower()}_00{i:02d}.nii.gz")
            
            if i != 1:       
                DCE = sitk.ReadImage(os.path.join(dce_folder, dce_file))
                print(f"Processing DCE file: {dce_file}")
                DCE = reconcile_metadata(DCE, DCE_1, interpolator=sitk.sitkLinear)
                DCE_cropped = crop_breast(DCE, breast_coords)
                DCE_resampled_cropped = resample_image_to_isotropic(DCE_cropped, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
                
                DCE_save_path = os.path.join(image_save_folder, f"{case_name}_00{i:02d}.nii.gz")
                sitk.WriteImage(DCE_resampled_cropped, DCE_save_path)

        # save the first DCE image as well
        print(f"Processing DCE file: {DCE_image_names[1]}")
        DCE_1_cropped = crop_breast(DCE_1, breast_coords)
        DCE_1_resampled_cropped = resample_image_to_isotropic(DCE_1_cropped, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
        DCE_1_save_path = os.path.join(image_save_folder, f"{case_name}_0001.nii.gz")
        sitk.WriteImage(DCE_1_resampled_cropped, DCE_1_save_path)

        # Process the mask
        print(f"Processing mask for case: {case_name}")
        mask_label_path = os.path.join(mask_path, case_name.lower() + '.nii.gz')
        mask = sitk.ReadImage(mask_label_path)
        mask = reconcile_metadata(mask, DCE_1, interpolator=sitk.sitkNearestNeighbor)
        mask_cropped = crop_breast(mask, breast_coords)
        mask_resampled_cropped = resample_image_to_isotropic(mask_cropped, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkNearestNeighbor)
        mask_save_path = os.path.join(mask_save_folder, f"{case_name}.nii.gz")
        sitk.WriteImage(mask_resampled_cropped, mask_save_path)
        
        print(f"Saved images for case {case_name} to {data_save_path}")

