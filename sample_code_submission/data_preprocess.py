import os
import json
from natsort import natsorted
import pandas as pd
import SimpleITK as sitk
import numpy as np
import pickle
import shutil
from utils import (reconcile_metadata, crop_breast, resample_image_to_isotropic, 
                   resample_to_spacing_1x1x1, get_weighted_centroid, get_baseline_mean, extract_fixed_bounding_box)

'''
Segmentation model data preprocessing.
'''

def seg_task_006_preprocess(DCE_image_names, patient_id, patient_info, image_save_path, negative_clamping_value=-5, num_post_contrast_images=1):
    # Breast coordinates exist. Extract unilateral breast.
    DCE_0 = sitk.ReadImage(DCE_image_names[0])
    # get the 1st post-contrast image
    DCE_1 = sitk.ReadImage(DCE_image_names[1]) 
    # check if the pre-contrast image needs resampling
    DCE_0 = reconcile_metadata(DCE_0, DCE_1, interpolator=sitk.sitkLinear)

    if not patient_info or "primary_lesion" not in patient_info:
        DCE_0_cropped = DCE_0
        DCE_1_cropped = DCE_1
    else:
        # Breast coordinates exist. Extract unilateral breast.
        breast_coordinates = patient_info['primary_lesion']['breast_coordinates']
         
        DCE_1_cropped = crop_breast(DCE_1, breast_coordinates)
        DCE_0_cropped = crop_breast(DCE_0, breast_coordinates)

    array_1 = sitk.GetArrayViewFromImage(DCE_1_cropped)
    if array_1.min() < negative_clamping_value:
        DCE_1_cropped = sitk.Clamp(DCE_1_cropped, lowerBound=negative_clamping_value)  
        
    array_0 = sitk.GetArrayViewFromImage(DCE_0_cropped)
    if array_0.min() < negative_clamping_value:
        DCE_0_cropped = sitk.Clamp(DCE_0_cropped, lowerBound=negative_clamping_value) 
        
    subtracted_10 = sitk.Subtract(DCE_1_cropped, DCE_0_cropped)
    # print('We do not clamp the subtraction images.')

    if num_post_contrast_images>1:
        DCE_1_convert_name = os.path.join(image_save_path,f"{patient_id}_{str(1).zfill(2)}_0000.nii.gz")
        
    else:
        DCE_1_convert_name = os.path.join(image_save_path,f"{patient_id}_0000.nii.gz")
    
    sitk.WriteImage(subtracted_10, DCE_1_convert_name)

def seg_task_008_preprocess(DCE_image_names, patient_id, patient_info, image_save_path, negative_clamping_value=-5, num_post_contrast_images=1):
    # Breast coordinates exist. Extract unilateral breast.
    DCE_0 = sitk.ReadImage(DCE_image_names[0])
    # get the 1st post-contrast image
    DCE_1 = sitk.ReadImage(DCE_image_names[1]) 
    # check if the pre-contrast image needs resampling
    DCE_0 = reconcile_metadata(DCE_0, DCE_1, interpolator=sitk.sitkLinear)

    if not patient_info or "primary_lesion" not in patient_info:
        DCE_0_cropped = DCE_0
        DCE_1_cropped = DCE_1
    else:
        # Breast coordinates exist. Extract unilateral breast.
        breast_coordinates = patient_info['primary_lesion']['breast_coordinates']
        DCE_0_cropped = crop_breast(DCE_0, breast_coordinates)
        DCE_1_cropped = crop_breast(DCE_1, breast_coordinates)
        

    array_1 = sitk.GetArrayViewFromImage(DCE_1_cropped)
    if array_1.min() < negative_clamping_value:
        DCE_1_cropped = sitk.Clamp(DCE_1_cropped, lowerBound=negative_clamping_value)  
        
    array_0 = sitk.GetArrayViewFromImage(DCE_0_cropped)
    if array_0.min() < negative_clamping_value:
        DCE_0_cropped = sitk.Clamp(DCE_0_cropped, lowerBound=negative_clamping_value) 
        
    subtracted_10 = sitk.Subtract(DCE_1_cropped, DCE_0_cropped)
    # print('We do not clamp the subtraction images.')

    if num_post_contrast_images>1:
        
        DCE_1_convert_name = os.path.join(image_save_path,f"{patient_id}_{str(1).zfill(2)}_0000.nii.gz")
        subtracted_10_name = os.path.join(image_save_path, f"{patient_id}_{str(1).zfill(2)}_0001.nii.gz")

    else:
        DCE_1_convert_name = os.path.join(image_save_path,f"{patient_id}_0000.nii.gz")
        subtracted_10_name = os.path.join(image_save_path, f"{patient_id}_0001.nii.gz") 
    
    sitk.WriteImage(DCE_1_cropped, DCE_1_convert_name)
    sitk.WriteImage(subtracted_10, subtracted_10_name)

def seg_task_010_preprocess(DCE_image_names, patient_id, patient_info, image_save_path, negative_clamping_value=-5, num_post_contrast_images=1):
    # Breast coordinates exist. Extract unilateral breast.
    DCE_0 = sitk.ReadImage(DCE_image_names[0])
    # get the 1st post-contrast image
    DCE_1 = sitk.ReadImage(DCE_image_names[1]) 
    # check if the pre-contrast image needs resampling
    DCE_0 = reconcile_metadata(DCE_0, DCE_1, interpolator=sitk.sitkLinear)

    if not patient_info or "primary_lesion" not in patient_info:
        DCE_0_cropped = DCE_0
        DCE_1_cropped = DCE_1
    else:
        # Breast coordinates exist. Extract unilateral breast.
        breast_coordinates = patient_info['primary_lesion']['breast_coordinates']
        DCE_0_cropped = crop_breast(DCE_0, breast_coordinates)
        DCE_1_cropped = crop_breast(DCE_1, breast_coordinates)
        
    array_1 = sitk.GetArrayViewFromImage(DCE_1_cropped)
    if array_1.min() < negative_clamping_value:
        DCE_1_cropped = sitk.Clamp(DCE_1_cropped, lowerBound=negative_clamping_value)  
        
    array_0 = sitk.GetArrayViewFromImage(DCE_0_cropped)
    if array_0.min() < negative_clamping_value:
        DCE_0_cropped = sitk.Clamp(DCE_0_cropped, lowerBound=negative_clamping_value) 
        
    subtracted_10 = sitk.Subtract(DCE_1_cropped, DCE_0_cropped)
    # print('We do not clamp the subtraction images.')

    if num_post_contrast_images>1:
        
        DCE_0_convert_name = os.path.join(image_save_path,f"{patient_id}_{str(1).zfill(2)}_0000.nii.gz")
        subtracted_10_name = os.path.join(image_save_path, f"{patient_id}_{str(1).zfill(2)}_0001.nii.gz")

    else:
        DCE_0_convert_name = os.path.join(image_save_path,f"{patient_id}_0000.nii.gz")
        subtracted_10_name = os.path.join(image_save_path, f"{patient_id}_0001.nii.gz") 
    
    sitk.WriteImage(DCE_0_cropped, DCE_0_convert_name)
    sitk.WriteImage(subtracted_10, subtracted_10_name)

def seg_task_012_preprocess(DCE_image_names, patient_id, image_save_path):

    # get the 1st post-contrast image
    DCE_1 = sitk.ReadImage(DCE_image_names[1], sitk.sitkFloat32) 
    DCE_1_array = sitk.GetArrayFromImage(DCE_1)

    full_stack= []
    for i, image_name in enumerate(DCE_image_names):
        image = sitk.ReadImage(image_name, sitk.sitkFloat32)

        if i != 1:# Skip the first DCE as it is the reference
            image = reconcile_metadata(image, DCE_1, interpolator=sitk.sitkLinear)

        full_stack.append(sitk.GetArrayFromImage(image))

    full_stack_np = np.stack(full_stack, axis=0)
    # Normalize with mean and std of all dce volumes
    mean_intensity = np.mean(full_stack_np)
    std_intensity = np.std(full_stack_np)

    case_metrics = {
        'Num_DCEs': len(DCE_image_names),
        'Mean_Full': float(mean_intensity),
        'Std_Full': float(std_intensity),
    }

    if std_intensity == 0:
        print(f"Standard deviation is zero for case {image_name}. Cannot normalize intensities.")
        DCE_1_normalized = DCE_1_array - mean_intensity
    
    else:
        DCE_1_normalized = (DCE_1_array - mean_intensity) / std_intensity

    dce_image_sitk = sitk.GetImageFromArray(DCE_1_normalized)
    dce_image_sitk.CopyInformation(DCE_1)

    dce_image_sitk = resample_image_to_isotropic(dce_image_sitk, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)

    DCE_convert_name = os.path.join(image_save_path,f"{patient_id}_0000.nii.gz")
    
    sitk.WriteImage(dce_image_sitk, DCE_convert_name)

    return case_metrics



'''
pcr_classification model data preprocessing.
'''
# initial version of deep learning model data preprocessing
def process_data_for_pcr_classification(image_folder_path, seg_folder_path, output_folder_path, case_name, box_size):
    print(case_name)
    # Read the segmentation mask
    seg_path = os.path.join(seg_folder_path, case_name)
    seg = sitk.ReadImage(seg_path)
    seg_spacing = seg.GetSpacing()
    seg_array = sitk.GetArrayFromImage(seg)
    # Read the segmentation probability map
    seg_probmap_path = os.path.join(seg_folder_path, case_name.replace('.nii.gz','_prob.nii.gz'))
    seg_probmap = sitk.ReadImage(seg_probmap_path)
    # seg_probmap_array = sitk.GetArrayFromImage(seg_probmap)
    
    # Read the images
    DCE0_path = os.path.join(image_folder_path, case_name.replace('.nii.gz','_0000.nii.gz'))
    DCE0 = sitk.ReadImage(DCE0_path)
    DCE0 = sitk.Maximum(DCE0, 0.0)# remove negative values
    # DCE0_array = sitk.GetArrayFromImage(DCE0)

    DCE1_path = os.path.join(image_folder_path, case_name.replace('.nii.gz','_0001.nii.gz'))
    DCE1 = sitk.ReadImage(DCE1_path)
    # DCE1_array = sitk.GetArrayFromImage(DCE1)

    DCE2_path = os.path.join(image_folder_path, case_name.replace('.nii.gz','_0002.nii.gz'))
    DCE2 = sitk.ReadImage(DCE2_path)
    # DCE2_array = sitk.GetArrayFromImage(DCE2)

    # Resample everybody to 1*1*1mm
    DCE0_resampled = resample_to_spacing_1x1x1(DCE0, spacing=(1.0, 1.0, 1.0),interpolation_method=sitk.sitkLinear)
    DCE0_resampled_array = sitk.GetArrayFromImage(DCE0_resampled)

    DCE1_resampled = resample_to_spacing_1x1x1(DCE1, spacing=(1.0, 1.0, 1.0),interpolation_method=sitk.sitkLinear)
    DCE1_resampled_array = sitk.GetArrayFromImage(DCE1_resampled)
    
    DCE2_resampled = resample_to_spacing_1x1x1(DCE2, spacing=(1.0, 1.0, 1.0),interpolation_method=sitk.sitkLinear)
    DCE2_resampled_array = sitk.GetArrayFromImage(DCE2_resampled)

    seg_resampled = resample_to_spacing_1x1x1(seg, spacing=(1.0, 1.0, 1.0), interpolation_method=sitk.sitkNearestNeighbor)
    seg_resampled_array = sitk.GetArrayFromImage(seg_resampled)

    seg_probmap_resampled = resample_to_spacing_1x1x1(seg_probmap, spacing=(1.0, 1.0, 1.0), interpolation_method=sitk.sitkLinear)
    seg_probmap_resampled_array = sitk.GetArrayFromImage(seg_probmap_resampled)
    
    # (2) get the weighted centroid of the predicted lesion
    centroid = get_weighted_centroid(seg_probmap_resampled_array, DCE0_resampled_array, 0.1)

    baseline_mean = get_baseline_mean(DCE0_resampled_array, seg_resampled_array)
    
    DCE0_resampled_array_norm = DCE0_resampled_array / baseline_mean
    DCE1_resampled_array_norm = DCE1_resampled_array / baseline_mean
    DCE2_resampled_array_norm = DCE2_resampled_array / baseline_mean

    DCE0_ROI_array = extract_fixed_bounding_box(DCE0_resampled_array_norm, centroid, box_size, pad_value=0)
    DCE1_ROI_array = extract_fixed_bounding_box(DCE1_resampled_array_norm, centroid, box_size, pad_value=0)
    DCE2_ROI_array = extract_fixed_bounding_box(DCE2_resampled_array_norm, centroid, box_size, pad_value=0)
    seg_probmap_ROI_array = extract_fixed_bounding_box(seg_probmap_resampled_array, centroid, box_size, pad_value=0)
    # stacked_inputs = np.stack([DCE1_ROI_array.astype(np.float32), DCE2_ROI_array.astype(np.float32), seg_probmap_ROI_array.astype(np.float32)], axis=0)# for classification_inputs_v1
    stacked_inputs = np.stack([DCE0_ROI_array.astype(np.float32), DCE1_ROI_array.astype(np.float32), DCE2_ROI_array.astype(np.float32), seg_probmap_ROI_array.astype(np.float32)], axis=0)# for classification_inputs_v2
    stacked_inputs = np.nan_to_num(stacked_inputs, nan=0.0, posinf=0.0, neginf=0.0)

    print("stacked input shape:",stacked_inputs.shape)  # Output: (3, Z, Y, X)

    stacked_inputs = np.transpose(stacked_inputs, (1, 2, 3, 0))  # (Z, Y, X, C)
    print("stacked input shape after transpose:",stacked_inputs.shape)  # Output: (Z, Y, X, C)

    # Create vector image
    sitk_image = sitk.GetImageFromArray(stacked_inputs, isVector=True)

    sitk_image.SetSpacing((1.0, 1.0, 1.0))             # Example spacing (z, y, x)
    sitk_image.SetOrigin((0.0, 0.0, 0.0))              # Default origin
    sitk_image.SetDirection((1.0, 0.0, 0.0,            # Identity direction matrix
                            0.0, 1.0, 0.0,
                            0.0, 0.0, 1.0))
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    output_path = os.path.join(output_folder_path, case_name)
    sitk.WriteImage(sitk_image, output_path)

# Extract radiomics features
