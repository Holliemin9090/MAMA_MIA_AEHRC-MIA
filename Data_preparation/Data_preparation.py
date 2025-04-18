# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:24:11 2023

@author: min105

Generate data for nnunet on DCE sequences
"""
import pandas as pd
from natsort import natsorted
import os
import argparse
import json
import SimpleITK as sitk
import numpy as np

def crop_breast(image, coords):
    # Compute starting index and size for cropping
    # Alert: the coordinates seems to be stored in [z, y, x] order, although it says [x, y, z]
    start = [coords['z_min'], coords['y_min'], coords['x_min']]  # [x, y, z]
    size = [
        coords['z_max'] - coords['z_min'],
        coords['y_max'] - coords['y_min'],
        coords['x_max'] - coords['x_min'],         
    ]

    # Crop the image
    cropped_image = sitk.RegionOfInterest(image, size=size, index=start)

    return cropped_image

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_or_test', help='specifiy if train/val or test data', default='train', type=str)
   
    args = parser.parse_args()
    
    root = '/datasets/work/hb-breast-nac-challenge/work/'# '/datasets/work/hb-breast-nac-challenge/work/'  '/Volumes/{hb-breast-nac-challenge}/work/'
    
    convert_path = os.path.join(root, 'MedNext_dataset', 'nnUNet_raw_data_base', 'nnUNet_raw_data', 'Task001_breastDCE')

    if not os.path.exists(convert_path):
        os.makedirs(convert_path, exist_ok=True)
     
    if args.train_or_test=='train':
        # '/Volumes/{hb-breast-nac-challenge}/work/min105/scripts/Data_preparation/cv_fold_splits.xlsx'
    
        data_partition_path = os.path.join(root, 'min105', 'scripts', 'Data_preparation', 'cv_fold_splits.xlsx')
        image_path = os.path.join(root, 'dataset', 'images')
        mask_path = os.path.join(root,  'dataset','segmentations', 'expert')
        patient_info_path = os.path.join(root, 'dataset', 'patient_info_files')
         
        train_case_names = pd.read_excel(data_partition_path, sheet_name='train_fold_0')['patient_id'].tolist()
        val_case_names = pd.read_excel(data_partition_path, sheet_name='val_fold_0')['patient_id'].tolist()
        
        case_names = natsorted(train_case_names + val_case_names)
        
        image_save_path = os.path.join(convert_path, 'imagesTr')
        mask_save_path = os.path.join(convert_path, 'labelsTr')  
    
    # elif args.train_or_test=='test':
        
    #     case_path = root + 'Test_reorient/'
        
    #     case_names = natsorted(os.listdir(case_path))
        
    #     image_save_path = convert_path + 'imagesTs/'
    #     mask_save_path = convert_path + 'labelsTs/'  
        
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path, exist_ok=True)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path, exist_ok=True)

    for i, case_name in enumerate(case_names):
        
        patient_json_file_path = os.path.join(patient_info_path, case_name.lower()+'.json')

        # Load JSON data (assuming it's stored in a file called 'patient_data.json')
        with open(patient_json_file_path, 'r') as f:
            patient_data = json.load(f)

        # Extract patient ID
        patient_id = patient_data['patient_id']

        # Extract breast coordinates
        breast_coordinates = patient_data['primary_lesion']['breast_coordinates']

        if case_name.upper()!= patient_id.upper():
            raise ValueError('Case name and patient ID do not match!')
        
        DCE_sequence_path = os.path.join(image_path, case_name)
        DCE_image_names = natsorted(os.listdir(DCE_sequence_path))
        DCE_image_names = [name for name in DCE_image_names if name.endswith('.nii.gz')]

        if DCE_image_names[0].find('_0000.nii.gz') == -1:
            raise ValueError('The first image in the DCE sequence does not end with _0000.nii.gz!')
        # get the pre-contrast image
        DCE_0_path = os.path.join(DCE_sequence_path, DCE_image_names[0]) 
        DCE_0 = sitk.ReadImage(DCE_0_path)
        # print(DCE_0.GetSize())
        DCE_0_cropped = crop_breast(DCE_0, breast_coordinates)
        # image_save_path = convert_path + 'imagesTs/'
        DCE_0_convert_name = os.path.join(image_save_path, 'breastDCE_' + str(i).zfill(4) + '_0000.nii.gz')
        sitk.WriteImage(DCE_0_cropped, DCE_0_convert_name)

        # get the 1st post-contrast image
        DCE_1_path = os.path.join(DCE_sequence_path, DCE_image_names[1])
        DCE_1 = sitk.ReadImage(DCE_1_path)      
        DCE_1_cropped = crop_breast(DCE_1, breast_coordinates)
        
        subtracted_10 = sitk.Subtract(DCE_1_cropped, DCE_0_cropped)
        subtracted_10_clamped = sitk.Maximum(subtracted_10, 0.0)
        DCE_1_convert_name = os.path.join(image_save_path,'breastDCE_' + str(i).zfill(4) + '_0001.nii.gz')
        sitk.WriteImage(subtracted_10_clamped, DCE_1_convert_name)

        # get the late post-contrast image
        DCE_2_path = os.path.join(DCE_sequence_path, DCE_image_names[-1])
        DCE_2 = sitk.ReadImage(DCE_2_path)
        DCE_2_cropped = crop_breast(DCE_2, breast_coordinates)
        subtracted_20 = sitk.Subtract(DCE_2_cropped, DCE_0_cropped)
        subtracted_20_clamped = sitk.Maximum(subtracted_20, 0.0)
        DCE_2_convert_name = os.path.join(image_save_path,'breastDCE_' + str(i).zfill(4) + '_0002.nii.gz')
        sitk.WriteImage(subtracted_20_clamped, DCE_2_convert_name)
        # Check if the images have negative values
        # min_value = sitk.GetArrayViewFromImage(DCE_1_cropped).min()
        if min(sitk.GetArrayViewFromImage(DCE_0_cropped).min(), 
               sitk.GetArrayViewFromImage(DCE_1_cropped).min(), 
               sitk.GetArrayViewFromImage(DCE_2_cropped).min()) < 0:
            print('Warning: The DCE images contains negative values.')

        # get the mask
        mask_label_path = os.path.join(mask_path, case_name.lower() + '.nii.gz')
        mask = sitk.ReadImage(mask_label_path)
        mask_cropped = crop_breast(mask, breast_coordinates)
        mask_convert_name = os.path.join(mask_save_path, 'breastDCE_' + str(i).zfill(4) + '.nii.gz')
        sitk.WriteImage(mask_cropped, mask_convert_name) 
        
        
    print('Data preparation completed!')
        
        
        
 
        
        
        