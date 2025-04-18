'''
Data analysis and sanity check
Beawre that some cases may not have pcr or acquisition times information
'''
import pandas as pd
import numpy as np
import os
from natsort import natsorted
import SimpleITK as sitk
import ast

data_path = '/datasets/work/hb-breast-nac-challenge/work/dataset'
image_path = os.path.join(data_path, 'images')
label_path = os.path.join(data_path, 'segmentations', 'expert')

case_names = natsorted([
    d for d in os.listdir(image_path)
    if os.path.isdir(os.path.join(image_path, d))
])
print('Number of cases:', len(case_names))

label_names = natsorted([
    d for d in os.listdir(label_path)
    if d.find('.nii.gz') != -1
])

print('Number of labels:', len(label_names))

for i, case in enumerate(case_names):
    case_folder_path = os.path.join(image_path, case)
    DCE_image_names = natsorted([
        d for d in os.listdir(case_folder_path)
        if d.endswith('.nii.gz')
    ])
    print(f'\nCase: {case}')
    print('Image names:', DCE_image_names)

    reference_spacing = None
    reference_direction = None
    reference_size = None
    consistent = True  # Assume consistent unless proven otherwise

    for j, image_name in enumerate(DCE_image_names):
        # Read the image using SimpleITK
        DCE_image_path = os.path.join(case_folder_path, image_name)
        image = sitk.ReadImage(DCE_image_path)

        spacing = image.GetSpacing()
        direction = image.GetDirection()
        size = image.GetSize()

        print(f"\nImage {image_name}")
        print("  Spacing:", spacing)
        print("  Orientation:", direction)
        print("  Array size:", size)# x y z

        if j == 0:
            # Use the first image as reference
            reference_spacing = spacing
            reference_direction = direction
            reference_size = size
        else:
            if spacing != reference_spacing:
                print("  ❌ Inconsistent spacing detected")
                consistent = False
            if direction != reference_direction:
                print("  ❌ Inconsistent orientation detected")
                consistent = False
            if size != reference_size:
                print("  ❌ Inconsistent array size detected")
                consistent = False

    if consistent:
        print("✅ All images in this case have consistent spacing, orientation, and size.")
    else:
        print("⚠️ Inconsistencies found in one or more images in this case.")
    # Check the labels

    tumor_label_name = case.lower() + '.nii.gz'
    tumor_label_path = os.path.join(label_path, tumor_label_name)
    if os.path.exists(label_path):
        label_image = sitk.ReadImage(tumor_label_path)
        label_spacing = label_image.GetSpacing()
        label_direction = label_image.GetDirection()
        label_size = label_image.GetSize()

        print(f"\nLabel {tumor_label_name}")
        print("  Spacing:", label_spacing)
        print("  Orientation:", label_direction)
        print("  Array size:", label_size)

        if (label_spacing != reference_spacing or
                label_direction != reference_direction or
                label_size != reference_size):
            print("⚠️ Label image has inconsistent properties compared to DCE images.")

        if label_size != reference_size:
            raise ValueError("⚠️ Label image size does not match DCE image size.")
    else:
        raise ValueError(f"⚠️ Label file {tumor_label_name} not found.")
    
    # Check image size infomation

    if reference_size[0]!=reference_size[1]:
        raise ValueError(f"⚠️ Image axial plane size is not square. Size: {reference_size}")
    
    if reference_size[0] < reference_size[2]:
        # need to change to print, cause the error will be raised, probably because the image is a unilateral breast
        print(f"⚠️ Image axial plane size is smaller than slice size. Size: {reference_size}")
    
    # read the clinical information and check if the image information matches
    clinical_info_path = '/datasets/work/hb-breast-nac-challenge/work/dataset/clinical_and_imaging_info.xlsx'

    clinical_info = pd.read_excel(clinical_info_path, sheet_name='dataset_info')

    patient_data = clinical_info[clinical_info["patient_id"] == case]
    # Print the entire row
    print(patient_data)

    # Or get a specific value
    if str(patient_data["acquisition_times"].values[0]) != 'nan':
        # If acuisition time was recorded
        acuisition_times = ast.literal_eval(patient_data["acquisition_times"].values[0])
        print("acquisition_times:", patient_data["acquisition_times"].values[0])

        if len(acuisition_times) != len(DCE_image_names):
            print(f"⚠️ Number of DCE images does not match the number of acquisition times. {len(acuisition_times)} vs {len(DCE_image_names)}")
    
    
    print("image_rows:", patient_data["image_rows"].values[0])
    print("image_columns:", patient_data["image_columns"].values[0])
    print("num_slices:", patient_data["num_slices"].values[0])
    print("pixel_spacing:", patient_data["pixel_spacing"].values[0])
    print("number of DCEs", len(DCE_image_names))
    #  This part has to be changed to print, cause the error will be raised, probably because missrecorded image size. May need to ask the organizer
    if patient_data["image_rows"].values[0]!= reference_size[1]:
        print(f"⚠️ Image rows does not match the image size. {patient_data['image_rows'].values[0]} vs {reference_size[1]}")
    if patient_data["image_columns"].values[0]!= reference_size[0]:
        print(f"⚠️ Image columns does not match the image size. {patient_data['image_columns'].values[0]} vs {reference_size[0]}")
    if patient_data["num_slices"].values[0]!= reference_size[2]:    
        print(f"⚠️ Number of slices does not match the image size. {patient_data['num_slices'].values[0]} vs {reference_size[2]}")


    

   