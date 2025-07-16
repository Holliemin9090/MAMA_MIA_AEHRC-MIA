'''
This is the final version of script to extract a subset of radiomics features in comparison to Extract_radiomics.py.
'''

import sys
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import argparse
import radiomics
print(radiomics.__version__)
from radiomics import featureextractor
from tqdm import tqdm
from natsort import natsorted
import re
# from scoring.utils import compute_ordinal_kinetics
from utils import ErodeByFraction #Dilation, 
import logging
logging.getLogger('radiomics.glcm').setLevel(logging.ERROR)  # Or logging.CRITICAL to suppress everything

def get_dce_keys_from_folder(case_id, folder_path):
    """
    Given a case ID and a folder path, return a sorted list of all DCE keys like '_0000', '_0001', etc.
    """
    pattern = re.compile(f"{case_id}_(\\d{{4}})\\.nii\\.gz")  # matches 'CASEID_000x.nii.gz'
    keys = []

    for fname in os.listdir(folder_path):
        match = pattern.match(fname)
        if match:
            dce_suffix = match.group(1)
            keys.append(f"_{dce_suffix}")

    return natsorted(keys)  # e.g., ['_0000', '_0001', '_0002']


def normalize_by_tumor_mean(image, t0_mean, epsilon=1e-5):
    image_array = sitk.GetArrayFromImage(image)
    image_array_norm = image_array / (t0_mean + epsilon)
    image_norm = sitk.GetImageFromArray(image_array_norm)
    image_norm.CopyInformation(image)
    return image_norm

def compute_ordinal_kinetics(I, sentinel=-1.0):
    if not isinstance(I, (list, np.ndarray)) or len(I) < 2:
        return {
            "MaxEnhancement": sentinel,
            "WashInStep": sentinel,
            "WashOutStep": sentinel,
            "EarlyLateRatio": sentinel,
            "SER": sentinel,
        }

    pre = I[0]
    posts = I[1:]

    if pre == 0 or len(posts) == 0:
        return {
            "MaxEnhancement": sentinel,
            "WashInStep": sentinel,
            "WashOutStep": sentinel,
            "EarlyLateRatio": sentinel,
            "SER": sentinel,
        }

    max_enh = (max(posts) - pre) / pre if pre != 0 else sentinel

    wash_in = (I[1] - pre) / pre if len(I) > 1 and pre != 0 else sentinel
    wash_out = (I[1] - I[-1]) / pre if len(I) > 2 and pre != 0 else sentinel
    early_late_ratio = I[-1] / I[1] if len(I) > 2 and I[1] != 0 else sentinel
    
    ser_denom = I[-1] - pre
    ser = (I[1] - pre) / ser_denom if len(I) > 2 and ser_denom != 0 else sentinel

    return {
        "MaxEnhancement": float(max_enh),
        "WashInStep": float(wash_in),
        "WashOutStep": float(wash_out),
        "EarlyLateRatio": float(early_late_ratio),
        "SER": float(ser),
    }

def contrast_kenitics(I, mode="tumour_to_background", sentinel=-1.0):
    """
    Calculate the contrast kinetics for a list of DCE intensities.
    I: List of DCE intensities.
    sentinel: Value to return if the calculation cannot be performed.
    """
    if not isinstance(I, (list, np.ndarray)) or len(I) < 3:
        return {
            "WashIn": sentinel,
            "WashOut": sentinel,
            "SER": sentinel,
            "EarlyLateRatio": sentinel,
        }

    pre = I[0]
    post1 = I[1]
    post_last = I[-1]

    wash_in = (post1 - pre) 
    wash_out = (post1 - post_last) 
    
    return {
        f"{mode}_Contrast_1stPost": float(post1),
        f"{mode}_Contrast_LastPost": float(post_last),
        f"{mode}_Contrast_WashIn": float(wash_in),
        f"{mode}_Contrast_WashOut": float(wash_out),
    }

def get_tumour_intensities(dce_images, tumor_mask):
    """
    Calculate the mean intensity of the tumor in each DCE image.
    """
    tumour_intensities = []
    
    for key, image in dce_images.items():
        image_array = sitk.GetArrayFromImage(image)
        tumor_values = image_array[tumor_mask]
        if tumor_values.size > 0:
            tumour_intensities.append(float(np.mean(tumor_values)))
        else:
            tumour_intensities.append(np.nan ) # No tumor values found

    return tumour_intensities

def get_tumour_contrast_to_background(dce_images, tumor_mask, background_mask):
    """
    Calculate the contrast to background for each DCE image.
    dce_images: the normalized DCE images as a dictionary with keys like '_0000', '_0001', etc.
    tumor_mask: a boolean mask for the tumor region.
    background_mask: a boolean mask for the background region.
    """
    tumour_contrast = []
    
    for key, image in dce_images.items():
        image_array = sitk.GetArrayFromImage(image)
        tumor_values = image_array[tumor_mask]
        background_values = image_array[background_mask]
        
        if tumor_values.size > 0 and background_values.size > 0:
            contrast = np.mean(tumor_values) - np.mean(background_values)
            tumour_contrast.append(float(contrast))
        else:
            tumour_contrast.append(np.nan)  # No values found

    return tumour_contrast

def get_peripheral_contrast_to_core(dce_images, core_mask, peripheral_mask):
    """
    Calculate the contrast to peripheral for each DCE image.
    dce_images: the normalized DCE images as a dictionary with keys like '_0000', '_0001', etc.
    core_mask: a boolean mask for the core region.
    peripheral_mask: a boolean mask for the peripheral region.
    """
    core_contrast = []
    
    for key, image in dce_images.items():
        image_array = sitk.GetArrayFromImage(image)
        core_values = image_array[core_mask]
        peripheral_values = image_array[peripheral_mask]
        
        if core_values.size > 0 and peripheral_values.size > 0:
            contrast = np.mean(peripheral_values) - np.mean(core_values)
            core_contrast.append(float(contrast))
        else:
            core_contrast.append(np.nan)  # No values found

    return core_contrast

def is_empty_mask(mask_array):
    return np.sum(mask_array > 0) == 0

def convert_mask_array_to_sitk(mask_array, reference_image):
    mask_sitk = sitk.GetImageFromArray(mask_array.astype(np.uint8))
    mask_sitk.CopyInformation(reference_image)
    return mask_sitk



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract subset of radiomics features")
    parser.add_argument('--bin_width', type=float, default=0.5, help='Bin width for radiomics extraction')
    args = parser.parse_args()

    bin_width = args.bin_width

    root = 'root path'
    data_partition_path = os.path.join(root, 'Data_preparation', 'Segmentation', 'cv_fold_splits.xlsx')
    image_path = os.path.join(root, 'dataset', 'images')
    mask_path = os.path.join(root,  'dataset','segmentations', 'expert')
    patient_info_path = os.path.join(root, 'dataset', 'patient_info_files')
    data_path = os.path.join(root, 'MedNext_dataset_python310', 'Radiomics_classification_inputs_v1')
    
    train_case_names = pd.read_excel(data_partition_path, sheet_name='train_fold_0')['patient_id'].tolist()
    val_case_names = pd.read_excel(data_partition_path, sheet_name='val_fold_0')['patient_id'].tolist()
    case_names = natsorted(train_case_names + val_case_names)
    # Define your dataset folder
    image_folder = os.path.join(data_path, "imagesTr")
    mask_folder = os.path.join(data_path, "labelsTr")

    output_path = 'output path'

    params_dce1 = {
    'imageType': {
        'Original': {},
        'LoG': {'sigma': [1.0, 2.0, 3.0]},
        'Wavelet': {}
    },
    'featureClass': {
        'firstorder': [],
        'glcm': [],
        'glszm': [],
        'shape': []  # 3D shape features only
    },
    'setting': {
        'binWidth': bin_width,
        'force2D': False,  # Ensure 3D features are extracted
    }
    }

    params_other_dce = {
    'imageType': {
        'Original': {},
        'LoG': {'sigma': [1.0, 2.0, 3.0]},
        'Wavelet': {}
    },
    'featureClass': {
        'firstorder': [],
        'glcm': [],
        'glszm': [],
    },
    'setting': {
        'binWidth': bin_width,
        'force2D': False,  # Ensure 3D features are extracted
    }
    }

    params_peripheral = {
    'imageType': {
        'Original': {},
        'LoG': {'sigma': [1.0, 2.0]},
    },
    'featureClass': {
        'firstorder': [],
        'glcm': [],
    },
    'setting': {
        'binWidth': bin_width,
        'force2D': False,
    }
    }

    # Initialize the radiomics feature extractor
    extractor_dce1 = featureextractor.RadiomicsFeatureExtractor(params_dce1)
    extractor_other_dce = featureextractor.RadiomicsFeatureExtractor(params_other_dce)
    extractor_periph = featureextractor.RadiomicsFeatureExtractor(params_peripheral)

    # Collect all features
    all_rows = []

    # Iterate through each case
    for case in tqdm(case_names):  # Adjust range as needed
        dce_keys = get_dce_keys_from_folder(case, image_folder)
        try:
            # Load DCE images and mask
            images = {k: sitk.ReadImage(os.path.join(image_folder, f'{case}{k}.nii.gz'), sitk.sitkFloat32) for k in dce_keys}
            mask = sitk.ReadImage(os.path.join(mask_folder, f'{case}.nii.gz'), sitk.sitkUInt8)
            spacing = mask.GetSpacing()

            # Convert DCE0 and mask to arrays
            dce0_array = sitk.GetArrayFromImage(images['_0000'])
            mask_array = sitk.GetArrayFromImage(mask)

            if is_empty_mask(mask_array):
                print(f"⚠️  Skipping {case}: tumor mask is empty.")
                continue

            tumor_mask = (mask_array > 0).astype(bool) #boolean mask for tumor region
            # Get contrast related features

            core_mask = ErodeByFraction(tumor_mask, spacing, fraction=0.3)

            # Peripheral: tumor minus core
            peripheral_mask = np.logical_and(tumor_mask, np.logical_not(core_mask))

            # Get tumor intensities kenitics
            dce_intensities = get_tumour_intensities(images, tumor_mask)
            kinetic_features = compute_ordinal_kinetics(dce_intensities, sentinel=-1.0)
            # Compute tumor mean once
            t0_mean = dce0_array[tumor_mask].mean()

            # Normalize each DCE image by tumor mean
            images_norm = {k: normalize_by_tumor_mean(images[k], t0_mean) for k in dce_keys}

            core_constrast = get_peripheral_contrast_to_core(images_norm, core_mask, peripheral_mask)
            core_contrast_kenitics = contrast_kenitics(core_constrast, mode="peripheral_to_core", sentinel=-1.0)

            # Generate subtraction images (already normalized)
            sub1 = sitk.Subtract(images_norm['_0001'], images_norm['_0000'])
            sublast = sitk.Subtract(images_norm[dce_keys[-1]], images_norm['_0000'])

            # Extract radiomics features
            f_dce1 = {f'dce1_{k}': v for k, v in extractor_dce1.execute(images_norm['_0001'], mask).items() if 'diagnostics' not in k}
            f_sub1 = {f'sub1_{k}': v for k, v in extractor_other_dce.execute(sub1, mask).items() if 'diagnostics' not in k}
            f_sublast = {f'sublast_{k}': v for k, v in extractor_other_dce.execute(sublast, mask).items() if 'diagnostics' not in k}

            f_periph_dce1 = {
                            f'periph_dce1_{k}': v
                            for k, v in extractor_periph.execute(images_norm['_0001'], convert_mask_array_to_sitk(peripheral_mask, mask)).items()
                            if 'diagnostics' not in k
                        }
                        
            # Combine features
            row = {'CaseID': case}
            row.update(f_dce1)
            row.update(f_sub1)
            row.update(f_sublast)
            row.update(f_periph_dce1)
            row.update(kinetic_features)
            row.update(core_contrast_kenitics)
            
            all_rows.append(row)

        except Exception as e:
            print(f"❌ Failed to process {case}: {e}")


    # Convert to DataFrame and save
    df_all = pd.DataFrame(all_rows)
    print(f"Number of remaining features: {df_all.shape[1] - 1}") 
    output_path = os.path.join(output_path, f'subset_radiomics_features_{bin_width}.csv')
    df_all.to_csv(output_path, index=False)
    print(f"✅ Features saved to {output_path}")
