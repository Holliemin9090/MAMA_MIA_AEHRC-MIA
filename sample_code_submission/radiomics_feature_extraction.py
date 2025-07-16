import SimpleITK as sitk
from radiomics import featureextractor
import logging
logging.getLogger('radiomics.glcm').setLevel(logging.ERROR)  # Or logging.CRITICAL to suppress everything
import os
import numpy as np
from skimage.morphology import ball
from scipy.ndimage import binary_erosion
from utils import reconcile_metadata, resample_image_to_isotropic, crop_breast
                  

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

def ErodeByFraction(mask, spacing, fraction=0.3):
    """
    Erodes a binary tumor mask by a fraction of its equivalent radius.

    Args:
        mask (ndarray): Binary tumor mask (3D). boolean type is expected.
        spacing (tuple): Voxel spacing (z, y, x).
        fraction (float): Fraction of equivalent radius to erode.

    Returns:
        ndarray: Eroded binary mask.
    """
    voxel_volume = np.prod(spacing)
    tumor_voxels = np.sum(mask)
    tumor_volume = tumor_voxels * voxel_volume

    # Compute equivalent radius (in mm)
    r_eq_mm = (3 * tumor_volume / (4 * np.pi))**(1/3)
    erosion_mm = fraction * r_eq_mm

    # Convert erosion radius from mm to voxel units (use mean spacing for isotropy)
    mean_spacing = np.mean(spacing)
    erosion_radius_voxels = max(1, int(round(erosion_mm / mean_spacing)))

    # Create spherical structuring element
    struct_elem = ball(erosion_radius_voxels)

    # Apply erosion
    eroded_mask = binary_erosion(mask.astype(bool), structure=struct_elem)

    return eroded_mask

def initialize_radiomics_extractors(bin_width=0.5):
    """
    Initialize radiomics feature extractors with specified parameters.
    
    Parameters:
    - bin_width: The width of the bins for histogram features.
    
    Returns:
    - extractor_dce1: Extractor for DCE 1 features.
    - extractor_other_dce: Extractor for other DCE features.
    - extractor_periph: Extractor for peripheral features.
    """
    
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

    return extractor_dce1, extractor_other_dce, extractor_periph

def process_data_and_extract_radiomics(DCE_image_names, patient_info, patient_id, seg_folder_path, extractor_dce1, extractor_other_dce, extractor_periph):
    images = {}
    DCE_1 = sitk.ReadImage(DCE_image_names[1])  

    for i, dce_file in enumerate(DCE_image_names):
        
        if i != 1:       
            DCE = sitk.ReadImage(dce_file)
            print(f"Processing DCE file: {dce_file}")
            DCE = reconcile_metadata(DCE, DCE_1, interpolator=sitk.sitkLinear)

            if not patient_info or "primary_lesion" not in patient_info:
                DCE_cropped = DCE
            else:
                breast_coordinates = patient_info['primary_lesion']['breast_coordinates']
                DCE_cropped = crop_breast(DCE, breast_coordinates)

            DCE_resampled_cropped = resample_image_to_isotropic(DCE_cropped, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)

            images[f'_{i:04d}'] = sitk.Cast(DCE_resampled_cropped, sitk.sitkFloat32)  # Ensure images are in float32 format
        else:
            # Process the first DCE image separately
            print(f"Processing DCE file: {dce_file}")
            DCE_1_cropped = crop_breast(DCE_1, patient_info['primary_lesion']['breast_coordinates']) if patient_info and "primary_lesion" in patient_info else DCE_1
            DCE_1_resampled_cropped = resample_image_to_isotropic(DCE_1_cropped, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear)
            images['_0001'] = sitk.Cast(DCE_1_resampled_cropped, sitk.sitkFloat32)  # Ensure images are in float32 format
            
    dce_keys = sorted(list(images.keys()))
    mask_label_path = os.path.join(seg_folder_path, f"{patient_id}.nii.gz")
    mask = sitk.ReadImage(mask_label_path)
    mask = reconcile_metadata(mask, DCE_1, interpolator=sitk.sitkNearestNeighbor)
    # Check if breast coordinates are available
    mask_cropped = crop_breast(mask, patient_info['primary_lesion']['breast_coordinates']) if patient_info and "primary_lesion" in patient_info else mask
   
    mask_resampled_cropped = resample_image_to_isotropic(mask_cropped, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkNearestNeighbor)
    mask_resampled_cropped = sitk.Cast(mask_resampled_cropped, sitk.sitkUInt8)  # Ensure mask is in uint8 format

    spacing = mask_resampled_cropped.GetSpacing()
    # extract radiomics features

    dce0_array = sitk.GetArrayFromImage(images['_0000'])
    mask_array = sitk.GetArrayFromImage(mask_resampled_cropped)

    if is_empty_mask(mask_array):
        print(f"⚠️  Skipping {patient_id}: tumor mask is empty.")
        # continue
        row = None  # No features to extract
    else:

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
        f_dce1 = {f'dce1_{k}': v for k, v in extractor_dce1.execute(images_norm['_0001'], mask_resampled_cropped).items() if 'diagnostics' not in k}
        f_sub1 = {f'sub1_{k}': v for k, v in extractor_other_dce.execute(sub1, mask_resampled_cropped).items() if 'diagnostics' not in k}
        f_sublast = {f'sublast_{k}': v for k, v in extractor_other_dce.execute(sublast, mask_resampled_cropped).items() if 'diagnostics' not in k}

        f_periph_dce1 = {
                        f'periph_dce1_{k}': v
                        for k, v in extractor_periph.execute(images_norm['_0001'], convert_mask_array_to_sitk(peripheral_mask, mask_resampled_cropped)).items()
                        if 'diagnostics' not in k
                    }
                    
        # Combine features
        row = {'patient_id': patient_id}
        row.update(f_dce1)
        row.update(f_sub1)
        row.update(f_sublast)
        row.update(f_periph_dce1)
        row.update(kinetic_features)
        row.update(core_contrast_kenitics)
        
    return row

