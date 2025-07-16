'''
Uitlity functions for the test suite.
'''
import os
import json
from natsort import natsorted
import pandas as pd
import SimpleITK as sitk
import numpy as np
import pickle

class MockRestrictedDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir  # e.g., '/path/to/mock_data'

    def get_patient_id_list(self):
        return natsorted([
        name for name in os.listdir(self.base_dir)
        if os.path.isdir(os.path.join(self.base_dir, name))
    ])

    def get_dce_mri_path_list(self, patient_id):
        image_dir = os.path.join(self.base_dir, patient_id, 'images')
        return natsorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])

    def read_json_file(self, patient_id):
        json_path = os.path.join(self.base_dir, patient_id, 'metadata.json')
        with open(json_path, 'r') as f:
            return json.load(f)
        
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

def restore_crop(image, mask_cropped_array, coords):
    x_min, x_max = coords["x_min"], coords["x_max"]
    y_min, y_max = coords["y_min"], coords["y_max"]
    z_min, z_max = coords["z_min"], coords["z_max"]

    image_array = sitk.GetArrayFromImage(image)
    mask_array = np.zeros_like(image_array, dtype = mask_cropped_array.dtype)

    mask_array[x_min:x_max, y_min:y_max, z_min:z_max] = mask_cropped_array 

    final_mask = sitk.GetImageFromArray(mask_array)
    final_mask.CopyInformation(image)          
    
    return final_mask

def read_probmap_npz(prob_map_npz_path):
    # prob_map_npz = np.load(prob_map_npz_path)
    
    # arrays_dict = {}
    # # # Iterate over all arrays in the .npz file and store them in the dictionary
    # for array_name in prob_map_npz.files:
    #     arrays_dict[array_name] = prob_map_npz[array_name]
    
    # Load the .npz file
    prob_map_npz = np.load(prob_map_npz_path)
    
    # Check if there is exactly one array
    if len(prob_map_npz.files) != 1:
        print("The .npz file should contain exactly one array.")
    
    # Extract the array
    array_name = prob_map_npz.files[0]
    array = prob_map_npz[array_name]
    
    return array[1,:,:,:]

def read_meta_pkl(pkl_file_path):
    # Open the .pkl file in binary read mode
    with open(pkl_file_path, 'rb') as file:
        # Load the data from the file
        pkl_data = pickle.load(file)     
    
    return pkl_data

def restore_ori_size_probmap(pkl_data, npz_data):
    prob_array = np.zeros(pkl_data['original_size_of_raw_data'], dtype=np.float16)
    prob_array[pkl_data['crop_bbox'][0][0]:pkl_data['crop_bbox'][0][1], pkl_data['crop_bbox'][1][0]:pkl_data['crop_bbox'][1][1], pkl_data['crop_bbox'][2][0]:pkl_data['crop_bbox'][2][1]]=npz_data
    return prob_array

def get_weighted_centroid(seg_probmap_array, DCE1, prob_threshold=0.1):
    '''
    Compute the weighted centroid of a lesion from the probability map.
    If no voxel exceeds the threshold, fall back to the centroid of the non-zero region in DCE1.
    '''

    # Get voxel coordinates and weights from the probability map
    coords = np.argwhere(seg_probmap_array > prob_threshold)
    weights = seg_probmap_array[seg_probmap_array > prob_threshold]

    if weights.sum() > 0:
        # Use probability map to compute weighted centroid
        centroid = np.average(coords, axis=0, weights=weights)
    else:
        # Fallback: use non-zero region of DCE1
        coords = np.argwhere(DCE1 > 0)
        if coords.size == 0:
            # If DCE1 is also empty, return center of the volume
            return np.array(seg_probmap_array.shape) // 2
        centroid = coords.mean(axis=0)

    centroid_rounded = np.round(centroid).astype(int)
    centroid_rounded = np.clip(centroid_rounded, [0, 0, 0], np.array(seg_probmap_array.shape) - 1)
    return centroid_rounded

def resample_to_spacing_1x1x1(image, spacing=(1.0, 1.0, 1.0), interpolation_method=sitk.sitkLinear):
    """
    Resample an image to 1x1x1 mm spacing.

    :param image: The input SimpleITK image to be resampled.
    :param spacing: Desired new spacing (default is 1x1x1 mm).
    :param interpolation_method: Interpolation method (e.g., sitk.sitkLinear, sitk.sitkNearestNeighbor).
    :return: Resampled SimpleITK image.
    """
    # Get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Compute new size based on the new spacing
    new_size = [
        int(round(original_size[i] * (original_spacing[i] / spacing[i])))
        for i in range(3)
    ]
    
    # Set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(spacing)
    resample.SetSize(new_size)
    resample.SetInterpolator(interpolation_method)
    
    # Preserve the original image's origin, direction, and pixel type
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetDefaultPixelValue(0)
    
    # Resample the image
    resampled_image = resample.Execute(image)
    
    return resampled_image

def resample_image_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0), interpolator=sitk.sitkLinear):
    # Original spacing and size
    original_spacing = image.GetSpacing()         # (x, y, z)
    original_size = image.GetSize()               # (x, y, z)
    shape = np.array(original_size)

    # Your line: compute the new shape
    new_shape = np.round((np.array(original_spacing) / np.array(target_spacing)).astype(float) * shape)

    # Convert shape back to SimpleITK (x, y, z) order
    new_size = tuple(int(x) for x in new_shape)

    # Set up the resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    # Perform resampling
    resampled_image = resampler.Execute(image)

    return resampled_image

def extract_fixed_bounding_box(image, centroid, box_size, pad_value=0):
    """
    Extracts a fixed-size bounding box centered at `centroid` from the image.
    If the box extends outside the image boundaries, the function will pad the result
    to ensure it is always of size `box_size`.
    
    Args:
        image (np.ndarray): The input image (e.g., 3D array).
        centroid (array-like): The center of the bounding box in voxel coordinates (z, y, x).
        box_size (tuple): The desired size of the box (z_size, y_size, x_size) in voxel units.
        pad_value (scalar): The value to use for padding when the crop goes out-of-bounds.
    
    Returns:
        np.ndarray: Cropped (and padded) image of shape `box_size`.
    """
    # Ensure inputs are numpy arrays
    centroid = np.array(np.round(centroid), dtype=int)
    box_size = np.array(box_size, dtype=int)
    
    # Compute half-sizes (using floor division)
    half_size = box_size // 2

    # Compute initial crop indices (may fall outside the image bounds)
    start = centroid - half_size
    end = start + box_size

    # Compute the amounts by which the indices fall outside the image
    pad_before = np.maximum(0, -start)
    pad_after  = np.maximum(0, end - np.array(image.shape))

    # Adjust crop indices so they are within the image
    start_in = np.maximum(start, 0)
    end_in = np.minimum(end, np.array(image.shape))
    
    # Extract the available part of the image
    crop = image[start_in[0]:end_in[0],
                 start_in[1]:end_in[1],
                 start_in[2]:end_in[2]]
    
    # Now pad the crop to the desired box_size
    padding = ((pad_before[0], pad_after[0]),
               (pad_before[1], pad_after[1]),
               (pad_before[2], pad_after[2]))
    crop_padded = np.pad(crop, padding, mode='constant', constant_values=pad_value)
    
    return crop_padded

def get_baseline_mean(DCE0_resampled_array, seg_resampled_array):
    if np.max(seg_resampled_array) > 0:
        masked_vals = DCE0_resampled_array[seg_resampled_array > 0]
    else:
        masked_vals = DCE0_resampled_array[DCE0_resampled_array > 0]

    if masked_vals.size > 0:
        baseline_mean = max(masked_vals.mean(), 1)
    else:
        baseline_mean = 1  # or np.nan, depending on how you want to handle this

    return baseline_mean

def needs_resampling(moving, reference):
    return (
        moving.GetSize() != reference.GetSize() or
        moving.GetSpacing() != reference.GetSpacing() or
        moving.GetOrigin() != reference.GetOrigin() or
        moving.GetDirection() != reference.GetDirection()
    )

def resample_if_needed(moving, reference, interpolator=sitk.sitkLinear):
    if needs_resampling(moving, reference):
        print("Resampling required")
        return sitk.Resample(
            moving,
            reference,
            sitk.Transform(),  # identity
            interpolator,
            0,
            moving.GetPixelID()
        )
    else:
        print("No resampling needed")
        return moving
    
def reconcile_metadata(moving, reference, interpolator=sitk.sitkLinear, atol=1e-5, verbose=False):
    """
    Adjusts the metadata of `moving` to match `reference` safely:
    - If metadata is exactly equal: do nothing
    - If metadata is almost equal (within atol): copy metadata
    - If metadata is substantially different: resample `moving` to `reference`
    """
    def metadata_equal(img1, img2):
        return (
            img1.GetSize() == img2.GetSize() and
            img1.GetSpacing() == img2.GetSpacing() and
            img1.GetOrigin() == img2.GetOrigin() and
            img1.GetDirection() == img2.GetDirection()
        )

    def metadata_almost_equal(img1, img2):
        return (
            img1.GetSize() == img2.GetSize() and
            np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=atol) and
            np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=atol) and
            np.allclose(np.array(img1.GetDirection()), np.array(img2.GetDirection()), atol=atol)
        )

    if metadata_equal(moving, reference):
        if verbose:
            print("✅ Metadata exactly matches. No action taken.")
        return moving

    elif metadata_almost_equal(moving, reference):
        if verbose:
            print("ℹ️ Metadata almost equal. Copying information.")
        moving.CopyInformation(reference)
        return moving

    else:
        # if verbose:
        print("⚠️ Metadata differs significantly. Resampling to match reference.")
        return sitk.Resample(
            moving,
            reference,
            sitk.Transform(),  # Identity transform
            interpolator,
            0,
            moving.GetPixelID()
        )

def seg_restore_wt_crop(images, patient_id, patient_info, output_dir_temp, output_dir_final):
    # segmentation restoration when the image is cropped to the breast region before feeding to the segmentation model
    DCE_1 = sitk.ReadImage(images[1])

    prob_map_raw_array = read_probmap_npz(os.path.join(output_dir_temp, f"{patient_id}.npz"))
    pkl_data = read_meta_pkl(os.path.join(output_dir_temp, f"{patient_id}.pkl"))

    if not np.array_equal(np.array(prob_map_raw_array.shape), pkl_data['original_size_of_raw_data']):
        print('prob map is not the same size as the preprocessed image! Therefore restore:')
        prob_map_cropped_array = restore_ori_size_probmap(pkl_data, prob_map_raw_array) # np array
    else: 
        prob_map_cropped_array = prob_map_raw_array # np array

    pred_mask_cropped = sitk.ReadImage(os.path.join(output_dir_temp, f"{patient_id}.nii.gz"))
    prob_map_cropped_array = prob_map_cropped_array.astype(np.float32)# sitk does not support float16
    # # save cropped probability map as SimpleITK image
    # prob_map_cropped_sitk = sitk.GetImageFromArray(prob_map_cropped_array)
    # prob_map_cropped_sitk.CopyInformation(pred_mask_cropped)  # Copy metadata
    # # Save the cropped probability map
    # prob_map_cropped_save_path = os.path.join(output_dir_temp, f"{patient_id}_prob.nii.gz")
    # sitk.WriteImage(prob_map_cropped_sitk, prob_map_cropped_save_path)

    # make sure the mask is not empty
    if sitk.GetArrayFromImage(pred_mask_cropped).max() <= 0:
        print(f"Warning: The mask for case {patient_id} has no positive values. Using probability map instead.")
        # Use the probability map to create a mask
        non_zero_values = prob_map_cropped_array[prob_map_cropped_array > 0]

        if non_zero_values.size > 0:
            dynamic_thresh = np.percentile(non_zero_values, 50)
            print(f"Using dynamic threshold (50th percentile): {dynamic_thresh:.4f}")
        else:
            print(f"Warning: No non-zero values found in probability map for {patient_id}. Using fallback threshold of 0.5.")
            dynamic_thresh = 0.5  # fallback

        pred_mask_array_cropped_revised = (prob_map_cropped_array >= dynamic_thresh).astype(np.uint8)
        pred_mask_cropped_revised = sitk.GetImageFromArray(pred_mask_array_cropped_revised)
        pred_mask_cropped_revised.CopyInformation(pred_mask_cropped)  # Copy metadata
        pred_mask_cropped = pred_mask_cropped_revised # overwrite the original mask with the revised one
       
    if not patient_info or "primary_lesion" not in patient_info:
        # no need to restore cropped image
        pred_mask = pred_mask_cropped
        
    else:
        # restore to the original image spacing based on the breast coordinates
        breast_coordinates = patient_info['primary_lesion']['breast_coordinates']
        # reverse the cropping of the breast region and save the final segmentations
        pred_mask_cropped_array = sitk.GetArrayFromImage(pred_mask_cropped)
        pred_mask = restore_crop(DCE_1, pred_mask_cropped_array, breast_coordinates)

    # MANDATORY: the segmentation masks should be named using the patient_id
    final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
    sitk.WriteImage(pred_mask, final_seg_path)


def calculate_tumor_size_kinetic(dce_files, patient_id, mask_path):
    full_stack = []
    # Read DCE1 (index 1) first as reference
    ref_image = sitk.ReadImage(dce_files[1])# DCE1

    # Add tumor mask 
    tumor_mask_path = os.path.join(mask_path, f"{patient_id}.nii.gz")
    tumor_mask = sitk.ReadImage(tumor_mask_path)
    tumor_mask = reconcile_metadata(tumor_mask, ref_image, interpolator=sitk.sitkNearestNeighbor)
    tumor_mask_array = sitk.GetArrayFromImage(tumor_mask)

    # if sitk.GetArrayViewFromImage(tumor_mask_cropped).max()<=0:
    #     raise ValueError(f'The cropped mask {tumor_mask_path} has no positive values, which means no tumor in this breast!')

    tumor_intensities = []
    for i, dce_file in enumerate(dce_files):
        
        image = sitk.ReadImage(dce_file)
        if i != 1:# Skip the first DCE as it is the reference
            image = reconcile_metadata(image, ref_image, interpolator=sitk.sitkLinear)

        image_array = sitk.GetArrayFromImage(image)
        full_stack.append(image_array)

        # Tumor mean intensity
        if np.sum(tumor_mask_array) > 0:
            tumor_values = image_array[tumor_mask_array > 0]
            tumor_mean = float(np.mean(tumor_values))
        else:
            tumor_mean = 0

        tumor_intensities.append(tumor_mean)

    full_stack_np = np.stack(full_stack, axis=0)

    case_metrics = {
        'Num_DCEs': len(dce_files),
        'Mean_Full': float(np.mean(full_stack_np)),
        'Std_Full': float(np.std(full_stack_np)),
    }

    if np.sum(tumor_mask_array) > 0:
        radius, volume, voxel_count = calculate_equivalent_radius(tumor_mask)

    else:
        radius = 0.0
        volume = 0.0
        voxel_count = 0

    case_metrics.update({
        'TumorRadius_mm': radius,
        'TumorVolume_mm3': volume,
        'TumorVoxelCount': voxel_count,
        'TumorIntensities': tumor_intensities,
    })

    return case_metrics

def calculate_equivalent_radius(mask):
    spacing = mask.GetSpacing()  # (x, y, z)
    voxel_volume = np.prod(spacing)
    array = sitk.GetArrayFromImage(mask)
    num_voxels = np.sum(array > 0)
    total_volume = num_voxels * voxel_volume
    radius = ((3 * total_volume) / (4 * np.pi)) ** (1 / 3)
    return radius, total_volume, int(num_voxels)

def compute_ordinal_kinetics_original(I, sentinel=-1.0):
    if not isinstance(I, (list, np.ndarray)) or len(I) < 2:
        return {
            "MaxEnhancement": sentinel,
            "TimeToPeak": sentinel,
            "WashInStep": sentinel,
            "WashOutStep": sentinel,
            "EarlyLateRatio": sentinel,
            "AUC": sentinel,
            "SER": sentinel,
        }

    pre = I[0]
    posts = I[1:]

    if pre == 0 or len(posts) == 0:
        return {
            "MaxEnhancement": sentinel,
            "TimeToPeak": sentinel,
            "WashInStep": sentinel,
            "WashOutStep": sentinel,
            "EarlyLateRatio": sentinel,
            "AUC": sentinel,
            "SER": sentinel,
        }

    max_enh = (max(posts) - pre) / pre if pre != 0 else sentinel
    time_to_peak = np.argmax([v - pre for v in posts]) / len(I)

    wash_in = (I[1] - pre) / pre if len(I) > 1 and pre != 0 else sentinel
    wash_out = (I[1] - I[-1]) / pre if len(I) > 2 and pre != 0 else sentinel
    early_late_ratio = I[-1] / I[1] if len(I) > 2 and I[1] != 0 else sentinel
    auc = (sum(posts) - pre) / pre * len(posts) if pre != 0 else sentinel
    ser_denom = I[-1] - pre
    ser = (I[1] - pre) / ser_denom if len(I) > 2 and ser_denom != 0 else sentinel

    return {
        "MaxEnhancement": float(max_enh),
        "TimeToPeak": float(time_to_peak),
        "WashInStep": float(wash_in),
        "WashOutStep": float(wash_out),
        "EarlyLateRatio": float(early_late_ratio),
        "AUC": float(auc),
        "SER": float(ser),
    }

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def majority_voting (all_preds, thresholds):
    '''
    all_preds: list of predictions from each fold (5 lists in on list)
    thresholds: list of thresholds from each fold
    '''
    all_binary_preds = []
    for i in range(len(thresholds)):
        all_binary_preds.append([1 if pred >= thresholds[i] else 0 for pred in all_preds[i]])

    all_binary_preds = np.array(all_binary_preds)
    majority_voting_prob = np.average(all_binary_preds, axis=0)
    majority_voting_pred = (majority_voting_prob >= 0.5).astype(int)
    return majority_voting_pred, majority_voting_prob

def resample_to_reference(image_to_resample, reference_image, interpolator=sitk.sitkNearestNeighbor):
    """
    Resample an image (e.g. mask or probability map) to match the reference image's spatial properties.

    Args:
        image_to_resample (SimpleITK.Image): The image to be resampled (e.g., from isotropic space).
        reference_image (SimpleITK.Image): The target image with desired spacing, size, origin, and direction.
        interpolator (SimpleITK interpolator): 
            - Use `sitkNearestNeighbor` for masks (categorical labels)
            - Use `sitkLinear` for probabilities

    Returns:
        SimpleITK.Image: Resampled image in the reference space
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image_to_resample)


def set_region_outside_breast_to_zero(segmentation, breast_coords, threshold=None):
    """
    Set the region outside the breast to zero in the segmentation mask or probability map.

    Args:
        segmentation (SimpleITK.Image): The segmentation mask or probability map.
        breast_coords (dict or None): Coordinates defining the breast region.
        threshold (float, optional): If provided, threshold the probability map at this value.

    Returns:
        SimpleITK.Image: Masked and optionally thresholded segmentation.
        float: Maximum value in the masked region.
    """
    seg_array = sitk.GetArrayFromImage(segmentation)

    # Optional thresholding if input is a probability map
    if threshold is not None:
        seg_array = (seg_array >= threshold).astype(np.uint8)

    # No breast coordinates → return whole image
    if breast_coords is None:
        print("Breast coordinates not provided. Returning full image.")

        if threshold is not None:
            seg = sitk.GetImageFromArray(seg_array)
            seg.CopyInformation(segmentation)

            return seg, np.max(seg_array)
        
        else:
            return segmentation, np.max(seg_array)
       
    # Crop to bounding box
    x_min, x_max = breast_coords["x_min"], breast_coords["x_max"]
    y_min, y_max = breast_coords["y_min"], breast_coords["y_max"]
    z_min, z_max = breast_coords["z_min"], breast_coords["z_max"]

    masked_seg_array = np.zeros_like(seg_array)
    masked_seg_array[x_min:x_max, y_min:y_max, z_min:z_max] = \
        seg_array[x_min:x_max, y_min:y_max, z_min:z_max]

    masked_seg_max = np.max(masked_seg_array)
    masked_seg_image = sitk.GetImageFromArray(masked_seg_array)
    masked_seg_image.CopyInformation(segmentation)

    return masked_seg_image, masked_seg_max

def seg_restore_Task012(images, patient_id, patient_info, output_dir_temp, output_dir_final, thresh=0.1):
    # segmentation restoration when the image is cropped to the breast region before feeding to the segmentation model
    DCE_1 = sitk.ReadImage(images[1])

    # prob_map_raw_array = read_probmap_npz(os.path.join(output_dir_temp, f"{patient_id}.npz"))
    # pkl_data = read_meta_pkl(os.path.join(output_dir_temp, f"{patient_id}.pkl"))

    # if not np.array_equal(np.array(prob_map_raw_array.shape), pkl_data['original_size_of_raw_data']):
    #     print('prob map is not the same size as the preprocessed image! Therefore restore:')
    #     prob_map_array = restore_ori_size_probmap(pkl_data, prob_map_raw_array) # np array
    # else: 
    #     prob_map_array = prob_map_raw_array # np array

    pred_mask = sitk.ReadImage(os.path.join(output_dir_temp, f"{patient_id}.nii.gz"), sitk.sitkUInt8)
    # prob_map_array = prob_map_array.astype(np.float32)# sitk does not support float16
    # # save probability map as SimpleITK image
    # prob_map = sitk.GetImageFromArray(prob_map_array)
    # prob_map.CopyInformation(pred_mask)  # Copy metadata

    mask_resampled = resample_to_reference(pred_mask, DCE_1, interpolator=sitk.sitkNearestNeighbor)
    
    if not patient_info or "primary_lesion" not in patient_info:
        # Breast coordinates do not exist.
       
        mask_resampled_refined, masked_segmentation_max = set_region_outside_breast_to_zero(mask_resampled, breast_coords=None, threshold=None)

    else:
        # restore to the original image spacing based on the breast coordinates
        breast_coordinates = patient_info['primary_lesion']['breast_coordinates']
        # reverse the cropping of the breast region and save the final segmentations
        mask_resampled_refined, masked_segmentation_max = set_region_outside_breast_to_zero(mask_resampled, breast_coords=breast_coordinates, threshold=None)

    # MANDATORY: the segmentation masks should be named using the patient_id
    final_seg_path = os.path.join(output_dir_final, f"{patient_id}.nii.gz")
    sitk.WriteImage(mask_resampled_refined, final_seg_path)

