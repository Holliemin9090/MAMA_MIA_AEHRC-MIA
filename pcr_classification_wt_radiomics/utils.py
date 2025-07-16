
import os
import numpy as np
import SimpleITK as sitk
import argparse
from natsort import natsorted
import pandas as pd
from skimage.morphology import ball
from scipy.ndimage import binary_dilation, binary_erosion

def extract_image_id(directory):
    All_file_id = []
    for filename in os.listdir(directory):
        # Check if the file starts with 'breastDCE' and ends with '.nii.gz'
        if filename.startswith("breastDCE") and filename.endswith(".nii.gz"):
            parts = filename.split('_')
            
            new_filename = parts[0] + '_' + parts[1]
        
            All_file_id.append(new_filename)

        if filename.startswith("breastDWIADC") and filename.endswith(".nii.gz"):
            parts = filename.split('_')
            
            new_filename = parts[0] + '_' + parts[1]
        
            All_file_id.append(new_filename)
        
    return natsorted(list(set(All_file_id)))

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

def calculate_iou_dsc(pred_mask, gt_mask):
    # Calculate True Positive, False Positive, and False Negative
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    
    # Calculate Intersection over Union
    if tp + fp + fn == 0:
        return 0.0  # Return 0 if both pred_mask and gt_mask are empty
    iou = tp / (tp + fp + fn)
    
    # Calculate Dice Coefficient
    if 2 * tp + fp + fn == 0:
        return 0.0  # Return 0 if both pred_mask and gt_mask are empty
    dsc = (2 * tp) / (2 * tp + fp + fn)
    
    return iou, dsc

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
    
def Dilation(mask, radius = 5):
    # Create a disk-shaped structuring element with a radius of 5
    structuring_element = ball(radius)
    
    # Apply binary dilation to expand the mask by 5 pixels
    expanded_mask = binary_dilation(mask.astype(bool), structure=structuring_element)
    
    return expanded_mask

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



def save_array_to_image(array, save_path, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=None):
    """
    Converts a NumPy array to a SimpleITK image with specified metadata.

    Parameters:
        array (ndarray): 3D NumPy array to convert.
        spacing (tuple): Voxel spacing in (z, y, x).
        origin (tuple): Origin of the image.
        direction (tuple): Direction cosine matrix (9 elements for 3D).

    Returns:
        sitk.Image: SimpleITK image with metadata set.
    """
    sitk_image = sitk.GetImageFromArray(array)  # Converts from z,y,x to x,y,z in physical space

    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    
    if direction is None:
        direction = (1.0, 0.0, 0.0,
                     0.0, 1.0, 0.0,
                     0.0, 0.0, 1.0)
    sitk_image.SetDirection(direction)

    sitk.WriteImage(sitk_image, save_path)
    
