import SimpleITK as sitk
import numpy as np
import os

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

def crop_breast(image, coords):
    start = [coords['z_min'], coords['y_min'], coords['x_min']]
    size = [
        coords['z_max'] - coords['z_min'],
        coords['y_max'] - coords['y_min'],
        coords['x_max'] - coords['x_min'],
    ]
    return sitk.RegionOfInterest(image, size=size, index=start)