"""
import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "16"
"""

import ants
from scipy.ndimage import rotate
from pathlib import Path
import shutil
import numpy as np
import tempfile


from dataset.dataset_utils import *

def ants_resample_to_target(image, target, output, interp_type="linear", imagetype=0, verbose=False):
    """
    Resample an image to match the space of a target reference image.
    Uses an identity transform via ants.apply_transforms to achieve proper resampling.

    Parameters
    ----------
    image : str or Path
        Path to the image to resample.
    target : str or Path
        Path to the reference image; output will match its resolution/origin/orientation/direction.
    output : str or Path
        Path where the resampled image will be written.
    interp_type : str
        Interpolation method. One of:
        linear, nearestNeighbor, multiLabel, gaussian, bSpline,
        cosineWindowedSinc, welchWindowedSinc, hammingWindowedSinc,
        lanczosWindowedSinc, genericLabel.
    imagetype : int
        0=scalar, 1=vector, 2=tensor, 3=time-series.
    verbose : bool
        Print command and run verbose application of transform.

    Returns
    -------
    Path
        Path to the written resampled image, or "Error" on failure.
    """
    moving = ants.image_read(str(image))
    fixed  = ants.image_read(str(target))

    try:
        resampled = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[],       # identity — no transform, just resample
            interpolator=interp_type,
            imagetype=imagetype,
            verbose=verbose,
        )
        ants.image_write(resampled, str(output))
    except Exception as e:
        print("An error occurred during resampling:")
        print(e)
        return "Error"

    return Path(output)


def convert_mni_to_usb(moving_image, output_path, lesion_path=None, lesion_output_path=None, interpolator="linear", return_image: bool = False):
    fixed = ants.image_read("/path/to/code/BrainDiff/dataset/ants_data/mask_13.nii.gz")
    
    if type(moving_image) == ants.core.ANTsImage:
        moving = moving_image
    else:
        moving = ants.image_read(str(moving_image))

    try:
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=["/path/to/code/BrainDiff/dataset/ants_data/inv_warp.nii.gz", "/path/to/code/BrainDiff/dataset/ants_data/fwd_affine.mat"],
            whichtoinvert=[False, True],
            interpolator=interpolator,
        )
        # rotate it into USB space        
        data = warped.numpy()
        rotated_data = rotate(data, angle=-245, axes=(1, 2), reshape=False)
        warped = ants.from_numpy(
            rotated_data,
            origin=warped.origin,
            spacing=warped.spacing,
            direction=warped.direction
        )

        # Apply the same inverse transform + rotation to the lesion if provided
        warped_lesion = None
        if lesion_path is not None:
            if type(lesion_path) == ants.core.ANTsImage:
                lesion = lesion_path
            else:
                lesion = ants.image_read(str(lesion_path))

            warped_lesion = ants.apply_transforms(
                fixed=fixed,
                moving=lesion,
                transformlist=["/path/to/code/BrainDiff/dataset/ants_data/inv_warp.nii.gz", "/path/to/code/BrainDiff/dataset/ants_data/fwd_affine.mat"],
                whichtoinvert=[False, True],
                interpolator="nearestNeighbor",  # preserve binary mask values
            )
            lesion_data = warped_lesion.numpy()
            rotated_lesion_data = rotate(lesion_data, angle=-245, axes=(1, 2), reshape=False, order=0)
            warped_lesion = ants.from_numpy(
                rotated_lesion_data,
                origin=warped_lesion.origin,
                spacing=warped_lesion.spacing,
                direction=warped_lesion.direction
            )

        if return_image:
            return warped, warped_lesion
        
        if output_path is not None:
            ants.image_write(warped, str(output_path))
            output_path = Path(output_path)

        if warped_lesion is not None:
            ants.image_write(warped_lesion, str(lesion_output_path))
            return output_path, Path(lesion_output_path)

    except Exception as e:
        print("An error occurred while applying inverse ElasticSyN transforms:")
        print(e)
        return None

    return Path(output_path)


def convert_niphd_to_usb(moving_image, output_path, lesion_path=None, lesion_output_path=None, interpolator="linear", return_image: bool = False):
    fixed = ants.image_read("/path/to/code/BrainDiff/dataset/ants_data/mask_13.nii.gz")
    
    if type(moving_image) == ants.core.ANTsImage:
        moving = moving_image
    else:
        moving = ants.image_read(str(moving_image))

    # Step 1: Rotate the fixed image 245 degrees
    fixed_data = fixed.numpy()
    rotated_fixed_data = rotate(fixed_data, angle=245, axes=(1, 2), reshape=False)
    rotated_fixed = ants.from_numpy(
        rotated_fixed_data,
        origin=fixed.origin,
        spacing=fixed.spacing,
        direction=fixed.direction
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            # Step 2: Affine register moving to rotated fixed
            registration = ants.registration(
                fixed=rotated_fixed,
                moving=moving,
                type_of_transform="Affine",
                outprefix=str(Path(tmp_dir) / "tx_"),
            )
        except Exception as e:
            print("An error occurred during Affine registration:")
            print(e)
            return None

        fwdtransforms = registration["fwdtransforms"]  # Affine: [affine.mat]

        # Step 3: Apply forward transform to get from NIPHD to rotated USB space
        warped = ants.apply_transforms(
            fixed=rotated_fixed,
            moving=moving,
            transformlist=fwdtransforms,
            interpolator=interpolator,
        )

        # Step 4: Rotate the moving image -245 degrees (back to original orientation)
        data = warped.numpy()
        rotated_data = rotate(data, angle=-245, axes=(1, 2), reshape=False)
        warped = ants.from_numpy(
            rotated_data,
            origin=fixed.origin,  # Use original fixed origin
            spacing=fixed.spacing,
            direction=fixed.direction
        )

        # Apply the same transform + rotation to the lesion if provided
        warped_lesion = None
        if lesion_path is not None:
            if type(lesion_path) == ants.core.ANTsImage:
                lesion = lesion_path
            else:
                lesion = ants.image_read(str(lesion_path))

            # Step 3 for lesion: Apply forward transform
            warped_lesion = ants.apply_transforms(
                fixed=rotated_fixed,
                moving=lesion,
                transformlist=fwdtransforms,
                interpolator="nearestNeighbor",  # preserve binary mask values
            )

            # Step 4 for lesion: Rotate -245 degrees
            lesion_data = warped_lesion.numpy()
            rotated_lesion_data = rotate(lesion_data, angle=-245, axes=(1, 2), reshape=False, order=0)
            warped_lesion = ants.from_numpy(
                rotated_lesion_data,
                origin=fixed.origin,  # Use original fixed origin
                spacing=fixed.spacing,
                direction=fixed.direction
            )

        if return_image:
            return warped, warped_lesion

        if output_path is not None:
            ants.image_write(warped, str(output_path))
            output_path = Path(output_path)

        if warped_lesion is not None:
            ants.image_write(warped_lesion, str(lesion_output_path))
            return output_path, Path(lesion_output_path)

    return Path(output_path)



def convert_image_to_target(moving_image, fixed = None, output_dir=None, warped_image_path=None, lesion_path=None, lesion_output_path=None, return_image: bool = False, input_USB: bool = False, affine_only: bool = False):
    """
    Parameters
    ----------
    moving_image : str, Path, or ANTsImage
        The moving image to register.
    output_dir : str or Path
        Directory where the transform files will be saved.
        If affine_only=False: fwd_warp.nii.gz, fwd_affine.mat, inv_warp.nii.gz
        If affine_only=True:  fwd_affine.mat, inv_affine.mat
    warped_image_path : str or Path, optional
        Explicit path to save the warped image. If None, saves to output_dir/warped_image.nii.gz.
    lesion_path : str, Path, or ANTsImage, optional
        Lesion mask to warp using the same transform.
    lesion_output_path : str or Path, optional
        Explicit path to save the warped lesion. If None and lesion_path is provided,
        saves to output_dir/warped_lesion.nii.gz.
    return_image : bool, optional
        If True, return (warped_image, warped_lesion) and do not save images to disk.
        warped_lesion will be None if lesion_path is not supplied.
    input_USB : bool, optional
        If True, apply pre-registration rotation to convert from USB space.
    affine_only : bool, optional
        If True, use Affine registration instead of ElasticSyN. Saves only
        affine transform files (no warp fields).

    Returns
    -------
    If return_image is True:
        tuple of (warped_image, warped_lesion) as ANTs image objects,
        where warped_lesion is None if lesion_path was not supplied.
    If return_image is False and lesion_path is provided:
        tuple of (warped_image_path, lesion_output_path) as Paths.
    If return_image is False and no lesion_path:
        Path to the saved warped image.
    None on failure.
    """
    if fixed is None:
        fixed = ants.image_read("/path/to/code/BrainDiff/dataset/ants_data/mni_reference.nii.gz")
    else:
        fixed = ants.image_read(fixed)

    if type(moving_image) == ants.core.ANTsImage:
        moving = moving_image
    else:
        moving = ants.image_read(str(moving_image))

    # if the input is USB, rotate it to common space
    if input_USB:
        data = moving.numpy()
        rotated_data = rotate(data, angle=245, axes=(1, 2), reshape=False)
        moving = ants.from_numpy(
            rotated_data,
            origin=moving.origin,
            spacing=moving.spacing,
            direction=moving.direction
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(output_dir) if output_dir is not None else Path(tmp_dir)
        transform_type = "Affine" if affine_only else "ElasticSyN"
        try:
            registration = ants.registration(
                fixed=fixed,
                moving=moving,
                type_of_transform=transform_type,
                outprefix=str(Path(tmp_dir) / "tx_"),
            )
        except Exception as e:
            print(f"An error occurred during {transform_type} registration:")
            print(e)
            return None

        fwd = registration["fwdtransforms"]  # Affine: [affine.mat] | ElasticSyN: [warp.nii.gz, affine.mat]
        inv = registration["invtransforms"]  # Affine: [affine.mat] | ElasticSyN: [affine.mat, inv_warp.nii.gz]

        warped_image = registration["warpedmovout"]

        # Apply the same transform to the lesion mask if provided
        warped_lesion = None
        if lesion_path is not None:
            if type(lesion_path) == ants.core.ANTsImage:
                lesion = lesion_path
            else:
                lesion = ants.image_read(str(lesion_path))

            if input_USB:
                lesion_data = lesion.numpy()
                rotated_lesion_data = rotate(lesion_data, angle=245, axes=(1, 2), reshape=False, order=0)
                lesion = ants.from_numpy(
                    rotated_lesion_data,
                    origin=lesion.origin,
                    spacing=lesion.spacing,
                    direction=lesion.direction
                )

            warped_lesion = ants.apply_transforms(
                fixed=fixed,
                moving=lesion,
                transformlist=fwd,
                interpolator="nearestNeighbor",
            )

        if return_image:
            return warped_image, warped_lesion

        # Copy transform files
        if affine_only:
            shutil.copy(fwd[0], output_dir / "fwd_affine.mat")
            shutil.copy(inv[0], output_dir / "inv_affine.mat")
        else:
            shutil.copy(fwd[0], output_dir / "fwd_warp.nii.gz")
            shutil.copy(fwd[1], output_dir / "fwd_affine.mat")
            shutil.copy(inv[1], output_dir / "inv_warp.nii.gz")

        # Save warped image
        warped_image_path = Path(warped_image_path) if warped_image_path is not None else output_dir / "warped_image.nii.gz"
        ants.image_write(warped_image, str(warped_image_path))

        # Save warped lesion if computed
        if warped_lesion is not None:
            lesion_output_path = Path(lesion_output_path) if lesion_output_path is not None else output_dir / "warped_lesion.nii.gz"
            ants.image_write(warped_lesion, str(lesion_output_path))
            return warped_image_path, lesion_output_path

    return warped_image_path


def ants_apply_elasticsyn(moving_path, fixed_path, output_path, fwd_warp, fwd_affine, interpolator="linear"):
    """
    Apply a precomputed ElasticSyN transform to a moving image.

    Parameters
    ----------
    moving_path : str or Path
    fixed_path : str or Path
        Reference image defining the output space.
    output_path : str or Path
    fwd_warp : str or Path
        Path to fwd_warp.nii.gz
    fwd_affine : str or Path
        Path to fwd_affine.mat
    interpolator : str
        e.g. "linear", "nearestNeighbor", "multiLabel"
    """
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    try:
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[str(fwd_warp), str(fwd_affine)],
            interpolator=interpolator,
        )
        ants.image_write(warped, str(output_path))
    except Exception as e:
        print("An error occurred while applying ElasticSyN transforms:")
        print(e)
        return None

    return Path(output_path)


# ATLAS is already aligned to MNI
def atlas_to_usb(moving_image, lesion_path, output_path, image_output_path = None, lesion_output_path = None):
    #img, lesion = convert_image_to_target(moving_image, output_path, lesion_path=lesion_path, return_image=True)
    return convert_mni_to_usb(moving_image, output_path=image_output_path, lesion_path=lesion_path, lesion_output_path=lesion_output_path, return_image=False)

def usb_to_mni(moving_image, lesion_path, image_output_path, lesion_output_path):
    return convert_image_to_target(moving_image=moving_image, warped_image_path=image_output_path, lesion_path=lesion_path, lesion_output_path=lesion_output_path, input_USB=True, affine_only=True)

def align_folder_mni(input_folder, lesion_folder):
    for file in Path(input_folder).iterdir():
        lesion_path = Path(lesion_folder) / modify_name(file.name, "lesion")
        convert_image_to_target(moving_image=file, warped_image_path=file,lesion_path=lesion_path, lesion_output_path=lesion_path, input_USB=True, affine_only=True)

def align_folder_target(target, file_pairs, mri_folder, lesion_folder):
    for brain, lesion in file_pairs:
        if mri_folder / brain.name:
            print(f"File has already been aligned: {brain.name}")
            continue
        lesion = Path(lesion)
        brain = Path(brain)
        convert_image_to_target(moving_image= str(brain), warped_image_path= str(mri_folder / brain.name), lesion_path= str(lesion), lesion_output_path= str(lesion_folder / lesion.name), input_USB=True, affine_only=True, fixed = str(target))

def align_image_to_nihpd(input_file, output_path, lesion_path=None, lesion_output_path=None):
    file = Path(input_file)
    output_path = Path(output_path)
    if Path(output_path).exists() and (lesion_output_path is None or Path(lesion_output_path).exists()):
        print(f"File has already been aligned: {output_path.name}")
        return
    return convert_image_to_target(moving_image=file, fixed="/path/to/code/BrainDiff/dataset/ants_data/nihpd_ref.nii.gz", warped_image_path= output_path, lesion_path=lesion_path, lesion_output_path=lesion_output_path, input_USB=False, affine_only=True)


def align_modalities_to_mni(output_dir, centered_image, t1w=None, t1ce=None,
                              t2w=None, flair=None, interpolator="linear"):
    """Align a study's modalities into MNI space and save each as {modality}.nii.gz.

    The centered image is affinely registered to the MNI reference once; because
    the modalities are already co-registered within the study, that single transform
    is applied to every supplied modality so they land in a consistent space.
    """
    fixed = ants.image_read("/path/to/code/BrainDiff/dataset/ants_data/mni_reference.nii.gz")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modalities = {"t1w": t1w, "t1ce": t1ce, "t2w": t2w, "flair": flair}
    modalities = {name: path for name, path in modalities.items() if path is not None}

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            registration = ants.registration(
                fixed=fixed,
                moving=ants.image_read(str(centered_image)),
                type_of_transform="Affine",
                outprefix=str(Path(tmp_dir) / "tx_"),
            )
        except Exception as e:
            print("An error occurred during Affine registration:")
            print(e)
            return None

        fwd = registration["fwdtransforms"]  # [affine.mat]

        for name, path in modalities.items():
            warped = ants.apply_transforms(
                fixed=fixed,
                moving=ants.image_read(str(path)),
                transformlist=fwd, 
                interpolator=interpolator,
            )
            ants.image_write(warped, str(output_dir / f"{name}.nii.gz"))

    return output_dir


def align_image_to_mni(input_file, output_path, lesion_path=None, lesion_output_path=None):
    file = Path(input_file)
    output_path = Path(output_path)
    if Path(output_path).exists() and (lesion_output_path is None or Path(lesion_output_path).exists()):
        print(f"File has already been aligned: {output_path.name}")
        return
    return convert_image_to_target(moving_image=file, warped_image_path= output_path, lesion_path=lesion_path, lesion_output_path=lesion_output_path, input_USB=False, affine_only=True)


def niphd_to_usb(moving_image, lesion_path, image_output_path = None, lesion_output_path = None):
    return convert_niphd_to_usb(moving_image, output_path=image_output_path, lesion_path=lesion_path, lesion_output_path=lesion_output_path, return_image=False)

def usb_to_nihpd(moving_image, lesion_path, image_output_path, lesion_output_path):
    return convert_image_to_target(moving_image=moving_image, warped_image_path=image_output_path, lesion_path=lesion_path, lesion_output_path=lesion_output_path, input_USB=True, affine_only=True)
