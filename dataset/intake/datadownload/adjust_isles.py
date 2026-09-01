import os
import shutil
import argparse
from pathlib import Path

from dataset.ants_interface import convert_image_to_target


def process_neuroimaging_files(input_dir, output_dir):
    """
    Process neuroimaging files from a BIDS-like directory structure.
    Aligns FLAIR to DWI space before saving.
    
    Structure expected:
    input_dir/
    ├── sub-XXX/
    │   └── ses-XXX/
    │       ├── anat/
    │       │   └── *_FLAIR.nii.gz
    │       └── dwi/
    │           └── *_dwi.nii.gz
    └── derivatives/
        └── sub-XXX/
            └── ses-XXX/
                └── *_msk.nii.gz
    
    Args:
        input_dir: Path to the input directory
        output_dir: Path to the output directory
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    files_processed = 0
    
    print("=" * 60)
    print("Processing FLAIR files and aligning to DWI space...")
    print("=" * 60)
    
    for subject_folder in input_path.iterdir():
        # Skip derivatives folder and non-directories
        if not subject_folder.is_dir() or subject_folder.name == 'derivatives':
            continue
        
        print(f"\nProcessing subject folder: {subject_folder.name}")
        
        # Open the next subfolder (e.g., session folder)
        for session_folder in subject_folder.iterdir():
            if not session_folder.is_dir():
                continue
            
            print(f"  Session folder: {session_folder.name}")
            
            # Look for anat and dwi folders
            anat_folder = session_folder / 'anat'
            dwi_folder = session_folder / 'dwi'
            
            # Check if both folders exist
            if not anat_folder.exists() or not anat_folder.is_dir():
                print(f"    Warning: No anat folder found in {session_folder.name}")
                continue
                
            if not dwi_folder.exists() or not dwi_folder.is_dir():
                print(f"    Warning: No dwi folder found in {session_folder.name}")
                continue
            
            # Find FLAIR file
            flair_files = list(anat_folder.glob('*_FLAIR.nii.gz'))
            if not flair_files:
                print(f"    Warning: No FLAIR file found in {anat_folder}")
                continue
            
            # Find DWI file
            dwi_files = list(dwi_folder.glob('*_dwi.nii.gz'))
            if not dwi_files:
                print(f"    Warning: No DWI file found in {dwi_folder}")
                continue
            
            flair_file = flair_files[0]
            dwi_file = dwi_files[0]
            
            print(f"    Found FLAIR: {flair_file.name}")
            print(f"    Found DWI: {dwi_file.name}")
            
            # Find corresponding mask in derivatives
            # Extract subject and session identifiers
            subject_name = subject_folder.name
            session_name = session_folder.name
            
            derivatives_path = input_path / 'derivatives' / subject_name / session_name
            mask_file = None
            
            if derivatives_path.exists():
                mask_files = list(derivatives_path.glob('*_msk.nii.gz'))
                if mask_files:
                    mask_file = mask_files[0]
                    print(f"    Found mask: {mask_file.name}")
                else:
                    print(f"    Warning: No mask file found in {derivatives_path}")
            else:
                print(f"    Warning: Derivatives path not found: {derivatives_path}")
            
            # Define output paths
            output_flair_path = output_path / flair_file.name
            
            if mask_file:
                # Rename mask from _msk.nii.gz to _FLAIR_lesion.nii.gz
                new_mask_name = mask_file.name.replace('_msk.nii.gz', '_FLAIR_lesion.nii.gz')
                output_mask_path = output_path / new_mask_name
            else:
                output_mask_path = None
            
            # Use your convert_image_to_target function
            print(f"    Aligning FLAIR to DWI space...")
            
            try:
                convert_image_to_target(
                    moving_image=str(flair_file),
                    fixed=str(dwi_file),
                    warped_image_path=str(output_flair_path),
                    affine_only=True
                    )
                shutil.copy2(mask_file, output_mask_path)
                
            
            except Exception as e:
                print(f"    Error processing {flair_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Processing complete! Total subjects processed: {files_processed}")
    print(f"Output directory: {output_path}")
    print("=" * 60)



if __name__ == '__main__':
    process_neuroimaging_files("/home/data/ISLES-2022-2/ISLES-2022", "/home/data/ISLES-2022-2/processed_data")