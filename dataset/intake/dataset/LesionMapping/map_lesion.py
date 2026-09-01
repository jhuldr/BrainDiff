from tqdm import tqdm
import nibabel as nib
import os
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from freesurfer_interface import *
from ants_interface import *
from dataset_utils import *

from tqdm import tqdm
import nibabel as nib

from freesurfer_interface import *
from dataset_utils import *


def generate_usb_aligned_lesions(input_path, input_names, lesion_path, lesion_output_path):
    pair_tuples = []

    lesion_output_path = Path(lesion_output_path)
    if not validate_path(lesion_output_path):
        make_directory(lesion_output_path)

    for file in tqdm(input_names, desc="Aligning lesions"):
        file = str(Path(file).name)
        file_path = Path(input_path) / file
        file_lesion_path = Path(lesion_path) / modify_name(file, "lesion")

        Path(lesion_output_path) / file_path.name

        if (lesion_output_path / file_lesion_path.name).exists():
            print(f"Skipping {file_lesion_path} - lesion already processed")
            pair_tuples.append((str(file_path), str(Path(lesion_output_path) / modify_name(file, "lesion"))))
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _, output = niphd_to_usb(file_path, file_lesion_path, image_output_path = tmpdir / Path(file).name, lesion_output_path=lesion_output_path / file_lesion_path.name)
            if output != "Error!":
                pair_tuples.append((str(file_path), str(Path(lesion_output_path) / str(output))))

    return pair_tuples