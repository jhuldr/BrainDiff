import os

import numpy as np
import tifffile
import time
from pathlib import Path
import shutil
import tempfile

from PIL import Image


def make_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)

def validate_path(path) -> bool:
    if not os.path.exists(path):
        return False
    return True

def generate_id():
    ts = int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(2), 'big')  # 2 random bytes

    combined = (ts << 16) | rand

    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    result = ""
    while combined:
        result = chars[combined % 62] + result
        combined //= 62

    return result

def save_image(image, output_path):

    if not validate_path(output_path):
        make_directory(output_path)

    file_name = f"mri_{generate_id()}.png"
    image_path = output_path / file_name

    # 99th percentile normalization to make it a little more robust
    p1, p99 = np.percentile(image, [1, 99])
    image = (image - p1) / (p99 - p1)
    image = image.clip(0, 1)

    image = (image * 255).astype(np.uint8)
    Image.fromarray(image, mode='L').convert('RGB').save(image_path)
    return image_path.resolve()

def modify_name(mri_name, type = "synthseg"):

    mri_name = str(mri_name)
    if mri_name.endswith('.nii.gz'):
        ext = '.nii.gz'
    else:
        ext = '.nii'
    name = mri_name.replace(ext, '')
    return f"{name}_{type}{ext}"


def move_lesion_files(root_dir: str | Path) -> Path:
    """
    Move all files under root_dir whose filename starts with 'x0' into a newly
    created temp directory, preserving their relative folder structure.

    Returns:
        temp_dir (Path): directory containing the moved files + a manifest file.
    """
    root_dir = Path(root_dir)
    temp_dir = Path(tempfile.mkdtemp())
    manifest = temp_dir / "_manifest.txt"

    with manifest.open("w", encoding="utf-8") as f:
        for p in root_dir.rglob("*"):
            if not (p.name.startswith("x0") or "lesion" in p.name):
                continue

            rel = p.relative_to(root_dir)          # preserve structure
            dst = temp_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(p, dst)
            f.write(str(rel) + "\n")               # record what we moved

    return temp_dir


def restore_lesion_files(root_dir: str | Path, temp_dir: str | Path) -> None:
    """
    Restore files moved by move lesion files back under root_dir.
    """
    root_dir = Path(root_dir)
    temp_dir = Path(temp_dir)
    manifest = temp_dir / "_manifest.txt"

    rel_paths = [Path(line.strip()) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    for rel in rel_paths:
        src = temp_dir / rel
        dst = root_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)

    manifest.unlink(missing_ok=True)