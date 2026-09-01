import shutil
from pathlib import Path
from collections import defaultdict


def get_subject_id(path: Path) -> str:
    """
    Extract subject ID from names like:
        BraTS-MET-00266-000
        BraTS-MET-00761-003

    Returns:
        00266
        00761
    """
    parts = path.name.split("-")

    if len(parts) < 4:
        raise ValueError(f"Unexpected file/folder name format: {path.name}")

    return parts[2]


def move_to_directory(src: Path, dst_dir: Path):
    """
    Move src into dst_dir.
    """
    dst = dst_dir / src.name

    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")

    shutil.move(str(src), str(dst))


def split_timepoints(input_dir: Path, single_timepoint_dir: Path, multi_timepoint_dir: Path):
    input_dir = input_dir.resolve()
    single_timepoint_dir.mkdir(parents=True, exist_ok=True)
    multi_timepoint_dir.mkdir(parents=True, exist_ok=True)

    subject_to_paths = defaultdict(list)

    # Collect all files/folders in input directory
    for path in input_dir.iterdir():
        if not path.name.startswith("BraTS-MET-"):
            continue

        subject_id = get_subject_id(path)
        subject_to_paths[subject_id].append(path)

    # Move each subject's files/folders based on number of timepoints
    for subject_id, paths in subject_to_paths.items():
        if len(paths) == 1:
            target_dir = single_timepoint_dir
        else:
            target_dir = multi_timepoint_dir

        for path in paths:
            print(f"Moving {path.name} -> {target_dir}")
            move_to_directory(path, target_dir)

if __name__ == "__main__":
    split_timepoints(
        input_dir=Path("/home/data/BraTS-MET"),
        single_timepoint_dir=Path("/home/data/BraTS-MET/single-timepoint"),
        multi_timepoint_dir=Path("/home/data/BraTS-MET/multi-timepoint"),
    )
