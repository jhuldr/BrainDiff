import shutil
from pathlib import Path

from dataset.dataset_utils import modify_name

MODALITIES = ["t1ce", "t1", "t2", "flair"]


def _is_seg(name: str) -> bool:
    return "seg" in name.lower()


def _is_excluded(name: str) -> bool:
    return "t1ce-t1" in name.lower() or _is_seg(name)


def _modality(name: str) -> str | None:
    lower = name.lower()
    for mod in MODALITIES:
        if mod in lower:
            return mod
    return None


def _timepoint(name: str) -> str | None:
    lower = name.lower()
    if "time2" in lower:
        return "time2"
    if "time1" in lower:
        return "time1"
    return None


def intake_ucfs(data_path: str | Path, output_path: str | Path) -> list[tuple]:
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    pairs = []

    for subfolder in sorted(p for p in data_path.iterdir() if p.is_dir()):
        nii_files = list(subfolder.glob("*.nii*"))

        seg_files = {_timepoint(f.name): f for f in nii_files if _is_seg(f.name)}
        mri_files = [f for f in nii_files if not _is_excluded(f.name)]

        mri_lookup: dict[tuple[str, str], Path] = {}
        for f in mri_files:
            tp = _timepoint(f.name)
            mod = _modality(f.name)
            if tp and mod:
                mri_lookup[(tp, mod)] = f

        if seg_files.get("time1") is None or seg_files.get("time2") is None:
            continue

        for f in mri_lookup.values():
            shutil.copy2(f, output_path / f.name)

        for mod in MODALITIES:
            t1_mri = mri_lookup.get(("time1", mod))
            t2_mri = mri_lookup.get(("time2", mod))
            if t1_mri is None or t2_mri is None:
                continue

            t1_lesion = modify_name(t1_mri.name, "lesion")
            t2_lesion = modify_name(t2_mri.name, "lesion")

            shutil.copy2(seg_files["time1"], output_path / t1_lesion)
            shutil.copy2(seg_files["time2"], output_path / t2_lesion)

            pairs.append((t1_mri.name, t2_mri.name, t1_lesion, t2_lesion))

    return pairs
