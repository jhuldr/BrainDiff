import argparse
import re
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dataset3D.ExtractData.extract_meta_positional import ExtractMetaDataPositional
from dataset3D.dataset_adapter import *
from dataset.ants_interface import align_image_to_mni


MODALITY_COLUMN = {"T1W": "T1w", "T2W": "T2w", "T1CE": "T1ce", "FLAIR": "FLAIR"}

# Files that are labels, not imaging. They must never be classified as a modality:
# extract_modality_brats_men falls through to "T1w" for anything it doesn't
# recognise, so a BraTS "-seg.nii.gz" competes with "-t1n.nii.gz" for the T1w slot
# and whichever iterdir() yields last wins. Today "-seg" happens to sort before
# "-t1n" so the image wins, but iterdir() order is not guaranteed and a relabelled
# T1w volume is silent -- it looks like a valid file of the right name.
NON_MODALITY_TOKENS = ("seg", "lesion", "mask", "label")


def is_modality_file(path) -> bool:
    return not any(tok in Path(path).name.lower() for tok in NON_MODALITY_TOKENS)


def parse_timepoint(folder_name):
    """Extract the timepoint number from a folder name (e.g. "visit_01" -> 1)."""
    folder_name = folder_name.split("-")[-1]
    match = re.search(r"\d+", folder_name)
    return int(match.group()) if match else None


def collect_modality_files(folder):
    """Files directly in `folder`, or if there are none, files one level down."""
    files = [p for p in folder.iterdir() if p.is_file()]
    if files:
        return files
    for child in sorted(p for p in folder.iterdir() if p.is_dir()):
        files.extend(p for p in child.iterdir() if p.is_file())
    return files


def generate_longitudinal_positional_dataset(data_path, aligned_data_path,
                                              output_path,
                                              output_csv_name: str = "longitudinal.csv",
                                              extract_modality_fn=None,
                                              find_lesion_fn=None,
                                              lesion_dir=None,
                                              shard: int = 0, nshards: int = 1):
    """
    Longitudinal variant of Pipeline3D.generate_positional_dataset.

    `data_path` contains per-subject folders, each holding timepoint subfolders
    (named with a digit somewhere in the name, e.g. "0", "visit_01"). Each
    timepoint folder is processed like a study folder in
    generate_positional_dataset: multiple modality files (classified via
    extract_modality_fn) plus an optional lesion (find_lesion_fn), aligned to
    MNI. No segmentation or box extraction runs here — each timepoint yields
    one CSV row with its aligned modality/lesion paths, StudyUID (one uuid4
    per subject, shared across its timepoints), Timepoint (parsed from the
    folder name), and Pathology.

    `extract_modality_fn`, if given, takes a file Path and returns its modality
    column name (one of "T1w"/"T1ce"/"T2w"/"FLAIR"). Defaults to
    `ExtractMetaDataPositional.extract_modality` + MODALITY_COLUMN.

    `find_lesion_fn`, if given, takes a timepoint folder Path and returns the
    lesion file Path, or None if that timepoint has no lesion (e.g.
    `dataset_adapter.find_seg_brats_men`). The lesion is warped with the SAME
    affine as each modality, nearest-neighbour, by
    `ants_interface.convert_image_to_target` -- so one mask is written PER
    MODALITY, not per timepoint. That is not redundancy: this function aligns
    every modality independently, so a single mask would be correct for at most
    one of them. `lesion_dir` is where they go.

    `shard`/`nshards` slice `subject_folders` for parallelism. Safe because the
    StudyUID is a uuid4 minted per subject, so shards never collide; each writes
    `<output_csv_name stem>_shard<i>.csv` for the caller to concatenate.
    """
    pos_meta_instance = ExtractMetaDataPositional()

    def default_extract_modality(f):
        return MODALITY_COLUMN.get(pos_meta_instance.extract_modality(f), "T1w")

    extract_modality_fn = extract_modality_fn or default_extract_modality

    data_path = Path(data_path)
    aligned_data_path = Path(aligned_data_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)
    aligned_data_path.mkdir(parents=True, exist_ok=True)
    if lesion_dir is not None:
        lesion_dir = Path(lesion_dir)
        lesion_dir.mkdir(parents=True, exist_ok=True)

    subject_folders = sorted(p for p in data_path.iterdir() if p.is_dir())
    if nshards > 1:
        subject_folders = subject_folders[shard::nshards]
        print(f"[shard {shard}/{nshards}] {len(subject_folders)} subjects")

    rows = []

    for subject_folder in tqdm(subject_folders, desc="Aligning subjects", unit="subject"):
        study_uid = str(uuid.uuid4())

        timepoint_folders = [
            (p, parse_timepoint(p.name)) for p in subject_folder.iterdir() if p.is_dir()
        ]
        timepoint_folders = sorted(
            (pair for pair in timepoint_folders if pair[1] is not None),
            key=lambda pair: pair[1],
        )

        for timepoint_folder, timepoint in tqdm(
            timepoint_folders, desc=f"{subject_folder.name} timepoints", unit="timepoint", leave=False
        ):
            candidate_files = collect_modality_files(timepoint_folder)
            col_to_src = {}
            for f in candidate_files:
                # Labels are not modalities. Filter BEFORE classifying: the BraTS
                # adapter's else-branch returns "T1w" for anything unrecognised,
                # which would let "-seg.nii.gz" overwrite the real T1w image.
                if not is_modality_file(f):
                    continue
                col = extract_modality_fn(f)
                if col is None:
                    continue        # was `pass`, which fell through to col_to_src[None] = f
                col_to_src[col] = f

            prefix = f"{subject_folder.name}_{timepoint_folder.name}"

            lesion_src = find_lesion_fn(timepoint_folder) if find_lesion_fn else None

            modality_paths = {}
            lesion_paths = {}
            for col, src in col_to_src.items():
                aligned_img = aligned_data_path / f"{prefix}_{col}.nii.gz"
                aligned_les = None
                if lesion_src is not None and lesion_dir is not None:
                    # One mask per modality: each modality gets its own affine
                    # below, so a single shared mask would be misaligned for
                    # three of the four.
                    aligned_les = lesion_dir / f"{prefix}_{col}_lesion.nii.gz"
                align_image_to_mni(src, aligned_img,
                                   lesion_path=lesion_src,
                                   lesion_output_path=aligned_les)
                modality_paths[col] = aligned_img
                if aligned_les is not None:
                    lesion_paths[col] = aligned_les

            row = {"StudyUID": study_uid, "Timepoint": timepoint}
            for col in ("T1w", "T1ce", "T2w", "FLAIR"):
                path = modality_paths.get(col)
                row[col] = str(path) if path is not None else None
            # Inert downstream -- dataloaders read only the four modality columns.
            row["Lesion"] = str(lesion_src) if lesion_src is not None else None
            rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_csv_name is None:
        csv_path = output_path / f"meta_df_{timestamp}.csv"
    else:
        csv_path = output_path / output_csv_name
    if nshards > 1:
        csv_path = csv_path.with_name(f"{csv_path.stem}_shard{shard:03d}{csv_path.suffix}")

    df = pd.DataFrame(rows, columns=["StudyUID", "Timepoint", "T1w", "T1ce", "T2w", "FLAIR",
                                     "Lesion"])
    df.to_csv(csv_path, index=False)
    print(f"wrote {len(df)} rows -> {csv_path}")
    return csv_path



if __name__ == "__main__":
    
    pipeline_configs = {
        "yale": {
            "kwargs": {
                "data_path": "/home/data/FOMO300K/PT035_Yale_Brain_Mets_Longitudinal",
                "aligned_data_path": "/home/data/BRAIN_DIFF_S3/YALE/ALIGNED",
                "output_path": "/home/data/BRAIN_DIFF_S3/YALE/OUTPUT",
                "extract_modality_fn": extract_modality_yale,
            },
        },
        # The two BraTS sets are the only S3 sources with lesion masks: every
        # timepoint folder ships a "<name>-seg.nii.gz" (GLI 1621/1621,
        # MET 773/773), located by find_seg_brats_men and warped with the same
        # affine as each modality, nearest-neighbour.
        "brats-gli": {
            "kwargs": {
                "data_path": "/home/data/BraTS-GLI",
                "aligned_data_path": "/home/data/BRAIN_DIFF_S3/BraTS-GLI/ALIGNED",
                "output_path": "/home/data/BRAIN_DIFF_S3/BraTS-GLI/OUTPUT",
                "lesion_dir": "/home/data/BRAIN_DIFF_S3/BraTS-GLI_LESION",
                "extract_modality_fn": extract_modality_brats_men,
                "find_lesion_fn": find_seg_brats_men,
            },
        },
        # FIXED 2026-08-09: data_path was the Yale path, identical to "yale"
        # above, so S3's "BraTS-MET" was Yale aligned a second time under fresh
        # uuid4s -- 8,239/8,239 identical filenames, volumes correlating
        # 0.994-0.999. That duplicated ~19.6k pairs AND leaked the val split,
        # since patient_of() groups on the per-run uuid: 226 of 251 val subjects
        # (90.0%) also appeared in train. The real multi-timepoint BraTS-MET
        # (287 subjects / 773 timepoints) had never been processed.
        "brats-met": {
            "kwargs": {
                "data_path": "/home/data/BraTS-MET/multi-timepoint",
                "aligned_data_path": "/home/data/BRAIN_DIFF_S3/BraTS-MET/ALIGNED",
                "output_path": "/home/data/BRAIN_DIFF_S3/BraTS-MET/OUTPUT",
                "lesion_dir": "/home/data/BRAIN_DIFF_S3/BraTS-MET_LESION",
                "extract_modality_fn": extract_modality_brats_men,
                "find_lesion_fn": find_seg_brats_men,
            },
        },

         "oasis-2": {
            "kwargs": {
                "data_path": "/home/data/FOMO300K/PT029_OASIS2",
                "aligned_data_path": "/home/data/BRAIN_DIFF_S3/OASIS-2/ALIGNED",
                "output_path": "/home/data/BRAIN_DIFF_S3/OASIS-2/OUTPUT",
                "extract_modality_fn": extract_modality_yale,
            },
        },


    }

    ap = argparse.ArgumentParser(
        description="Align a longitudinal dataset to MNI. Resumable: "
                    "align_image_to_mni skips work already on disk.")
    ap.add_argument("--config", required=True, choices=sorted(pipeline_configs))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-lesions", action="store_true",
                    help="skip mask warping even if the config defines it")
    args = ap.parse_args()

    kwargs = dict(pipeline_configs[args.config]["kwargs"])
    if args.no_lesions:
        kwargs.pop("find_lesion_fn", None)
        kwargs.pop("lesion_dir", None)
    generate_longitudinal_positional_dataset(
        **kwargs, shard=args.shard, nshards=args.nshards)
