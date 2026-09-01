#!/usr/bin/env python3
"""
Output-tree layout for the OASIS-3 pipeline.

Kept separate from skullstrip.py for the same reason process_bind/paths.py is: the
CPU workers must be able to locate a mask without importing torch and the HD-BET
ensemble, which a module-level import would otherwise pull into every worker
process.

    /home/data/BRAIN_DIFF_S3/OASIS-3/
        _masks/{subject}_{session_label}.nii.gz     native-space T1w brain mask
        _masks_mni/{subject}_ses-{NN}.nii.gz        same mask on the T1w's MNI affine
        ALIGNED/{subject}_ses-{NN}_{mod}.nii.gz     193x229x193 @ 1 mm
        OUTPUT/sessions.csv                         which raw file was chosen
        OUTPUT/longitudinal.csv                     S3 image schema
        OUTPUT/pairs.csv                            S3 main schema
        OUTPUT/failures.csv

NOTE the modality capitalisation. S3's ALIGNED trees use `T1w`/`T2w`/`FLAIR`
(dataset3D/create_longitudinal.py MODALITY_COLUMN), while S2/S4/BIND write
lowercase `t1w`/`t2w`/`flair` per-study directories. This pipeline feeds S3, so it
follows S3.
"""

from pathlib import Path

RAW_ROOT = Path("/home/data/OASIS-3")
OUTPUT_ROOT = Path("/home/data/BRAIN_DIFF_S3/OASIS-3")
S3_ROOT = Path("/home/data/BRAIN_DIFF_S3")

# S3 spelling, not S2/S4 spelling. OASIS-3 ships no post-contrast series, so T1ce
# is absent by construction -- it stays in the CSV schema as an empty column.
MODALITIES = ["T1w", "T2w", "FLAIR"]
CSV_MODALITIES = ["T1w", "T1ce", "T2w", "FLAIR"]

MNI_REFERENCE = Path(
    "/path/to/code/BrainDiff/dataset/ants_data/mni_reference.nii.gz"
)

# The geometry of mni_reference.nii.gz, and therefore of every BRAIN_DIFF stage.
# Asserted on every written volume.
EXPECTED_SHAPE = (193, 229, 193)


def aligned_dir(output_root=OUTPUT_ROOT):
    return Path(output_root) / "ALIGNED"


def output_dir(output_root=OUTPUT_ROOT):
    return Path(output_root) / "OUTPUT"


def session_stem(subject_id, timepoint):
    """`sub-OAS30001_ses-03` -- the ordinal form used in every written filename.

    Ordinal rather than the `ses-dXXXX` day label because
    dataset3D/create_longitudinal.py::parse_timepoint and
    dataloaders/diff_pairs.py::patient_of both need `<patient>_<small int>`
    structure to recover the timepoint. The true day offset lives in a column.

    This form is also what dataloaders/diff_pairs.py::subject_of parses for the
    val split -- its `_SUBJECT_RE` is `^(BraTS-\\w+-\\d+|sub-[A-Za-z0-9]+)`, which
    takes `sub-OAS30001` off the front (verified). That matters because grouping
    on the uuid4 StudyUID is what leaked 90% of S3's val set into train; here the
    StudyUID is a uuid5 of the subject id, so the two split keys agree instead of
    one silently splitting a subject in half.
    """
    return f"{subject_id}_ses-{int(timepoint):02d}"


def aligned_path(subject_id, timepoint, modality, output_root=OUTPUT_ROOT):
    """One written MNI-space volume."""
    return aligned_dir(output_root) / f"{session_stem(subject_id, timepoint)}_{modality}.nii.gz"


def mask_path(subject_id, session_label, output_root=OUTPUT_ROOT):
    """Cache location for one session's native-space HD-BET mask.

    Keyed on the raw `ses-dXXXX` label rather than the ordinal, so the GPU phase
    does not depend on the ordinal numbering staying stable if sessions are added
    by a later download.
    """
    return Path(output_root) / "_masks" / f"{subject_id}_{session_label}.nii.gz"


def mask_mni_path(subject_id, timepoint, output_root=OUTPUT_ROOT):
    """The same mask carried into MNI on the T1w's affine."""
    return Path(output_root) / "_masks_mni" / f"{session_stem(subject_id, timepoint)}.nii.gz"
