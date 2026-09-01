#!/usr/bin/env python3
"""
Output-tree layout for the BIND pipeline.

Kept separate from skullstrip.py so the CPU workers can locate a mask without
importing torch and the HD-BET ensemble, which module-level import would
otherwise pull into every one of ~24 worker processes.

    /home/data/BIND-BRAINDIFF/
        _masks/{site}/{session_uid}.nii.gz          native-space brain mask
        {site}/{session_uid}/{modality}.nii.gz      193x229x193 @ 1 mm MNI
        image.csv, main.csv                         manifests
"""

from pathlib import Path

OUTPUT_ROOT = Path("/home/data/BIND-BRAINDIFF")

MODALITIES = ["t1w", "t1ce", "t2w", "flair"]

# The geometry of dataset/ants_data/mni_reference.nii.gz, and therefore of
# BRAIN_DIFF_S2/S4. Asserted on every written volume.
EXPECTED_SHAPE = (193, 229, 193)


def mask_path(output_root, site, session_uid):
    """Cache location for one session's native-space brain mask."""
    return Path(output_root) / "_masks" / str(site) / f"{session_uid}.nii.gz"


def session_dir(output_root, site, session_uid):
    """Output directory holding one session's MNI-space modalities."""
    return Path(output_root) / str(site) / str(session_uid)
