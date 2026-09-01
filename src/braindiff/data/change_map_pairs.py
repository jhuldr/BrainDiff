"""Supervised variant of the S3 pair loader: adds change labels to each sample.

Subclasses `DiffPairDataset` rather than reimplementing it, so volume loading, caching and
tokenisation stay as the existing S3 stage does them.

Two base-class behaviours are wrong for a densely-labelled row and are suppressed for those
rows only:

1. Spatial augmentation. `RandAffined` + `Rand3DElasticd` warp the volume, but the
   per-token occupancy target was computed on the unwarped mask, so a rotation of a few
   degrees moves an 8 mm lesion across a 16 mm token boundary and the target no longer
   describes the input. Intensity augmentation is kept: it moves no voxels.
2. Duplicate synthesis. `dup_fraction` replaces the main timepoint with the ref one to
   manufacture a zero-change pair, but the stored target still describes the real pair.

Unlabelled rows keep the full augmentation pipeline.
"""
import os

import numpy as np
import pandas as pd
import torch

from braindiff.data.diff_pairs import MODALITIES, DiffPairDataset, get_diff_pair_dataloader

TOKEN_DIR = "/home/data/BRAIN_DIFF_S3/LESION_TOKENS"
GRID = (12, 14, 12)
N_TOKENS = GRID[0] * GRID[1] * GRID[2]


def _identity(data):
    return data


def load_occupancy(prefix):
    """[M, 2016] float32 per-token tumour fraction, one row per modality.

    Modalities without a cached map (the mask is warped per modality, and a
    modality can be absent) are left at zero and flagged in the returned mask.
    """
    occ = np.zeros((len(MODALITIES), N_TOKENS), dtype=np.float32)
    have = np.zeros(len(MODALITIES), dtype=bool)
    if not isinstance(prefix, str):
        return occ, have
    for i, mod in enumerate(MODALITIES):
        path = os.path.join(TOKEN_DIR, f"{prefix}_{mod}.npy")
        if os.path.exists(path):
            occ[i] = np.load(path).reshape(-1)
            have[i] = True
    return occ, have


class ChangeMapPairDataset(DiffPairDataset):
    """`meta` must be row-aligned with `df`; build_s3sup_pairs.py pre-filters the
    table so the base constructor drops nothing, and __init__ asserts that held."""

    def __init__(self, df, image_csv, **kw):
        super().__init__(df=df, image_csv=image_csv, **kw)
        if len(self.items) != len(df):
            raise SystemExit(
                f"DiffPairDataset dropped {len(df) - len(self.items)} rows; labels would "
                f"be misaligned against samples. Re-run process_mrrate.build_s3sup_pairs, "
                f"which pre-filters so this cannot happen.")
        self.meta = df.reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        dense = bool(row["has_dense"])

        if dense:
            # Save/restore rather than a constructor flag: the base __getitem__ reads
            # these off self and takes no arguments. Safe because DataLoader workers
            # are separate processes and __getitem__ runs sequentially within one.
            aug, dup = self.spatial_aug, self.dup_fraction
            self.spatial_aug, self.dup_fraction = _identity, 0.0
            try:
                sample = super().__getitem__(idx)
            finally:
                self.spatial_aug, self.dup_fraction = aug, dup
        else:
            sample = super().__getitem__(idx)

        occ_ref, have_ref = load_occupancy(row["prefix_1"])
        occ_main, have_main = load_occupancy(row["prefix_2"])
        sample["occ_ref"] = torch.from_numpy(occ_ref)
        sample["occ_main"] = torch.from_numpy(occ_main)
        # A dense target is only valid where BOTH timepoints have that modality's map.
        sample["occ_valid"] = torch.from_numpy(have_ref & have_main)
        sample["has_dense"] = torch.tensor(dense)
        # A duplicate is ONE scan augmented twice, so its true change is zero -- but
        # the row's label describes the REAL pair it was built from. Attaching it
        # unconditionally (as this did) mislabels ~15% of has_global rows as changed
        # when nothing changed, and since dup suppression above covers only the
        # has_dense rows, nothing else caught it. Drop the global label on dups
        # rather than relabelling to Stable: a dup is not a stable pair, it is the
        # same image twice, and feeding it as Stable would teach the change head
        # that "identical" is what Stable looks like.
        is_dup = bool(sample.get("is_dup", False))
        sample["has_global"] = torch.tensor(bool(row["has_global"]) and not is_dup)
        sample["change_label"] = torch.tensor(int(row["change_label"]))
        sample["sample_idx"] = torch.tensor(idx)
        return sample


def get_change_map_pair_dataloader(df, image_csv, batch_size, num_workers,
                                        is_train=True, dup_fraction=0.15,
                                        modality_dropout=0.5,
                                        distributed=False,
                                        cache_dir="/home/data/BRAIN_DIFF_S3/tmp_nvfm",
                                        split_name="train"):
    """Same contract as get_diff_pair_dataloader, with the supervised dataset.

    Reuses the base helper for cache warming and sampler construction by swapping
    the dataset class in, so the warm-marker and DDP sharding logic stay in one place.
    """
    orig = get_diff_pair_dataloader.__globals__["DiffPairDataset"]
    get_diff_pair_dataloader.__globals__["DiffPairDataset"] = \
        lambda **kw: ChangeMapPairDataset(**kw)
    try:
        return get_diff_pair_dataloader(
            df=df, image_csv=image_csv, batch_size=batch_size,
            num_workers=num_workers, is_train=is_train, dup_fraction=dup_fraction,
            modality_dropout=modality_dropout,
            distributed=distributed, cache_dir=cache_dir, split_name=split_name)
    finally:
        get_diff_pair_dataloader.__globals__["DiffPairDataset"] = orig
