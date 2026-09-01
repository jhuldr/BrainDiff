"""S4 dataloader that also hands the model the PRIOR REPORT.

Motivation, measured: prior report text alone predicts change-vs-stable at 0.693 AUC,
against 0.564 for the frozen imaging stack and 0.803 for the current report (the
ceiling). The model currently gets the prior IMAGE and has to re-derive from voxels
what the prior radiologist already wrote down -- while a real radiologist reads the
prior report. `report1` is already in the S4 CSVs and was simply unused.

Wraps dataloaders/dual.py. Nothing there is modified, so the existing S4 stage keeps
building.

JOIN SAFETY: MultiModalDiffDataset drops rows (empty caption/classification, missing
study, no modality present at both timepoints) and keeps no index, so positional
alignment against the CSV is wrong -- it would silently attach the wrong prior report
to every row after the first drop. `generated_report` is not unique either (40 dupes
in train). The item's `paths` DO encode both study UIDs, which are unique by
construction, so the join key is recovered from there.
"""
import os
import re

import pandas as pd
import torch
from torch.utils.data import DataLoader

from braindiff.data.dual import (MODALITIES, MultiModalDiffDataset, BalancedBatchSampler,
                              get_diff_caption_dataset)
from torch.utils.data.distributed import DistributedSampler


def uid_from_path(path):
    """/home/data/BRAIN_DIFF_S4/batch04/2225XWZ74V/t1w.nii.gz -> 2225XWZ74V"""
    return os.path.basename(os.path.dirname(str(path)))


class PriorReportDataset(MultiModalDiffDataset):
    """Adds `prior_report` (raw text) to every sample."""

    def __init__(self, csv_file, image_csv, **kw):
        super().__init__(csv_file=csv_file, image_csv=image_csv, **kw)
        df = pd.read_csv(csv_file)
        lut = {(str(a), str(b)): str(r) for a, b, r in
               zip(df["study_uid1"], df["study_uid2"], df["report1"])}

        self.prior_reports, missing = [], 0
        for it in self.items:
            ref = next((p for k, p in it["paths"].items() if k.startswith("ref_")), None)
            main = next((p for k, p in it["paths"].items() if k.startswith("main_")), None)
            key = (uid_from_path(ref), uid_from_path(main))
            rep = lut.get(key)
            if rep is None or rep == "nan":
                missing += 1
                rep = ""
            self.prior_reports.append(rep)
        if missing:
            print(f"[prior_report] {missing}/{len(self.items)} items had no prior report "
                  f"(empty string used)", flush=True)
        assert len(self.prior_reports) == len(self.items)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["prior_report"] = self.prior_reports[idx]
        sample["sample_idx"] = torch.tensor(idx)
        return sample


def collate_with_prior(batch, base_collate):
    """Strings cannot be default-collated into a tensor; keep them as a list."""
    priors = [b.pop("prior_report") for b in batch]
    out = base_collate(batch) if base_collate else torch.utils.data.default_collate(batch)
    out["prior_report"] = priors
    return out


def get_prior_report_dataloader(csv_file, image_csv, img_size, batch_size, num_workers,
                                tokenizer=None, max_caption_length=128, is_train=True,
                                distributed=False, content_weight=1.0):
    """Mirrors dual.get_diff_caption_dataloader, with the prior-report dataset."""
    ds = PriorReportDataset(
        csv_file=csv_file, image_csv=image_csv, img_size=img_size,
        tokenizer=tokenizer, max_caption_length=max_caption_length,
        is_train=is_train, content_weight=content_weight,
    )
    collate = lambda b: collate_with_prior(b, torch.utils.data.default_collate)

    if is_train:
        num_replicas = torch.distributed.get_world_size() if distributed else 1
        rank = torch.distributed.get_rank() if distributed else 0
        sampler = BalancedBatchSampler(
            classifications=[i["classification"] for i in ds.items],
            batch_size=batch_size, num_replicas=num_replicas, rank=rank)
        return DataLoader(ds, batch_sampler=sampler, num_workers=num_workers,
                          pin_memory=True, collate_fn=collate)

    sampler = DistributedSampler(ds, shuffle=False) if distributed else None
    return DataLoader(ds, batch_size=batch_size, shuffle=False, sampler=sampler,
                      num_workers=num_workers, pin_memory=True, collate_fn=collate)
