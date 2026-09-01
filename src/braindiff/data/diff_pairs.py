import os
import random
import re
import time

import pandas as pd
import torch
from monai.data import PersistentDataset
from monai.transforms import (
    Compose, Rand3DElasticd, RandAffined, RandBiasFieldd, RandGaussianNoised,
)
from braindiff.data.neurovfm_transforms import (
    NeuroVFMGridd, NeuroVFMTokenize, N_TOKENS, TOKEN_DIM)
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# torch>=2.6 defaults torch.load to weights_only=True, which chokes on the
# plain MetaTensor/numpy objects PersistentDataset pickles into its cache
# (monai/data/dataset.py calls torch.load(hashfile) with no override). These
# cache files are entirely self-generated, so restore the pre-2.6 default.
_torch_load = torch.load
def _weights_only_default_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)
torch.load = _weights_only_default_false


# Names match multi_dual_dataloader exactly. S3 and S4 are the same two-timepoint
# problem, and every place they diverge without a reason is a place they can drift.
MODALITIES = ["T1w", "T1ce", "T2w", "FLAIR"]
TIMEPOINTS = {"ref": 1, "main": 2}


def patient_of(uid: str) -> str:
    """S3 UIDs are `<StudyUID>_<timepoint>`; the trailing index is the study.

WARNING -- not a stable patient identity. `StudyUID` is a uuid4 minted per subject folder
per build run, so the same real subject processed twice carries two uuids and reads as two
patients; grouping on it once put 226 of 251 val subjects (90.0%) into train as well. Use
`subject_of` for splitting; this is kept for callers that want the study prefix.
    """
    return re.sub(r"_\d+$", "", str(uid))


# Real subject id as it appears at the head of an aligned filename. Stable across
# rebuilds, unlike the uuid4 StudyUID.
_SUBJECT_RE = re.compile(r"^(BraTS-\w+-\d+|sub-[A-Za-z0-9]+)")


def subject_of(uid, uid_to_path: dict) -> str:
    """Stable subject identity, parsed from the volume path rather than the UID.

Falls back to `patient_of(uid)` when no path is known -- a missing row should become its own
group rather than silently merging with everything else.
    """
    path = uid_to_path.get(uid)
    if not isinstance(path, str) or not path:
        return patient_of(uid)
    m = _SUBJECT_RE.match(os.path.basename(path))
    return m.group(1) if m else patient_of(uid)


def _uid_to_path(image_csv) -> dict:
    """UID -> first present modality path, for subject_of."""
    if image_csv is None:
        return {}
    img = pd.read_csv(image_csv, low_memory=False)
    if "UID" not in img.columns:
        return {}
    out = {}
    for _, r in img.iterrows():
        for c in MODALITIES:
            v = r.get(c)
            if isinstance(v, str) and v:
                out[r["UID"]] = v
                break
    return out


def split_pairs(csv_file, val_fraction=0.1, seed=10, image_csv=None):
    """Subject-grouped split.

All S3 pairs are same-patient and are consecutive timepoints, so a random row split would
put the same patient, often the same study, on both sides.

Groups on the real subject id from the path rather than the uuid4 StudyUID, which is per
build run and leaked 90.0% of val into train. Pass `image_csv` to enable it; without it this
degrades to uuid grouping and warns.
    """
    df = pd.read_csv(csv_file)
    u2p = _uid_to_path(image_csv)
    if not u2p:
        print("[split_pairs] WARNING: no image_csv -- grouping on the uuid4 StudyUID, "
              "which does not identify a subject across build runs. Pass image_csv.")
    key = lambda u: subject_of(u, u2p)
    patients = sorted({key(u) for u in df.UID_1} | {key(u) for u in df.UID_2})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_val = max(1, int(round(val_fraction * len(patients))))
    val_set = set(patients[:n_val])

    is_val = df.UID_1.map(lambda u: key(u) in val_set)
    # A pair whose two studies straddle the split would leak; none should exist,
    # but assert rather than assume since the whole point is no shared patients.
    straddle = is_val != df.UID_2.map(lambda u: key(u) in val_set)
    if straddle.any():
        raise ValueError(f"{int(straddle.sum())} pairs straddle the patient split")
    train, val = df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)

    # The check the uuid grouping silently failed. Cheap, and it is the only thing
    # standing between a val number and a repeat of the 90% leak.
    tr_s = {key(u) for u in train.UID_1} | {key(u) for u in train.UID_2}
    va_s = {key(u) for u in val.UID_1} | {key(u) for u in val.UID_2}
    shared = tr_s & va_s
    if shared:
        raise ValueError(
            f"{len(shared)} subjects appear in BOTH train and val "
            f"(e.g. {sorted(shared)[:3]}). If image_csv was omitted this is the "
            f"uuid-grouping bug; pass it.")
    return train, val


class DiffPairDataset(Dataset):
    """Self-supervised longitudinal pair dataset for S3 diff-module pretraining.

No report text. Emits the same tensors as the S4 dual loader -- tokens [M, N, 1024] fp16,
coords [M, N, 3] int16, present [M] bool per timepoint -- so the delta module is pretrained
on the features it will consume at S4.

With probability `dup_fraction` (train only), an item is instead an augmented-duplicate
pair: one timepoint loaded once, given the same spatial warp and two independent intensity
augmentations, so its true anatomical delta is zero. Warping the two views independently
would instead teach that a geometric shift means no change.
    """

    def __init__(self, df, image_csv, is_train=True, dup_fraction=0.15,
                 modality_dropout=0.5,
                 cache_dir="/home/data/BRAIN_DIFF_S3/tmp_nvfm"):
        self.is_train = is_train
        self.dup_fraction = dup_fraction if is_train else 0.0
        # On duplicate (zero-change) rows, with this probability drop a random proper
        # subset of modalities from ONE view. A modality present at one timepoint and
        # absent at the other is pure acquisition nuisance, not change; teaching the
        # gate that on the zero-change control keeps it from firing on missing-modality
        # asymmetry. Train-only; never applied to real pairs.
        self.modality_dropout = modality_dropout if is_train else 0.0

        # Per-timepoint key lists, e.g. "ref_T1w", "main_FLAIR".
        self.keys = {tp: [f"{tp}_{mod}" for mod in MODALITIES] for tp in TIMEPOINTS}
        self.all_keys = self.keys["ref"] + self.keys["main"]

        # === BUILD ITEMS (drop incomplete studies) ===
        image_df = pd.read_csv(image_csv).set_index("UID")
        self.items = []
        dropped = 0

        for _, row in df.iterrows():
            paths, present = {}, {tp: [] for tp in TIMEPOINTS}
            for tp, t in TIMEPOINTS.items():
                image_row = image_df.loc[row[f"UID_{t}"]]
                for mod in MODALITIES:
                    path = image_row.get(mod)
                    if pd.isna(path) or not os.path.exists(path):
                        present[tp].append(False)
                    else:
                        present[tp].append(True)
                        paths[f"{tp}_{mod}"] = path

            if not (any(present["ref"]) and any(present["main"])):
                dropped += 1
                continue

            self.items.append({
                "paths": paths,
                "present_ref": present["ref"],
                "present_main": present["main"],
            })
        print(f"DiffPairDataset: kept {len(self.items)}, dropped {dropped}")

        # === PERSISTENT CACHE for the deterministic per-volume transform ===
        # Each volume is re-read ~3.2x per epoch (208k loads over 65k unique
        # volumes), and 92 of the ~100 ms per volume is the nibabel read. S1/S2/S4
        # have no comparable reuse, which is why only this stage caches.
        #
        # The GRID is cached, not the tokens: augmentation needs a spatial volume,
        # and tokenizing is a 0.8 ms reshape afterwards. No intensity normalization
        # here -- S2/S4 use the fixed affine inside NeuroVFMTokenize, and a
        # per-volume rescale would mix anatomy with scaling in the very difference
        # this stage learns.
        unique_paths = sorted({p for item in self.items for p in item["paths"].values()})
        self.path_to_idx = {p: i for i, p in enumerate(unique_paths)}
        self.vol_cache = PersistentDataset(
            data=[{"img": p} for p in unique_paths],
            transform=Compose([NeuroVFMGridd(keys="img", allow_missing_keys=False)]),
            cache_dir=cache_dir,
        )

        # Spatial shared across both timepoints, intensity independent per
        # timepoint -- identical to multi_dual_dataloader. Sharing the warp keeps
        # ref and main in correspondence (the delta is token-wise); sharing the bias
        # field instead would leave a matching intensity signature the model could
        # pair on without reading anatomy.
        self.spatial_aug = self._make_spatial_augment(self.all_keys) if is_train else None
        self.intensity_aug = {
            tp: self._make_intensity_augment(self.keys[tp]) for tp in TIMEPOINTS
        } if is_train else None
        # Applied to the `main` view of a duplicate ONLY -- see the method docstring.
        self.reposition_aug = (self._make_reposition_augment(self.keys["main"])
                               if is_train else None)

        self.to_tokens = NeuroVFMTokenize(keys=self.all_keys, allow_missing_keys=True)

    def _make_spatial_augment(self, keys):
        """Geometry, applied identically to every key in one call so ref and main
        stay aligned. Same magnitudes as S4: deformation is kept mild because the
        lesions of interest have a median size of 8 mm."""
        return Compose([
            RandAffined(keys=keys, prob=0.3, rotate_range=(0.05, 0.05, 0.05),
                        scale_range=(0.05, 0.05, 0.05), mode="bilinear",
                        allow_missing_keys=True),
            Rand3DElasticd(keys=keys, sigma_range=(5, 8), magnitude_range=(50, 100),
                           prob=0.15, mode="bilinear", allow_missing_keys=True),
        ])

    def _make_reposition_augment(self, keys):
        """A minor rigid perturbation, applied to one view of a duplicate only.

Without it both views share a single `spatial_aug` call and differ only in intensity, so the
gate learns to ignore intensity nuisance and nothing else -- while a stable lesion differs
from its prior mainly by residual misregistration. Measured consequence: the gate scored
CHANGE 0.1672 vs STABLE 0.1666.

Rigid only, and an order of magnitude below `_make_spatial_augment`: 0.02 rad (~1.1 deg)
against 0.05, and 1.5 voxels of translation. It must not move a lesion across a 16 mm token
boundary. prob=1.0, since it is the signal rather than an augmentation.
        """
        return Compose([
            RandAffined(keys=keys, prob=1.0,
                        rotate_range=(0.02, 0.02, 0.02),
                        translate_range=(1.5, 1.5, 1.5),
                        mode="bilinear", allow_missing_keys=True),
        ])

    def _make_intensity_augment(self, keys):
        """Appearance, randomized separately per timepoint."""
        return Compose([
            RandBiasFieldd(keys=keys, prob=0.3, allow_missing_keys=True),
            RandGaussianNoised(keys=keys, prob=0.2, allow_missing_keys=True),
        ])

    def __len__(self):
        return len(self.items)

    def _lookup(self, key_to_path):
        """Fetch cached grids for a {key: path} mapping."""
        return {key: self.vol_cache[self.path_to_idx[path]]["img"]
                for key, path in key_to_path.items()}

    def warm_cache(self):
        """Force every volume through the cache once, sequentially in this process.

PersistentDataset's first write is not safe under concurrent writers (DataLoader workers,
DDP ranks), so this must complete from a single process before the dataset is handed to a
multi-worker or multi-rank DataLoader.
        """
        for i in tqdm(range(len(self.vol_cache)), desc="Warming volume cache"):
            _ = self.vol_cache[i]

    def _stack(self, data, tp):
        """Present modalities for one timepoint -> [M,N,1024] fp16, [M,N,3] int16,
        [M] bool. Absent modalities stay all-zero and are masked out of the
        connector's attention rather than contributing a block of zeros."""
        tokens = torch.zeros(len(MODALITIES), N_TOKENS, TOKEN_DIM, dtype=torch.float16)
        coords = torch.zeros(len(MODALITIES), N_TOKENS, 3, dtype=torch.int16)
        present = torch.zeros(len(MODALITIES), dtype=torch.bool)
        for i, key in enumerate(self.keys[tp]):
            entry = data.get(key)
            # An off-grid volume is dropped, not stacked at the wrong shape.
            if entry is None or entry["tokens"].shape[0] != N_TOKENS:
                continue
            tokens[i], coords[i], present[i] = entry["tokens"], entry["coords"], True
        return tokens, coords, present

    def __getitem__(self, idx):
        item = self.items[idx]
        is_dup = self.is_train and random.random() < self.dup_fraction

        if is_dup:
            # Augmented-duplicate: take the timepoint with more present modalities
            # and expose it under BOTH views' keys, so one spatial_aug call warps
            # them identically. Only the intensity augmentation differs.
            src = ("ref" if sum(item["present_ref"]) >= sum(item["present_main"])
                   else "main")
            present_src = item[f"present_{src}"]
            paths = {f"{view}_{mod}": item["paths"][f"{src}_{mod}"]
                     for view in TIMEPOINTS for mod in MODALITIES
                     if f"{src}_{mod}" in item["paths"]}
            data = self._lookup(paths)
            data = self.spatial_aug(data)
            # Break the geometric identity of the two views: without this a dup is
            # pixel-identical across views and the nuisance control only ever sees
            # intensity jitter.
            data = self.reposition_aug(data)
            for tp in TIMEPOINTS:
                data = self.intensity_aug[tp](data)
            data = self.to_tokens(data)

            # Modality dropout: on the zero-change control, drop a random proper subset
            # of modalities from one view so the pair carries a present/absent asymmetry
            # the gate must learn to ignore (missing modality != change).
            per_view = {tp: list(present_src) for tp in TIMEPOINTS}
            present_idx = [i for i, on in enumerate(present_src) if on]
            if self.modality_dropout > 0 and len(present_idx) >= 2 \
                    and random.random() < self.modality_dropout:
                view = random.choice(list(TIMEPOINTS))
                k = random.randint(1, len(present_idx) - 1)   # keep >=1 in this view
                for i in random.sample(present_idx, k):
                    per_view[view][i] = False

            sample = {"is_dup": torch.tensor(True)}
            for tp in TIMEPOINTS:
                t, c, p = self._stack(data, tp)
                # Presence follows the SOURCE timepoint, minus any modality-dropout.
                p = p & torch.tensor(per_view[tp], dtype=torch.bool)
                sample[f"tokens_{tp}"], sample[f"coords_{tp}"] = t, c
                sample[f"present_{tp}"] = p
            return sample

        # Real pair: warp both timepoints together, vary appearance independently.
        data = self._lookup(item["paths"])
        if self.is_train:
            data = self.spatial_aug(data)
            for tp in TIMEPOINTS:
                data = self.intensity_aug[tp](data)
        data = self.to_tokens(data)

        sample = {"is_dup": torch.tensor(False)}
        for tp in TIMEPOINTS:
            t, c, p = self._stack(data, tp)
            sample[f"tokens_{tp}"], sample[f"coords_{tp}"] = t, c
            sample[f"present_{tp}"] = p
        return sample


def get_diff_pair_dataloader(
    df,
    image_csv,
    batch_size,
    num_workers,
    is_train=True,
    dup_fraction=0.15,
    modality_dropout=0.5,
    distributed: bool = False,
    cache_dir="/home/data/BRAIN_DIFF_S3/tmp_nvfm",
    split_name="train",
):
    """Returns the DataLoader. Under DDP a DistributedSampler shards the data;
    access it via loader.sampler for set_epoch()."""

    ds = DiffPairDataset(
        df=df,
        image_csv=image_csv,
        is_train=is_train,
        dup_fraction=dup_fraction,
        modality_dropout=modality_dropout,
        cache_dir=cache_dir,
    )

    # Warm the persistent cache from a single process before any DataLoader
    # worker or other DDP rank can race to write the same missing entry. This
    # must finish *before* the process group is even created: warming can take
    # a long time, and waiting for it inside a dist.barrier() means the other
    # ranks are blocked on NCCL's communicator rendezvous, which has its own
    # (much shorter) timeout and aborts long before a slow warm-up finishes.
    # So call this before ddp_setup()/init_process_group(), and synchronize
    # with a plain filesystem marker instead of a collective.
    #
    # The marker is per split: train and val hold disjoint volumes, so a single
    # shared marker would let the second split start on a cold cache.
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    warm_marker = os.path.join(cache_dir, f".warm_done_{split_name}")
    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(warm_marker):
            print(f"Warming persistent cache ({len(ds.vol_cache)} volumes, {split_name})...")
            ds.warm_cache()
            open(warm_marker, "w").close()
    elif world_size > 1:
        while not os.path.exists(warm_marker):
            time.sleep(5)

    # Shard every split, not just train: with val unsharded each rank would
    # iterate the whole set and the all-reduce would average identical numbers.
    sampler = (DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                  shuffle=is_train)
               if distributed else None)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
