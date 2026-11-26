"""Stage 1-D and Stage 2 datasets, samplers and collation.

Tables under $BRAINDIFF_DATA (default ./data; built by braindiff/dataset/):
  stage1/pairs.parquet, stage1/entity_text.parquet, stage1/tasks.parquet, splits/bind_patient_split.csv   Stage 1-D
  stage2/pairs.parquet                                                                                    Stage 2
$BRAINDIFF_GRID_CACHE (default off) reads a MONAI PersistentDataset grid cache for volumes already in it.

Keys are tuples built by the sampler, so every random choice is fixed outside the workers: ("pair", i, rev, seed) and
("task", j, 0, seed). Augmentation (train only): one spatial warp shared by every series; gamma / blur / low resolution
shared by both timepoints; bias field + noise per timepoint. No flips (laterality). Box tasks get no spatial warp.

Stage 2 targets of >= max_target_tokens tokens are dropped, not truncated (the Impression is last).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from braindiff.data.transforms import NeuroVFMGridd, NeuroVFMTokenize
from braindiff.models.prompts import status_line, status_target

MODS = ["T1w", "T1ce", "T2w", "FLAIR"]
TASKS = ("aref", "gcap", "caref", "pcls")


def data_root():
    return os.environ.get("BRAINDIFF_DATA", "data")


def _stacked(tf, data, ks, seed):
    """One spatial transform applied to every key at once, stacked as channels: the same random draw for every key."""
    x = tf.set_random_state(seed)({"x": torch.cat([torch.as_tensor(data[k]) for k in ks])})["x"]
    for j, k in enumerate(ks):
        data[k] = x[j:j + 1]


def make_spatial_augment(keys):
    """Geometry, identical for every key; mild (lesions of interest are median 8 mm)."""
    from monai.transforms import Compose, Rand3DElasticd, RandAffined
    return Compose([
        RandAffined(keys=keys, prob=0.3, rotate_range=(0.05, 0.05, 0.05), scale_range=(0.05, 0.05, 0.05),
                    mode="bilinear", allow_missing_keys=True),
        Rand3DElasticd(keys=keys, sigma_range=(5, 8), magnitude_range=(50, 100), prob=0.15, mode="bilinear",
                       allow_missing_keys=True),
    ])


def make_noise_augment(keys):
    """Acquisition noise, independent per timepoint: bias field + Gaussian noise."""
    from monai.transforms import Compose, RandBiasFieldd, RandGaussianNoised
    return Compose([RandBiasFieldd(keys=keys, prob=0.3, allow_missing_keys=True),
                    RandGaussianNoised(keys=keys, prob=0.2, allow_missing_keys=True)])


def _shared_lowres(**kw):
    """RandSimulateLowResolutiond draws a new zoom per key and ignores set_random_state; this draws one seeded zoom
    shared by every key."""
    from monai.transforms import RandSimulateLowResolutiond

    class SharedLowResd(RandSimulateLowResolutiond):
        def set_random_state(self, seed=None, state=None):
            super().set_random_state(seed, state)
            self.sim_lowres_tfm.set_random_state(seed, state)
            return self

        def __call__(self, data):
            d = dict(data)
            self.randomize(None)
            if not self._do_transform:
                return d
            self.sim_lowres_tfm.randomize()
            for key in self.key_iterator(d):
                d[key] = self.sim_lowres_tfm(d[key], randomize=False)
            return d

    return SharedLowResd(**kw)


def make_appearance_augment(keys):
    """Gamma, blur, simulated low resolution: ONE draw shared by every timepoint of the item."""
    from monai.transforms import Compose, RandAdjustContrastd, RandGaussianSmoothd
    return Compose([
        RandAdjustContrastd(keys=keys, gamma=(0.7, 1.5), prob=0.2, allow_missing_keys=True),
        RandGaussianSmoothd(keys=keys, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0), prob=0.2,
                            allow_missing_keys=True),
        _shared_lowres(keys=keys, zoom_range=(0.5, 1.0), prob=0.25, allow_missing_keys=True),
    ])


class VolumeDataset(Dataset):
    """Volume loading, augmentation and tokenisation shared by both stages."""

    def init_volumes(self, augment):
        self.augment = augment
        self.grid = NeuroVFMGridd(keys=["k"])
        self.tok = NeuroVFMTokenize(keys=["k"])
        self.grid_cache = os.environ.get("BRAINDIFF_GRID_CACHE", "")

    def load_grid(self, path):
        """[1, 48, 224, 192] grid, from the PersistentDataset cache when present, else decoded here."""
        if self.grid_cache:
            from monai.data.utils import pickle_hashing
            f = os.path.join(self.grid_cache, pickle_hashing({"img": path}).decode() + ".pt")
            if os.path.exists(f):
                return torch.as_tensor(torch.load(f, weights_only=False)["img"])
        return self.grid({"k": path})["k"]

    def _load(self, paths_by_tp, seed, spatial=True):
        """{tp: [path per series]} -> {tp: tokens [S, 2016, 1024] fp16}."""
        keys = {tp: [f"{tp}_{i}" for i in range(len(ps))] for tp, ps in paths_by_tp.items()}
        data = {k: self.load_grid(p) for tp in keys for k, p in zip(keys[tp], paths_by_tp[tp])}
        if self.augment:
            allk = [k for ks in keys.values() for k in ks]
            if spatial:
                _stacked(make_spatial_augment(["x"]), data, allk, seed)
            data = make_appearance_augment(allk).set_random_state(seed + 2)(data)       # shared draw
            for j, tp in enumerate(keys):
                data = make_noise_augment(keys[tp]).set_random_state(seed + 3 + j)(data)
        return {tp: torch.stack([self.tok({"k": data[k]})["k"]["tokens"] for k in ks]) for tp, ks in keys.items()}

    def __len__(self):
        return len(self.pairs)


def pair_split(pairs, root=None):
    """pairs.parquet rows -> split name ('train' / 'val' / 'test' / 'external_test'); MR-RATE = 'train'."""
    s = pd.read_csv(f"{root or data_root()}/splits/bind_patient_split.csv").set_index("patient").split
    out = pd.Series("train", index=pairs.index)
    b = pairs.source == "BIND"
    out[b] = pairs.patient[b].map(s)
    assert out[b].notna().all(), f"{out[b].isna().sum()} BIND pairs without a split"
    return out


class EntTextDataset(VolumeDataset):
    """Stage 1-D: report pairs with their status target, plus single-study task rows (`tasks`)."""

    def __init__(self, split="train", augment=True, tasks=(), root=None):
        root = root or data_root()
        p = pd.read_parquet(f"{root}/stage1/pairs.parquet")
        p = p[p.has_lent & (p.n_common > 0) & p.source.isin(["MR-RATE", "BIND"])].copy()
        p["pair_split"] = pair_split(p, root)
        e = pd.read_parquet(f"{root}/stage1/entity_text.parquet")
        p["site"] = p.site.where(p.source != "MR-RATE", "MR-RATE")
        p = p.merge(e, on=["source", "site", "study1", "study2"], how="left", validate="1:1")
        assert p.ent_text.notna().all(), f"{p.ent_text.isna().sum()} has_lent pairs without a target"
        self.n_not_ok = int((~p.ent_ok & (p.pair_split == split)).sum())
        p = p[(p.pair_split == split) & p.ent_ok].reset_index(drop=True)
        st = [set(x.split(",")) - {""} for x in p.text_statuses]
        p["target"] = [status_target(s) for s in st]
        p["any_change"] = [bool(s) for s in st]
        self.pairs = p
        assert set(tasks) <= set(TASKS), tasks
        t = pd.read_parquet(f"{root}/stage1/tasks.parquet") if tasks else None
        self.tasks = t[(t.split == split) & t.task.isin(tasks)].reset_index(drop=True) if tasks else None
        self.init_volumes(augment)

    def task_draw(self, rng, n):
        """-> n row indices of self.tasks for one epoch (half boxes, half of those with a lesion; half pcls)."""
        t = self.tasks
        box = (t.task != "pcls").values
        les = box & (t.with_lesion.fillna(0).values > 0)
        pools = [np.where(les)[0], np.where(box & ~les)[0], np.where(~box)[0]]
        share = np.array([.25, .25, .5]) * np.array([len(p) > 0 for p in pools])
        k = rng.multinomial(n, share / share.sum())
        return rng.permutation(np.concatenate([rng.choice(p, m) for p, m in zip(pools, k) if m]))

    def __getitem__(self, key):
        kind, i, _, seed = key
        if kind == "task":      # one study, current slot only, the row's own prompt
            r = self.tasks.iloc[i]
            mods = [m for m in MODS if isinstance(r[m], str)]
            v = self._load({"b": [r[m] for m in mods]}, seed, spatial=r.task == "pcls")
            return dict(mods=torch.tensor([MODS.index(m) for m in mods]), ta=v["b"][:0], tb=v["b"], text=r.target,
                        prompt=r.prompt, task=r.task, lesion=bool(r.with_lesion > 0), pair_id=-1, patient=r.subject)
        r = self.pairs.iloc[i]
        mods = [m for m in MODS if isinstance(r[f"{m}_1"], str) and isinstance(r[f"{m}_2"], str)]
        v = self._load({"a": [r[f"{m}_1"] for m in mods], "b": [r[f"{m}_2"] for m in mods]}, seed)
        return dict(mods=torch.tensor([MODS.index(m) for m in mods]), ta=v["a"], tb=v["b"], text=r.target,
                    pair_id=int(r.pair_id), patient=r.patient)


class Stage2Dataset(VolumeDataset):
    """Stage 2: report pairs with scans, the edited prior report and the status-first caption."""

    def __init__(self, split="train", augment=True, sources=("BIND", "MR-RATE"), max_target_tokens=None, tokenizer=None,
                 status_first=False, root=None):
        p = pd.read_parquet(f"{root or data_root()}/stage2/pairs.parquet")
        p = p[(p.split == split) & p.source.isin(sources)].reset_index(drop=True)
        p["caption"] = (p.apply(status_line, axis=1) + "\n" + p.target) if status_first else p.target
        self.n_long = 0
        if max_target_tokens:
            n = [len(x) for x in tokenizer(p.caption.tolist(), add_special_tokens=False).input_ids]
            keep = pd.Series(n) < max_target_tokens          # + <|im_end|>
            self.n_long = int((~keep).sum())
            p = p[keep.values].reset_index(drop=True)
        self.pairs, self.cond = p, "correct"
        self.init_volumes(augment)

    def set_condition(self, cond):
        """Image condition of the direction tests (eval/knows_says.py): correct / same_cur / same_prior / swap. Keeps
        pairs with exactly one of WORSENED / IMPROVED, or exactly one of NEW / RESOLVED."""
        p = self.pairs
        self.pairs = p[(p.st_WORSENED != p.st_IMPROVED) | (p.st_NEW != p.st_RESOLVED)].reset_index(drop=True)
        self.cond = cond
        return self

    def __getitem__(self, key):
        _, i, _, seed = key
        r = self.pairs.iloc[i]
        mods = [m for m in MODS if isinstance(r[f"{m}_1"], str) and isinstance(r[f"{m}_2"], str)]
        v = self._load({"a": [r[f"{m}_1"] for m in mods], "b": [r[f"{m}_2"] for m in mods]}, seed)
        ta, tb = {"same_cur": ("b", "b"), "same_prior": ("a", "a"), "swap": ("b", "a")}.get(self.cond, ("a", "b"))
        return dict(mods=torch.tensor([MODS.index(m) for m in mods]), ta=v[ta], tb=v[tb], text=r.caption,
                    prior=r.prior_edit, pair_id=int(r.pair_id), patient=r.patient)


def collate(items):
    """item / mods index the current-slot series (tb), item_a / mods_a the prior slot (ta; none for a task item)."""
    extra = {k: [it.get(k) for it in items] for k in ("prompt", "task", "lesion", "prior") if k in items[0]}
    return dict(n_items=len(items), **extra,
                item=torch.cat([torch.full((len(it["mods"]),), j) for j, it in enumerate(items)]),
                mods=torch.cat([it["mods"] for it in items]),
                item_a=torch.cat([torch.full((len(it["ta"]),), j) for j, it in enumerate(items)]),
                mods_a=torch.cat([it["mods"][:len(it["ta"])] for it in items]),
                ta=torch.cat([it["ta"] for it in items]), tb=torch.cat([it["tb"] for it in items]),
                text=[it["text"] for it in items], pair_id=[it["pair_id"] for it in items],
                patient=[it["patient"] for it in items])


def sub_batch(b, ix):
    """Items ix (contiguous) of a collated batch; series are contiguous per item."""
    lo, hi = ix[0], ix[-1] + 1
    m = (b["item"] >= lo) & (b["item"] < hi)
    ma = (b["item_a"] >= lo) & (b["item_a"] < hi)
    extra = {k: b[k][lo:hi] for k in ("prompt", "task", "lesion", "prior") if k in b}
    return dict(n_items=hi - lo, item=b["item"][m] - lo, mods=b["mods"][m], item_a=b["item_a"][ma] - lo,
                mods_a=b["mods_a"][ma], ta=b["ta"][ma], tb=b["tb"][m], text=b["text"][lo:hi],
                pair_id=b["pair_id"][lo:hi], patient=b["patient"][lo:hi], **extra)


class PairSampler(Sampler):
    """Every pair once per epoch as GLOBAL batches; rank r takes slice r::world. shuffle=False: fixed order, seed 0."""

    def __init__(self, n, batch, world_size=1, rank=0, seed=0, shuffle=True, drop_last=True):
        assert batch % world_size == 0, (batch, world_size)
        self.n, self.batch, self.world, self.rank, self.seed = n, batch, world_size, rank, seed
        self.shuffle, self.drop_last, self.epoch = shuffle, drop_last, 0

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        order = rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        if self.shuffle:
            rng.random(self.n)                 # unused reversal draw: keeps the seed stream
            seeds = rng.integers(2 ** 31, size=self.n)
        else:
            seeds = np.zeros(self.n, int)
        stop = self.n - self.batch + 1 if self.drop_last else self.n
        for lo in range(0, max(stop, 0), self.batch):
            g = [("pair", int(i), False, int(seeds[i])) for i in order[lo:lo + self.batch]]
            yield g[self.rank::self.world]

    def __len__(self):
        return self.n // self.batch if self.drop_last else -(-self.n // self.batch)


class MixedBatchSampler(Sampler):
    """Stage 1-D epoch = every pair once + round(task_ratio * n_pairs) task items, as homogeneous GLOBAL batches;
    rank r takes the contiguous slice r of each batch."""

    def __init__(self, ds, batch, world_size=1, rank=0, seed=0, task_ratio=1.0):
        assert batch % world_size == 0, f"global batch {batch} not divisible by world_size {world_size}"
        self.ds, self.batch, self.world, self.rank, self.seed = ds, batch, world_size, rank, seed
        self.task_ratio, self.epoch = task_ratio, 0

    def set_epoch(self, e):
        self.epoch = e

    def _batches(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        npairs = len(self.ds.pairs)
        rng.random(npairs)                     # unused reversal draw: keeps the seed stream
        streams = [[("pair", int(i), False) for i in rng.permutation(npairs)]]
        if self.task_ratio:
            streams.append([("task", int(i), 0) for i in self.ds.task_draw(rng, int(round(self.task_ratio * npairs)))])
        out = []
        for keys in streams:
            for lo in range(0, len(keys) - self.batch + 1, self.batch):        # drop_last: equal shards on every rank
                out.append([k + (int(rng.integers(2 ** 31)),) for k in keys[lo:lo + self.batch]])
        return [out[j] for j in rng.permutation(len(out))]

    def __iter__(self):
        for b in self._batches():
            n = len(b) // self.world
            yield b[self.rank * n:(self.rank + 1) * n]

    def __len__(self):
        npairs = len(self.ds.pairs)
        return npairs // self.batch + (int(round(self.task_ratio * npairs)) // self.batch if self.task_ratio else 0)
