"""Precompute per-token tumour occupancy from the aligned BraTS masks.

create_longitudinal.py already warped every segmentation with the same affine as its
modality, nearest-neighbour, writing 9,576 files under BraTS-{GLI,MET}_LESION/. Nothing
downstream can see them because image.csv records the SOURCE path instead, which is in
original BraTS space (320x320x110) and does not match the volumes the model reads.

This script reduces each mask to the NeuroVFM token grid once, so training never opens a
nii.gz. One token is a (4,16,16) patch at 1x1x4 mm spacing, ~16 mm isotropic, and the grid
over a 193x229x193 template volume is 12x14x12 = 2016.

Output per volume+modality: float32 [12,14,12] giving the FRACTION of each token's voxels
that are tumour. A fraction rather than a bool, because the downstream target is whether
tumour burden in a token moved, which a threshold would discard. Masks are per-modality
(each modality got its own affine), so they are kept separate and never intersected.

    python -m dataset3D.build_lesion_tokens --workers 16
"""
import argparse
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

LESION_DIRS = ("/home/data/BRAIN_DIFF_S3/BraTS-GLI_LESION",
               "/home/data/BRAIN_DIFF_S3/BraTS-MET_LESION")
OUT_DIR = "/home/data/BRAIN_DIFF_S3/LESION_TOKENS"
GRID = (12, 14, 12)
NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<mod>T1w|T1ce|T2w|FLAIR)_lesion\.nii\.gz$")


def token_occupancy(mask, grid=GRID):
    """[D,H,W] mask -> [gd,gh,gw] float32 fraction of voxels that are tumour.

np.array_split defines the token boundaries: the template dims are not divisible by the
grid (193/12, 229/14, 193/12), so tokens differ in size by one voxel and a reshape-based
reduction would silently misalign.
    """
    binary = (mask > 0)
    out = np.zeros(grid, dtype=np.float32)
    zs = np.array_split(np.arange(binary.shape[0]), grid[0])
    ys = np.array_split(np.arange(binary.shape[1]), grid[1])
    xs = np.array_split(np.arange(binary.shape[2]), grid[2])
    for i, z in enumerate(zs):
        for j, y in enumerate(ys):
            for k, x in enumerate(xs):
                block = binary[np.ix_(z, y, x)]
                out[i, j, k] = block.mean() if block.size else 0.0
    return out


def process(path):
    import nibabel as nib
    m = NAME_RE.match(os.path.basename(path))
    if m is None:
        return path, None, "unparseable filename"
    key = f"{m.group('prefix')}_{m.group('mod')}"
    out_path = os.path.join(OUT_DIR, key + ".npy")
    if os.path.exists(out_path):
        return path, key, "cached"
    try:
        data = nib.load(path).get_fdata()
    except Exception as e:                      # a truncated nii.gz should not kill the run
        return path, None, f"load failed: {type(e).__name__}"
    occ = token_occupancy(np.asarray(data))
    np.save(out_path, occ)
    return path, key, "ok"


def main(workers, limit):
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(p for d in LESION_DIRS for p in glob.glob(os.path.join(d, "*_lesion.nii.gz")))
    if limit:
        paths = paths[:limit]
    print(f"{len(paths)} masks -> {OUT_DIR}", flush=True)

    counts, failures = {}, []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process, p) for p in paths]
        for n, fut in enumerate(as_completed(futs), 1):
            path, key, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if key is None:
                failures.append((path, status))
            if n % 500 == 0:
                print(f"  {n}/{len(paths)}  {counts}", flush=True)

    print(f"done: {counts}", flush=True)
    for path, status in failures[:10]:
        print(f"  FAILED {os.path.basename(path)}: {status}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} masks failed; fix before training")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N masks")
    a = p.parse_args()
    main(a.workers, a.limit)
