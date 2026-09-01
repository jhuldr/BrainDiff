"""3D -> 2D conversion for the proprietary-model benchmark, per the methods spec, adapted
to our template-space volumes.

Spec: 224x224 axial slices, keep every other slice, [0,1] float-normalized, drop >90%-black
slices. Sequence priority (MRI): T1ce > FLAIR > T2w > T1w (DWI/ADC, SWI absent here).

ADAPTATION (documented): our volumes are template-space 193x229x193 @ 1mm (193 axial slices),
so literal "every other slice" = 96 slices/sequence, which violates the API slice budget
(Sonnet ~96 slices TOTAL / 4 seq ~= 24/seq; GPT ~360 / 15 ~= 24/seq). We therefore subsample
to `slices_per_seq` (default 24) EVENLY-spaced non-empty slices, matching the intended budget.
Set slices_per_seq high to recover literal every-other behavior.
"""
import base64, io, os
import numpy as np
import nibabel as nib
import cv2
from PIL import Image

MODALITY_PRIORITY = ["T1ce", "FLAIR", "T2w", "T1w"]   # post-contrast T1 > FLAIR > T2 > T1
SIZE = 224
BLACK_EPS = 1e-3          # a pixel is "black" below this (post-normalization)
BLACK_FRAC = 0.90         # drop slices with > this fraction black

def volume_to_slices(path, slices_per_seq=24, size=SIZE, axis=2):
    """Return a list of uint8 [size,size] axial slices (float-normalized -> 0..255)."""
    vol = nib.load(path).get_fdata().astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-6:
        return []
    vol = (vol - vmin) / (vmax - vmin)                 # [0,1] per volume
    vol = np.moveaxis(vol, axis, 0)                     # [Z, H, W] axial-first
    # keep non-(mostly-)black slices
    keep = [i for i in range(vol.shape[0])
            if (vol[i] < BLACK_EPS).mean() <= BLACK_FRAC]
    if not keep:
        return []
    # evenly subsample to slices_per_seq (budget-respecting stand-in for "every other slice")
    if len(keep) > slices_per_seq:
        sel = np.linspace(0, len(keep) - 1, slices_per_seq).round().astype(int)
        keep = [keep[j] for j in sel]
    out = []
    for i in keep:
        s = cv2.resize(vol[i], (size, size), interpolation=cv2.INTER_LINEAR)
        out.append((np.clip(s, 0, 1) * 255).astype(np.uint8))
    return out

def slice_to_b64_png(sl):
    buf = io.BytesIO()
    Image.fromarray(sl, mode="L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def study_sequences(image_rec, slices_per_seq=24):
    """image_rec: dict study_uid->modality paths. Returns ordered [(modality, [b64png,...])]
    by clinical priority, skipping absent modalities."""
    seqs = []
    for m in MODALITY_PRIORITY:
        p = image_rec.get(m)
        if isinstance(p, str) and p and os.path.exists(p):
            sls = volume_to_slices(p, slices_per_seq=slices_per_seq)
            if sls:
                seqs.append((m, [slice_to_b64_png(s) for s in sls]))
    return seqs

if __name__ == "__main__":
    import csv, sys
    csv.field_size_limit(10**7)
    img = {}
    with open("/home/data/BRAIN_DIFF_S4/image_extended.csv", newline="") as f:
        for r in csv.DictReader(f):
            img[r["study_uid"]] = r
    with open("paper/cache/subset_64.csv", newline="") as f:
        row = next(csv.DictReader(f))
    uid = row["study_uid2"]
    seqs = study_sequences(img[uid], slices_per_seq=24)
    print(f"study {uid}: {len(seqs)} sequences")
    for m, sl in seqs:
        print(f"  {m}: {len(sl)} slices, first b64 len={len(sl[0])}")
    # save one PNG to eyeball
    import base64 as b64
    open("benchmarks/mrrate_proprietary/_sample_slice.png", "wb").write(b64.b64decode(seqs[0][1][len(seqs[0][1])//2]))
    print("wrote _sample_slice.png (mid slice of first sequence)")
