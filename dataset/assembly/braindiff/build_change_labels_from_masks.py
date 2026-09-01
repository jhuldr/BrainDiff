"""Derive 7-way change labels from paired BraTS segmentations.

Gives S3 objective change labels for the 1,633 mask pairs, in the same class space as S4's
`classification`, so both label sources train one head.

Label semantics were verified empirically on this data (z-scored intensity per label, 24
volumes across both cohorts), matching the BraTS-2024 post-treatment convention:

    1  NETC  non-enhancing tumour core   T1ce-T1w +0.28, FLAIR 1.40
    2  SNFH  edema                       highest FLAIR 1.56, largest volume
    3  ET    enhancing tumour            T1ce-T1w +1.41  <- the measurable disease
    4  RC    resection cavity            T2 2.25 with FLAIR -0.10

Label 4 is a static post-operative structure that dilutes change signal, so "enhancing
tumour only" is `--labels 3`, not `--labels 4,3`. Burden is ET (+ NETC with --use_netc);
edema is excluded as reactive and steroid-responsive, cavity as barely changing.

Threshold is +/-40% by volume, not RANO's +/-25%: RANO is bidimensional and volume goes as
roughly the 1.5 power of area. At +/-25% only 305 of 1,315 both-present pairs come out
stable, and an interquartile log2 ratio of +/-0.9 says that band is mostly segmentation
noise.

--pairs selects the table to label IN PLACE and must be one build_s3sup_pairs.py wrote,
since dense rows are keyed on its prefix_1/prefix_2 columns. Run this after every
build_s3sup_pairs.py run: that script resets change_label to -1.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

class _Notify:
    """Slack notifications are not part of the release; calls become no-ops."""
    @staticmethod
    def send(*_a, **_k):
        return None


notify = _Notify()

S3_ROOT = "/home/data/BRAIN_DIFF_S3"
PAIRS = os.path.join(S3_ROOT, "main_supervised.csv")
COHORTS = ("BraTS-GLI", "BraTS-MET")

CHANGE_CLASSES = ("Stable", "New lesion", "Indeterminate", "Progressed",
                  "Improved", "Mixed interval change", "Resolved")
IDX = {c: i for i, c in enumerate(CHANGE_CLASSES)}

ET, NETC = 3, 1
THRESH = 0.40
MIN_COMPONENT = 50     # voxels; below this a "lesion" is segmentation speckle
MATCH_RADIUS = 20      # voxels (~2 cm at 1 mm) to call two components the same lesion


def mask_path(prefix, modality="T1ce"):
    for coh in COHORTS:
        p = os.path.join(S3_ROOT, f"{coh}_LESION", f"{prefix}_{modality}_lesion.nii.gz")
        if os.path.exists(p):
            return p
    return None


def burden(prefix, use_netc):
    """(total burden voxels, per-component volumes, component centroids)."""
    import nibabel as nib
    from scipy import ndimage
    p = mask_path(prefix)
    if p is None:
        return None
    d = np.asarray(nib.load(p).get_fdata()).astype(np.int16)
    m = (d == ET) | (d == NETC) if use_netc else (d == ET)
    if not m.any():
        return 0, np.zeros((0,), dtype=int), np.zeros((0, 3))
    lab, n = ndimage.label(m)
    sizes = np.array(ndimage.sum(m, lab, range(1, n + 1)), dtype=int)
    cents = np.array(ndimage.center_of_mass(m, lab, range(1, n + 1))).reshape(-1, 3)
    keep = sizes >= MIN_COMPONENT
    return int(m.sum()), sizes[keep], cents[keep]


def classify(a, b):
    """(class name, diagnostics) for one pair, from the two burdens."""
    tot_a, comp_a, cent_a = a
    tot_b, comp_b, cent_b = b

    if tot_a == 0 and tot_b == 0:
        return "Stable", {"reason": "no measurable disease either timepoint"}
    if tot_a == 0:
        return "New lesion", {"reason": "absent at prior"}
    if tot_b == 0:
        return "Resolved", {"reason": "absent at current"}

    ratio = tot_b / tot_a
    # Mixed: divergent components. Only meaningful with several lesions -- the
    # metastasis case. A single lesion cannot be "mixed" by this definition, so
    # requiring >=2 matched components stops one noisy blob producing the label.
    if len(comp_a) >= 2 and len(comp_b) >= 2:
        matched, used = [], set()
        for i, c in enumerate(cent_a):
            dist = np.linalg.norm(cent_b - c, axis=1)
            for j in np.argsort(dist):
                if j not in used and dist[j] < MATCH_RADIUS:
                    matched.append((comp_a[i], comp_b[j]))
                    used.add(int(j))
                    break
        if len(matched) >= 2:
            r = np.array([y / x for x, y in matched], dtype=float)
            if (r >= 1 + THRESH).any() and (r <= 1 - THRESH).any():
                return "Mixed interval change", {"ratio": float(ratio),
                                                 "matched": len(matched)}
    if ratio >= 1 + THRESH:
        return "Progressed", {"ratio": float(ratio)}
    if ratio <= 1 - THRESH:
        return "Improved", {"ratio": float(ratio)}
    return "Stable", {"ratio": float(ratio)}


def job(args):
    p1, p2, use_netc = args
    a, b = burden(p1, use_netc), burden(p2, use_netc)
    if a is None or b is None:
        return p1, p2, None
    cls, _ = classify(a, b)
    return p1, p2, cls


def main(a):
    df = pd.read_csv(a.pairs)
    dense = df[df["has_dense"] == 1]
    print(f"{len(dense)} mask-labelled pairs", flush=True)

    tasks = [(r["prefix_1"], r["prefix_2"], a.use_netc) for _, r in dense.iterrows()]
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, t) for t in tasks]
        for n, f in enumerate(as_completed(futs), 1):
            p1, p2, cls = f.result()
            if cls:
                out[(p1, p2)] = cls
            if n % 400 == 0:
                print(f"  {n}/{len(tasks)}", flush=True)

    counts = pd.Series(list(out.values())).value_counts()
    print("\nderived label distribution:")
    print(counts.to_string())
    print(f"  unlabelled (mask unreadable): {len(tasks) - len(out)}")

    if "label_source" not in df.columns:
        df["label_source"] = "none"
    df.loc[df["has_global"] == 1, "label_source"] = "s4_report"

    n_set = 0
    for i, r in df.iterrows():
        cls = out.get((r["prefix_1"], r["prefix_2"]))
        if cls is None:
            continue
        df.at[i, "change_label"] = IDX[cls]
        df.at[i, "has_global"] = 1
        df.at[i, "label_source"] = "brats_mask"
        n_set += 1

    summary = (f"mask-derived labels: {n_set} pairs\n" +
               "\n".join(f"  {k:<24}{v}" for k, v in counts.items()) +
               f"\ntotal labelled now: {int(df['has_global'].sum())} "
               f"({int((df['label_source'] == 'brats_mask').sum())} objective "
               f"+ {int((df['label_source'] == 's4_report').sum())} report-derived)")
    print("\n" + summary, flush=True)

    if not a.dry_run:
        df.to_csv(a.pairs, index=False)
        print(f"Wrote {a.pairs}", flush=True)
        notify.send(f"*build_change_labels_from_masks* done\n```\n{summary}\n```")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--pairs", default=PAIRS,
                   help="pair table to label in place (build_s3sup_pairs.py output)")
    p.add_argument("--use_netc", action="store_true",
                   help="count non-enhancing core in the burden as well as ET")
    p.add_argument("--dry_run", action="store_true")
    main(p.parse_args())
