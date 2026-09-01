"""Does the S3-trained diff path attend to where change actually happened?

Run after S3. Needs lesion masks at BOTH timepoints of a pair.

Every token is assigned to one of three regions from the two masks:

    CHANGE  = mask_current XOR mask_prior     lesion appeared or disappeared
    STABLE  = mask_current AND mask_prior     lesion present at both timepoints
    BACKGR  = neither

and the mean `gate` is reported in each. The three-way split separates the failure modes a
lesion-vs-background test cannot: high on CHANGE and low on STABLE is the diff path working;
high on both is a lesion detector rather than a change detector, which matters because 52.4%
of S4 is Stable; flat everywhere means the delta carries no localised signal.

NULL CONTROL (do not skip): the same statistic with masks shuffled across pairs. Lesions are
not uniformly distributed and neither is the gate, so a model favouring central brain tokens
would score enriched against a uniform expectation. Read enrichment against the shuffled
null, not against 1.0.

Also reports WindowedCrossAttention behaviour: per token, the attention mass on the identity
neighbour and the mean displacement over its 27 neighbours.

    python -m braindiff.eval.diag_gate_localization --ckpt checkpoints/nv_stage3_deltatune.pt --n 200

Vision-only (no decoder). ~2 GB, hard-capped so it cannot disturb a running job.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from braindiff.data.neurovfm_transforms import NeuroVFMGridd, NeuroVFMTokenize, N_TOKENS
from braindiff.models.diff_encoder import TOKEN_GRID, neighbour_index

MODALITIES = ["T1w", "T1ce", "T2w", "FLAIR"]
PATCH = (4, 16, 16)                 # voxels per token on the 48x224x192 grid
GD, GH, GW = TOKEN_GRID             # 12, 14, 12
CAP_GIB = 10.0


# ---------------------------------------------------------------- mask handling
def mask_to_tokens(mask_path, grid_tf, thresh=0.05, labels=None):
    """Voxel lesion mask -> [2016] bool on the token grid.

Uses the same spatial transform as the volumes, so the mask lands on the same 48x224x192
grid, then pools each 4x16x16 patch; a token is lesion if more than `thresh` of its voxels
are. Token order is row-major (d, h, w) with w fastest, matching tokenize_volume_fast and
the order neighbour_index() assumes.
    """
    if labels:
        # Selecting specific labels REQUIRES binarising before the grid transform
        # pools. After pooling, a patch containing labels 2 and 4 is a blended
        # scalar and the constituent labels cannot be recovered. So bypass the
        # dict transform and call to_neurovfm_grid on the binarised array directly
        # -- same function the transform uses, so the geometry is identical.
        import nibabel as nib
        from braindiff.data.neurovfm_transforms import to_neurovfm_grid
        img = nib.load(str(mask_path))
        raw = img.get_fdata(dtype=np.float32)
        sel = np.isin(np.rint(raw), list(labels)).astype(np.float32)
        vol = to_neurovfm_grid(sel, nib.aff2axcodes(img.affine)).unsqueeze(0)
    else:
        vol = grid_tf({"k": mask_path})["k"]
    arr = np.asarray(vol)[0]
    # These are LABEL maps, not binary: BraTS-GLI uses {2 edema, 4 enhancing},
    # BraTS-MET {1 necrotic, 2 edema, 3 enhancing}. grid_tf mean-pools BEFORE we get
    # here, so thresholding the pooled map at 0.5 makes sensitivity depend on the
    # label's numeric VALUE: a patch 15% occupied by label 4 pools to 0.6 and counts,
    # while 40% occupancy of label 1 pools to 0.4 and does not. Measured cost of the
    # old `> 0.5`: 1 of 32 lesion tokens dropped on GLI, but 2 of 8 (25%) on MET,
    # whose labels are smaller numbers. Any nonzero pooled value means some lesion
    # voxel contributed, so threshold just above zero and let `thresh` below decide
    # occupancy -- that criterion is label-agnostic.
    arr = (arr > 1e-6).astype(np.float32)
    d, h, w = arr.shape
    assert (d, h, w) == (GD * PATCH[0], GH * PATCH[1], GW * PATCH[2]), \
        f"mask grid {arr.shape} != expected {(GD*PATCH[0], GH*PATCH[1], GW*PATCH[2])}"
    frac = arr.reshape(GD, PATCH[0], GH, PATCH[1], GW, PATCH[2]).mean(axis=(1, 3, 5))
    return torch.from_numpy((frac > thresh).reshape(-1))      # [2016]


def resolve_mask(vol_path, suffix, mask_map):
    if mask_map is not None:
        return mask_map.get(vol_path)
    for mod in MODALITIES:
        cand = vol_path.replace(f"_{mod}.nii.gz", suffix)
        if cand != vol_path and os.path.exists(cand):
            return cand
    cand = vol_path.replace(".nii.gz", suffix)
    return cand if os.path.exists(cand) else None


# ---------------------------------------------------------------- statistics
def auc(scores, labels):
    """Rank-based AUC, no sklearn. labels bool, scores float."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = s.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def neighbour_offsets():
    """Displacement in tokens for each of the K neighbours, from idx geometry."""
    idx, valid = neighbour_index(TOKEN_GRID, 1)                # [N, K]
    n = torch.arange(idx.shape[0])
    to_dhw = lambda t: torch.stack([t // (GH * GW), (t // GW) % GH, t % GW], -1).float()
    off = to_dhw(idx) - to_dhw(n)[:, None, :]                  # [N, K, 3]
    return idx, valid, off


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="S3 checkpoint (nv_stage3_deltatune.pt)")
    ap.add_argument("--pairs-csv", default="/home/data/BRAIN_DIFF_S3/main.csv")
    ap.add_argument("--image-csv", default="/home/data/BRAIN_DIFF_S3/image.csv")
    ap.add_argument("--mask-suffix", default="_lesion.nii.gz")
    ap.add_argument("--mask-csv", default=None,
                    help="optional CSV with columns volume_path,mask_path")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-change-tokens", type=int, default=3)
    ap.add_argument("--labels", default=None,
                    help="comma-separated BraTS labels to count as lesion, e.g. "
                         "'4,3' for enhancing tumour only (4=GLI, 3=MET). Default: "
                         "every nonzero label merged, which is ~84%% edema by volume "
                         "on GLI and therefore measures edema movement, not tumour "
                         "change. Applied BEFORE the grid pool, which is the only "
                         "point at which labels are still separable.")
    ap.add_argument("--uid-a", default="UID_1")
    ap.add_argument("--uid-b", default="UID_2")
    a = ap.parse_args()

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(CAP_GIB / total, 0)
    print(f"[guard] capped at {CAP_GIB:.0f} GiB of {total:.0f}", flush=True)

    from braindiff.models.diff_encoder import DiffPretrainModel
    from braindiff.training.checkpoint import load_stage_checkpoint

    model = DiffPretrainModel(num_queries=64, device="cuda:0",
                              vision_lora_r=32, vision_lora_alpha=64).to("cuda:0").eval()
    state = torch.load(a.ckpt, map_location="cuda:0")
    load_stage_checkpoint(model.captioner, state, label=a.ckpt,
                          strict_groups=("change_map",))
    diff = model.captioner.change_map

    # Capture WindowedCrossAttention weights. The module does not return them, so
    # wrap its softmax by monkeypatching forward once.
    grabbed = {}
    wca = diff.cross_attn
    orig_forward = wca.forward

    def patched(main_, ref_, idx_, valid_):
        import math as _m
        b_, n_, _ = main_.shape
        q = wca.to_q(wca.norm_q(main_))
        k, v = wca.to_kv(wca.norm_kv(ref_)).chunk(2, dim=-1)
        flat = idx_.reshape(-1)
        k = k[:, flat].view(b_, n_, idx_.shape[1], wca.heads, wca.head_dim)
        v = v[:, flat].view(b_, n_, idx_.shape[1], wca.heads, wca.head_dim)
        q = q.view(b_, n_, wca.heads, wca.head_dim)
        sc = torch.einsum("bnhd,bnkhd->bnkh", q, k) / _m.sqrt(wca.head_dim)
        sc = sc.masked_fill(~valid_[None, :, :, None], float("-inf"))
        w = torch.softmax(sc, dim=2)
        grabbed["w"] = w.detach().mean(-1)                    # [B*M, N, K] over heads
        out = torch.einsum("bnkh,bnkhd->bnhd", w, v)
        return wca.to_out(out.reshape(b_, n_, -1))

    wca.forward = patched

    grid_tf = NeuroVFMGridd(keys=["k"], allow_missing_keys=True)
    tok_tf = NeuroVFMTokenize(keys=["k"], allow_missing_keys=True)
    _, _, offsets = neighbour_offsets()
    off_norm = offsets.norm(dim=-1)                            # [N, K] tokens

    label_set = ([int(x) for x in a.labels.split(",")] if a.labels else None)
    if label_set:
        print(f"[masks] restricted to labels {label_set} (binarised before pooling)",
              flush=True)
    mask_map = None
    if a.mask_csv:
        mm = pd.read_csv(a.mask_csv)
        mask_map = dict(zip(mm["volume_path"], mm["mask_path"]))

    pairs = pd.read_csv(a.pairs_csv, low_memory=False)
    img = pd.read_csv(a.image_csv, low_memory=False)
    # Pick the image.csv column that actually holds the pair UIDs, by coverage,
    # rather than assuming a name. image.csv carries BOTH `StudyUID` and `UID`, and
    # they are not interchangeable: StudyUID is the SUBJECT-level id that repeats
    # across timepoints, while main.csv's UID_1/UID_2 are per-STUDY and match `UID`.
    # Hard-coding "StudyUID" indexed on the subject id, so every membership test
    # failed and the run reported "No usable pairs" with no error to explain it.
    want = set(pairs[a.uid_a]) | set(pairs[a.uid_b])
    cands = [c for c in ("UID", "StudyUID", "study_uid") if c in img.columns]
    if not cands:
        raise SystemExit(f"{a.image_csv} has none of UID/StudyUID/study_uid; "
                         f"got {list(img.columns)}")
    cover = {c: len(want & set(img[c].dropna())) for c in cands}
    key = max(cover, key=cover.get)
    print(f"[join] pair-UID coverage by image.csv column: {cover} -> using '{key}'",
          flush=True)
    if cover[key] == 0:
        raise SystemExit(f"No pair UID appears in any of {cands}. The pairs CSV and "
                         f"image CSV do not refer to the same studies.")
    img = img.set_index(key)

    def encode(path):
        v = grid_tf({"k": path})["k"]
        d = tok_tf({"k": v})["k"]
        with torch.no_grad():
            return model.captioner.vision_encoder(
                d["tokens"].unsqueeze(0).cuda().float(),
                d["coords"].unsqueeze(0).cuda().long())        # [1, N, 768]

    rec = {r: [] for r in ("change", "stable", "backgr")}
    # Same three-way split applied to the RAW ENCODER FEATURES, before the
    # DiffEncoder touches them. This separates two very different diagnoses when the
    # gate reads at chance:
    #   features separate, gate does not -> the DiffEncoder/gate is discarding a
    #       signal it was handed; S3's objective is what needs changing.
    #   features do not separate either   -> the signal is already gone at
    #       tokenisation (16 mm isotropic tokens vs ~8 mm median lesions), and no
    #       amount of S3/S4 training can recover it. A design finding, not a
    #       training one.
    # L2 and cosine are both recorded: L2 is what the pointwise (B-A) term actually
    # sees, cosine is scale-free and survives any per-token magnitude drift.
    rec_l2 = {r: [] for r in ("change", "stable", "backgr")}
    rec_cos = {r: [] for r in ("change", "stable", "backgr")}
    all_l2, all_cos = [], []
    all_g, all_y = [], []
    disp = {"change": [], "backgr": []}
    centre = {"change": [], "backgr": []}
    token_masks, used = [], 0

    for _, row in pairs.iterrows():
        if used >= a.n:
            break
        ua, ub = row[a.uid_a], row[a.uid_b]
        if ua not in img.index or ub not in img.index:
            continue
        mods = [m for m in MODALITIES
                if isinstance(img.loc[ua].get(m), str) and isinstance(img.loc[ub].get(m), str)]
        if not mods:
            continue
        pa, pb = img.loc[ua, mods[0]], img.loc[ub, mods[0]]
        ma, mb = resolve_mask(pa, a.mask_suffix, mask_map), resolve_mask(pb, a.mask_suffix, mask_map)
        if ma is None or mb is None:
            continue
        try:
            t_a = mask_to_tokens(ma, grid_tf, labels=label_set)
            t_b = mask_to_tokens(mb, grid_tf, labels=label_set)
        except Exception:
            continue
        change, stable = t_a ^ t_b, t_a & t_b
        if int(change.sum()) < a.min_change_tokens:
            continue

        fa, fb = encode(pa), encode(pb)                        # [1, N, 768] each
        with torch.no_grad():
            # change_map wants [B, M, N, 768]; single modality here -> M=1.
            _, aux = diff(fa.unsqueeze(1), fb.unsqueeze(1))
        # aux["gate"] is now the PER-TOKEN saliency [1,1,N] -> score directly against the
        # per-token change/stable/background masks (no upsampling).
        g = aux["gate"].reshape(-1).float().cpu().numpy()                 # [N]

        with torch.no_grad():
            d_l2 = (fb - fa).norm(dim=-1).squeeze(0).float().cpu().numpy()      # [N]
            d_cos = (1.0 - F.cosine_similarity(fa, fb, dim=-1)).squeeze(0).float().cpu().numpy()

        backgr = ~(t_a | t_b)
        for name, m_ in (("change", change), ("stable", stable), ("backgr", backgr)):
            if m_.any():
                mnp = m_.numpy()
                rec[name].append(float(g[mnp].mean()))
                rec_l2[name].append(float(d_l2[mnp].mean()))
                rec_cos[name].append(float(d_cos[mnp].mean()))
        all_g.append(g); all_y.append(change.numpy())
        all_l2.append(d_l2); all_cos.append(d_cos)
        token_masks.append(change.numpy())

        w = grabbed.get("w")
        if w is not None:
            w0 = w[0].float().cpu()                            # [N, K]
            d_mean = (w0 * off_norm).sum(-1).numpy()           # expected |offset|
            c_mass = w0[torch.arange(w0.shape[0]),
                        off_norm.argmin(dim=-1)].numpy()       # identity neighbour
            for name, m_ in (("change", change.numpy()), ("backgr", backgr.numpy())):
                if m_.any():
                    disp[name].append(float(d_mean[m_].mean()))
                    centre[name].append(float(c_mass[m_].mean()))
        used += 1
        if used % 25 == 0:
            print(f"  {used}/{a.n} pairs", flush=True)

    if used == 0:
        print("\nNo usable pairs. Check --mask-suffix / --mask-csv and the UID columns.")
        return 1

    # ---- null: shuffle the change masks across pairs ----
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(token_masks))
    null = [float(all_g[i][token_masks[perm[i]]].mean())
            for i in range(len(all_g)) if token_masks[perm[i]].any()]
    # Same permutation for the feature nulls, so gate and features are judged
    # against an identically-constructed chance level.
    null_l2 = [float(all_l2[i][token_masks[perm[i]]].mean())
               for i in range(len(all_l2)) if token_masks[perm[i]].any()]
    null_cos = [float(all_cos[i][token_masks[perm[i]]].mean())
                for i in range(len(all_cos)) if token_masks[perm[i]].any()]

    m = {k: float(np.mean(v)) for k, v in rec.items() if v}
    null_m = float(np.mean(null))
    tok_auc = float(np.mean([auc(g, y) for g, y in zip(all_g, all_y) if y.any() and (~y).any()]))

    print(f"\n{'='*70}\nGATE LOCALISATION  --  {used} pairs, {a.ckpt}\n{'='*70}")
    print(f"  mean gate | CHANGE (xor)        {m.get('change', float('nan')):.4f}")
    print(f"  mean gate | STABLE lesion (and) {m.get('stable', float('nan')):.4f}")
    print(f"  mean gate | BACKGROUND          {m.get('backgr', float('nan')):.4f}")
    print(f"  mean gate | SHUFFLED mask (null){null_m:.4f}   <- the honest chance level")
    print(f"\n  enrichment CHANGE / BACKGROUND  {m.get('change',np.nan)/m.get('backgr',np.nan):.2f}x")
    print(f"  enrichment CHANGE / SHUFFLED    {m.get('change',np.nan)/null_m:.2f}x  <- read THIS one")
    print(f"  CHANGE vs STABLE ratio          {m.get('change',np.nan)/m.get('stable',np.nan):.2f}x")
    print(f"  per-token AUC (gate detects change) {tok_auc:.3f}   (0.5 = chance)")

    # ---- upstream check: do the RAW ENCODER FEATURES separate at all? ----
    ml2 = {k: float(np.mean(v)) for k, v in rec_l2.items() if v}
    mcos = {k: float(np.mean(v)) for k, v in rec_cos.items() if v}
    nl2, ncos = float(np.mean(null_l2)), float(np.mean(null_cos))
    l2_auc = float(np.mean([auc(d, y) for d, y in zip(all_l2, all_y)
                            if y.any() and (~y).any()]))
    cos_auc = float(np.mean([auc(d, y) for d, y in zip(all_cos, all_y)
                             if y.any() and (~y).any()]))
    print(f"\n  ENCODER FEATURES (before the DiffEncoder) -- is the signal even there?")
    print(f"    ||f_main - f_ref||   CHANGE {ml2.get('change',np.nan):.4f}   "
          f"STABLE {ml2.get('stable',np.nan):.4f}   BACKGR {ml2.get('backgr',np.nan):.4f}   "
          f"SHUFFLED {nl2:.4f}")
    print(f"      enrichment CHANGE/SHUFFLED {ml2.get('change',np.nan)/nl2:.2f}x   "
          f"per-token AUC {l2_auc:.3f}")
    print(f"    cosine distance      CHANGE {mcos.get('change',np.nan):.4f}   "
          f"STABLE {mcos.get('stable',np.nan):.4f}   BACKGR {mcos.get('backgr',np.nan):.4f}   "
          f"SHUFFLED {ncos:.4f}")
    print(f"      enrichment CHANGE/SHUFFLED {mcos.get('change',np.nan)/ncos:.2f}x   "
          f"per-token AUC {cos_auc:.3f}")
    feat_sep = max(ml2.get('change', 0) / max(nl2, 1e-9),
                   mcos.get('change', 0) / max(ncos, 1e-9))
    print(f"    -> features {'DO' if feat_sep > 1.10 else 'DO NOT'} separate change from "
          f"the shuffled null (best {feat_sep:.2f}x)")

    if disp["change"]:
        print(f"\n  WindowedCrossAttention:")
        print(f"    mean |displacement| tokens   change {np.mean(disp['change']):.3f}   "
              f"background {np.mean(disp['backgr']):.3f}")
        print(f"    mass on identity neighbour   change {np.mean(centre['change']):.3f}   "
              f"background {np.mean(centre['backgr']):.3f}")
        print(f"    (1 token = 16 mm. ~0 displacement and ~1.0 centre mass means the")
        print(f"     window is unused and the module is an expensive identity.)")

    print(f"\n  READING IT:")
    ch, st = m.get('change', np.nan), m.get('stable', np.nan)
    if not np.isfinite(ch / null_m) or ch / null_m < 1.15:
        print("    Gate is at chance -> the delta carries no localised change signal.")
        print("    Bottleneck is upstream: nuisance suppression, not the connector.")
    elif ch / st < 1.15:
        print("    Gate fires on CHANGE and STABLE alike -> LESION detector, not a")
        print("    CHANGE detector. It will not separate 'Stable' (52.4% of S4) from")
        print("    real progression. Target the delta on mask XOR, not presence.")
    else:
        print("    Gate is enriched on change and suppressed on stable disease ->")
        print("    the diff path is doing its job; look downstream for the bottleneck.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
