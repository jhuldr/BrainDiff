"""Fig. 2: within-pair change-decodability AUROC stratified by interval-change MAGNITUDE.
Tests token-scale hypothesis (AUROC rises with change size) vs registration/pipeline defect
(flat). Sharded feature dump + merge (GroupKFold per-pair AUROC, then stratify by |log2 vol
ratio|). Frozen NeuroVFM (S2 base). GPU.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os, sys, json, glob, argparse, traceback
sys.path.insert(0, _REPO)
import numpy as np

VOX_MM3 = 1.0 * 1.0 * 4.0   # NeuroVFM grid spacing 1x1x4 mm -> per-voxel volume in the grid frame

def dump(a):
    import torch
    from braindiff.eval.diag_gate_localization import mask_to_tokens, resolve_mask, MODALITIES
    from braindiff.data.neurovfm_transforms import NeuroVFMGridd, NeuroVFMTokenize
    from braindiff.models.diff_encoder import DiffPretrainModel
    from braindiff.training.checkpoint import load_stage_checkpoint
    import pandas as pd, nibabel as nib
    dev = f"cuda:{a.gpu}"; torch.cuda.set_device(dev)
    model = DiffPretrainModel(num_queries=64, device=dev, vision_lora_r=32, vision_lora_alpha=64).to(dev).eval()
    load_stage_checkpoint(model.captioner, torch.load("checkpoints/nv_stage2_reportgen.pt", map_location=dev),
                          label="S2", is_main=True)
    grid_tf = NeuroVFMGridd(keys=["k"], allow_missing_keys=True); tok_tf = NeuroVFMTokenize(keys=["k"], allow_missing_keys=True)
    label_set = [int(x) for x in a.labels.split(",")]
    def enc(p):
        d = tok_tf({"k": grid_tf({"k": p})["k"]})["k"]
        with torch.no_grad():
            return model.captioner.vision_encoder(d["tokens"].unsqueeze(0).cuda().float(),
                                                  d["coords"].unsqueeze(0).cuda().long()).squeeze(0)
    def lesion_vox(mpath):
        m = nib.load(mpath).get_fdata()
        return float(np.isin(m, label_set).sum())
    mm = pd.read_csv("/home/data/BRAIN_DIFF_S3/lesion_index.csv"); mask_map = dict(zip(mm["volume_path"], mm["mask_path"]))
    pairs = pd.read_csv("/home/data/BRAIN_DIFF_S3/main.csv", low_memory=False)
    img3 = pd.read_csv("/home/data/BRAIN_DIFF_S3/image.csv", low_memory=False).set_index("UID")
    X, y, grp, vmag = [], [], [], {}
    gidx = 0
    for _, row in pairs.iterrows():
        if gidx >= a.n_pairs: break
        ua, ub = row.get("UID_1"), row.get("UID_2")
        if ua not in img3.index or ub not in img3.index: continue
        mods = [m for m in MODALITIES if isinstance(img3.loc[ua].get(m), str) and isinstance(img3.loc[ub].get(m), str)]
        if not mods: continue
        pa, pb = img3.loc[ua, mods[0]], img3.loc[ub, mods[0]]
        ma, mb = resolve_mask(pa, "_lesion.nii.gz", mask_map), resolve_mask(pb, "_lesion.nii.gz", mask_map)
        if ma is None or mb is None: continue
        try:
            ta = mask_to_tokens(ma, grid_tf, labels=label_set); tb = mask_to_tokens(mb, grid_tf, labels=label_set)
        except Exception: continue
        ch, st = (ta ^ tb), (ta & tb)
        if int(ch.sum()) < 3 or int(st.sum()) < 3: continue
        this = gidx; gidx += 1
        if this % a.num_shards != a.shard_index: continue
        try:
            va, vb = lesion_vox(ma), lesion_vox(mb)
        except Exception: continue
        if va < 1 or vb < 1: continue
        fa, fb = enc(pa), enc(pb); d = torch.cat([(fb - fa), (fb - fa).abs()], -1).cpu().numpy()
        for i in np.where(ch.numpy())[0]: X.append(d[i]); y.append(1); grp.append(this)
        for i in np.where(st.numpy())[0]: X.append(d[i]); y.append(0); grp.append(this)
        vmag[this] = abs(np.log2(vb / va))    # |log2 volume ratio| = interval-change magnitude
    np.savez(a.dump, X=np.array(X), y=np.array(y), grp=np.array(grp),
             vk=np.array(list(vmag.keys())), vv=np.array(list(vmag.values())))
    print(f"shard {a.shard_index}: tokens={len(X)} pairs={len(vmag)} -> {a.dump}")

def merge(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    fs = sorted(glob.glob(a.dump_glob))
    X = np.vstack([np.load(f)["X"] for f in fs if len(np.load(f)["X"])])
    y = np.concatenate([np.load(f)["y"] for f in fs if len(np.load(f)["y"])])
    grp = np.concatenate([np.load(f)["grp"] for f in fs if len(np.load(f)["grp"])])
    mag = {}
    for f in fs:
        z = np.load(f)
        for k, v in zip(z["vk"], z["vv"]): mag[int(k)] = float(v)
    # per-pair within-pair AUROC via GroupKFold
    pair_auc = {}
    for tr, te in GroupKFold(5).split(X, y, grp):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]); sc = clf.predict_proba(X[te])[:, 1]
        for g in np.unique(grp[te]):
            m = grp[te] == g
            if y[te][m].sum() >= 1 and (~y[te][m].astype(bool)).sum() >= 1:
                pair_auc[int(g)] = roc_auc_score(y[te][m], sc[m])
    pairs = [(mag[g], auc) for g, auc in pair_auc.items() if g in mag]
    pairs.sort()
    mags = np.array([p[0] for p in pairs]); aucs = np.array([p[1] for p in pairs])
    # quartile strata by magnitude
    qs = np.quantile(mags, [0, 0.25, 0.5, 0.75, 1.0])
    strata = []
    for i in range(4):
        lo, hi = qs[i], qs[i + 1]
        sel = (mags >= lo) & (mags <= hi) if i == 3 else (mags >= lo) & (mags < hi)
        strata.append({"stratum": i + 1, "log2ratio_range": [float(lo), float(hi)],
                       "median_log2ratio": float(np.median(mags[sel])),
                       "approx_fold_change": float(2 ** np.median(mags[sel])),
                       "n_pairs": int(sel.sum()), "mean_auroc": float(np.mean(aucs[sel]))})
    res = {"overall_within_pair_auroc": float(np.mean(aucs)), "n_pairs": len(aucs), "strata": strata,
           "note": "magnitude = |log2(lesion_vol_t2/lesion_vol_t1)|; token grid 1x1x4mm, one token ~16mm iso"}
    json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=2)); print("FIG2_DONE")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump", "merge"], default="dump")
    ap.add_argument("--n-pairs", type=int, default=500); ap.add_argument("--labels", default="4,3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--dump", default=""); ap.add_argument("--dump-glob", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "fig2_strata_results.json"))
    a = ap.parse_args()
    if a.mode == "dump":
        try: dump(a)
        except Exception as e: print("ERR", e, traceback.format_exc()[-800:])
    else:
        merge(a)

if __name__ == "__main__":
    main()
