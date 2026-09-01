"""Table 5 NUISANCE control: augmented duplicate pairs (one scan augmented twice -> nonzero
feature delta, ZERO true change). Tests whether the within-pair change probe fires on
augmentation-only feature differences. Sharded across GPUs (dump features per shard, then merge).

Reported numbers:
  (A) separability AUROC: real-CHANGE tokens (1) vs duplicate-pair tokens (0). HIGH => the probe's
      change signal is specific to REAL interval change (validates the 0.601 within-pair result);
      ~0.5 => the probe cannot tell real change from augmentation noise.
  (B) applied-probe change-rate: train change(1)-vs-stable(0) on REAL pairs, apply to duplicate
      tokens -> mean predicted P(change); clean if ~= real-stable base rate (near chance).
Frozen NeuroVFM (S2 base). GPU.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os, sys, json, glob, argparse, traceback
sys.path.insert(0, _REPO)
import numpy as np

def dump(a):
    import torch
    from monai.transforms import RandBiasField, RandGaussianNoise
    from braindiff.eval.diag_gate_localization import mask_to_tokens, resolve_mask, MODALITIES
    from braindiff.data.neurovfm_transforms import NeuroVFMGridd, NeuroVFMTokenize
    from braindiff.models.diff_encoder import DiffPretrainModel
    from braindiff.training.checkpoint import load_stage_checkpoint
    import pandas as pd
    dev = f"cuda:{a.gpu}"; torch.cuda.set_device(dev)
    model = DiffPretrainModel(num_queries=64, device=dev, vision_lora_r=32, vision_lora_alpha=64).to(dev).eval()
    load_stage_checkpoint(model.captioner, torch.load("checkpoints/nv_stage2_reportgen.pt", map_location=dev),
                          label="S2", is_main=True)
    grid_tf = NeuroVFMGridd(keys=["k"], allow_missing_keys=True)
    tok_tf = NeuroVFMTokenize(keys=["k"], allow_missing_keys=True)
    label_set = [int(x) for x in a.labels.split(",")]
    bias = RandBiasField(prob=1.0, coeff_range=(0.0, 0.1)); noise = RandGaussianNoise(prob=1.0, std=0.03)
    def _enc(vol):
        d = tok_tf({"k": vol})["k"]
        with torch.no_grad():
            return model.captioner.vision_encoder(d["tokens"].unsqueeze(0).cuda().float(),
                                                  d["coords"].unsqueeze(0).cuda().long()).squeeze(0)
    def enc_plain(p): return _enc(grid_tf({"k": p})["k"])
    def enc_aug(p, seed):
        vol = grid_tf({"k": p})["k"]; bias.set_random_state(seed); noise.set_random_state(seed + 10007)
        return _enc(noise(bias(vol)))
    mm = pd.read_csv("/home/data/BRAIN_DIFF_S3/lesion_index.csv"); mask_map = dict(zip(mm["volume_path"], mm["mask_path"]))
    pairs = pd.read_csv("/home/data/BRAIN_DIFF_S3/main.csv", low_memory=False)
    img3 = pd.read_csv("/home/data/BRAIN_DIFF_S3/image.csv", low_memory=False).set_index("UID")
    Xc, Xs, Xd, gc, gs, gd = [], [], [], [], [], []
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
        if this % a.num_shards != a.shard_index: continue   # this GPU's slice; eligibility stays global
        fa, fb = enc_plain(pa), enc_plain(pb)
        dr = torch.cat([(fb - fa), (fb - fa).abs()], -1).cpu().numpy()
        for i in np.where(ch.numpy())[0]: Xc.append(dr[i]); gc.append(this)
        for i in np.where(st.numpy())[0]: Xs.append(dr[i]); gs.append(this)
        da1, da2 = enc_aug(pa, 2 * this + 1), enc_aug(pa, 2 * this + 2)
        dd = torch.cat([(da2 - da1), (da2 - da1).abs()], -1).cpu().numpy()
        for i in np.where(ta.numpy())[0]: Xd.append(dd[i]); gd.append(this)
    np.savez(a.dump, Xc=np.array(Xc), Xs=np.array(Xs), Xd=np.array(Xd),
             gc=np.array(gc), gs=np.array(gs), gd=np.array(gd))
    print(f"shard {a.shard_index}/{a.num_shards} done: change={len(Xc)} stable={len(Xs)} dup={len(Xd)} -> {a.dump}")

def merge(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    fs = sorted(glob.glob(a.dump_glob))
    Xc = np.vstack([np.load(f)["Xc"] for f in fs if len(np.load(f)["Xc"])])
    Xs = np.vstack([np.load(f)["Xs"] for f in fs if len(np.load(f)["Xs"])])
    Xd = np.vstack([np.load(f)["Xd"] for f in fs if len(np.load(f)["Xd"])])
    gc = np.concatenate([np.load(f)["gc"] for f in fs if len(np.load(f)["gc"])])
    gd = np.concatenate([np.load(f)["gd"] for f in fs if len(np.load(f)["gd"])])
    res = {"n_change": len(Xc), "n_stable": len(Xs), "n_duplicate": len(Xd),
           "n_pairs": int(len(set(gc.tolist()) | set(gd.tolist())))}
    X = np.vstack([Xc, Xd]); y = np.r_[np.ones(len(Xc)), np.zeros(len(Xd))]; g = np.r_[gc, gd]
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    res["separability_change_vs_duplicate_auroc"] = float(np.mean(aucs))
    Xt = np.vstack([Xc, Xs]); yt = np.r_[np.ones(len(Xc)), np.zeros(len(Xs))]
    clf = LogisticRegression(max_iter=2000).fit(Xt, yt)
    res["applied_meanP_change_on_duplicate"] = float(clf.predict_proba(Xd)[:, 1].mean())
    res["applied_meanP_change_on_real_stable"] = float(clf.predict_proba(Xs)[:, 1].mean())
    res["applied_meanP_change_on_real_change"] = float(clf.predict_proba(Xc)[:, 1].mean())
    json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=2)); print("TABLE5_DUPLICATE_DONE")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump", "merge"], default="dump")
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--labels", default="4,3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1); ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--dump", default="")
    ap.add_argument("--dump-glob", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "table5_duplicate_results.json"))
    a = ap.parse_args()
    if a.mode == "dump":
        try: dump(a)
        except Exception as e: print("ERR", e, traceback.format_exc()[-800:])
    else:
        merge(a)

if __name__ == "__main__":
    main()
