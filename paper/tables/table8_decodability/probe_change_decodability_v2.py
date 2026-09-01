"""Table 5 REBUILD — the v1 dense probe pooled tokens across pairs and split by token, so it
learned BETWEEN-pair change magnitude (0.711), not within-pair change. This version:

  DENSE (within-pair): GroupKFold by PAIR; for each held-out pair, AUROC over THAT pair's own
    change-vs-stable tokens; average. Reports mean within-pair AUROC (the honest target) AND
    the confounded pooled AUROC for contrast.
  POSITIVE CONTROL (pathology): mean+max pooled study features, labels with >=MINPOS positives,
    more studies -> a strong control that should clear ~0.75.

Frozen NeuroVFM (S2 base). GPU. Writes table5_v2_results.json.
"""
import os as _os

_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import os, sys, csv, json, argparse, traceback
sys.path.insert(0, _REPO)
csv.field_size_limit(10**7)
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
from braindiff.eval.diag_gate_localization import mask_to_tokens, resolve_mask, MODALITIES
from braindiff.data.neurovfm_transforms import NeuroVFMGridd, NeuroVFMTokenize
from braindiff.models.diff_encoder import DiffPretrainModel
from braindiff.training.checkpoint import load_stage_checkpoint

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--n-studies", type=int, default=1200)
    ap.add_argument("--minpos", type=int, default=30)
    ap.add_argument("--labels", default="4,3")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "table5_v2_results.json"))
    a = ap.parse_args()
    dev = f"cuda:{a.gpu}"; torch.cuda.set_device(dev)
    res = {}
    model = DiffPretrainModel(num_queries=64, device=dev, vision_lora_r=32, vision_lora_alpha=64).to(dev).eval()
    load_stage_checkpoint(model.captioner, torch.load("checkpoints/nv_stage2_reportgen.pt", map_location=dev),
                          label="S2", is_main=True)
    grid_tf = NeuroVFMGridd(keys=["k"], allow_missing_keys=True); tok_tf = NeuroVFMTokenize(keys=["k"], allow_missing_keys=True)
    label_set = [int(x) for x in a.labels.split(",")]
    import pandas as pd
    def enc(p):
        d = tok_tf({"k": grid_tf({"k": p})["k"]})["k"]
        with torch.no_grad():
            return model.captioner.vision_encoder(d["tokens"].unsqueeze(0).cuda().float(), d["coords"].unsqueeze(0).cuda().long()).squeeze(0)

    # ---------- DENSE within-pair ----------
    try:
        mm = pd.read_csv("/home/data/BRAIN_DIFF_S3/lesion_index.csv"); mask_map = dict(zip(mm["volume_path"], mm["mask_path"]))
        pairs = pd.read_csv("/home/data/BRAIN_DIFF_S3/main.csv", low_memory=False)
        img3 = pd.read_csv("/home/data/BRAIN_DIFF_S3/image.csv", low_memory=False).set_index("UID")
        X, y, grp = [], [], []; used = 0
        for _, row in pairs.iterrows():
            if used >= a.n_pairs: break
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
            if int(ch.sum()) < 3 or int(st.sum()) < 3: continue      # need >=3 of each for within-pair AUROC
            fa, fb = enc(pa), enc(pb); d = torch.cat([(fb-fa), (fb-fa).abs()], -1).cpu().numpy()
            for i in np.where(ch.numpy())[0]: X.append(d[i]); y.append(1); grp.append(used)
            for i in np.where(st.numpy())[0]: X.append(d[i]); y.append(0); grp.append(used)
            used += 1
        X, y, grp = np.array(X), np.array(y), np.array(grp)
        # confounded pooled (for contrast)
        pooled = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            pooled.append(roc_auc_score(y[te], LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:,1]))
        # within-pair: GroupKFold by pair, per held-out pair AUROC
        wp = []
        for tr, te in GroupKFold(5).split(X, y, grp):
            clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]); sc = clf.predict_proba(X[te])[:,1]
            for g in np.unique(grp[te]):
                m = grp[te] == g
                if y[te][m].sum() >= 1 and (~y[te][m].astype(bool)).sum() >= 1:
                    wp.append(roc_auc_score(y[te][m], sc[m]))
        res["dense_pooled_confounded_auroc"] = float(np.mean(pooled))
        res["dense_within_pair_auroc"] = float(np.mean(wp)); res["dense_within_pair_n"] = len(wp); res["dense_pairs"] = used
    except Exception as e:
        res["dense_error"] = f"{e}\n{traceback.format_exc()[-600:]}"

    # ---------- POSITIVE CONTROL: pathology, mean+max pool, common labels ----------
    try:
        lab = pd.read_csv("/home/data/MR-RATE/pathology_labels/mrrate_labels.csv")
        labcols = [c for c in lab.columns if c != "study_uid"]; lab = lab.set_index("study_uid")
        img = pd.read_csv("/home/data/BRAIN_DIFF_S4/image_extended.csv", low_memory=False).set_index("study_uid")
        F, Y = [], []
        for uid in img.index:
            if uid not in lab.index or len(F) >= a.n_studies: continue
            paths = [img.loc[uid, m] for m in MODALITIES if isinstance(img.loc[uid].get(m), str) and os.path.exists(str(img.loc[uid].get(m)))]
            if not paths: continue
            v = torch.stack([enc(p) for p in paths]).reshape(-1, 768)   # all tokens, all modalities
            feat = torch.cat([v.mean(0), v.amax(0)]).cpu().numpy()       # mean + max pool
            F.append(feat); Y.append(lab.loc[uid, labcols].values.astype(float))
        F, Y = np.array(F), np.array(Y)
        aucs = []
        for j in range(Y.shape[1]):
            pos = int(Y[:, j].sum())
            if a.minpos <= pos <= len(Y) - a.minpos:
                a5 = []
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F, Y[:, j]):
                    a5.append(roc_auc_score(Y[te, j], LogisticRegression(max_iter=2000).fit(F[tr], Y[tr, j]).predict_proba(F[te])[:,1]))
                aucs.append(np.mean(a5))
        res["positive_control_pathology_macro_auroc"] = float(np.mean(aucs)); res["positive_control_n_labels"] = len(aucs); res["positive_control_n_studies"] = len(F)
    except Exception as e:
        res["positive_control_error"] = f"{e}\n{traceback.format_exc()[-600:]}"

    json.dump(res, open(a.out, "w"), indent=2); print(json.dumps(res, indent=2)); print("TABLE5_V2_DONE")

if __name__ == "__main__":
    main()
