"""Base NeuroVFM (released, un-fine-tuned) -> CURRENT-study report per subset study, using
the model's OWN native inference pipeline (neurovfm.pipelines.load_vlm + StudyPreprocessor +
FindingsGenerationPipeline) -- its own prompt, preprocessing, and native JSON/findings style.
NOT our S4 prompt. The USER then synthesizes a longitudinal report from (this current report +
the prior report) via their longitudinal module; that is what gets scored.

Feeds each study's modality volumes (BRAIN_DIFF_S4 nii.gz) through the base preprocessor, which
re-resamples to the model's native 1x1x4mm and tokenizes. Needs a GPU
(H200 -> PYTHONPATH=/path/to/fa_builds/sm90).

Output: outputs/neurovfm_base/current_reports.jsonl  ({study_uid2, hyp_current, report2})."""
import os, sys, csv, json, argparse
sys.path.insert(0, "/path/to/code/neurovfm")
csv.field_size_limit(10**7)
import torch
from neurovfm.pipelines import load_vlm

MODS = ["T1w", "T1ce", "T2w", "FLAIR"]
HERE = os.path.dirname(os.path.abspath(__file__))
IMAGE_CSV = "/home/data/BRAIN_DIFF_S4/image_extended.csv"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--subset-csv", default=os.path.join(HERE, "subset_64.csv"))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(HERE, "outputs/neurovfm_base/current_reports.jsonl"))
    a = ap.parse_args()
    device = f"cuda:{a.gpu}"

    generator, preproc = load_vlm("mlinslab/neurovfm-llm", device=device)

    imap = {}
    with open(IMAGE_CSV, newline="") as f:
        for r in csv.DictReader(f):
            imap[r["study_uid"]] = r
    subset = list(csv.DictReader(open(a.subset_csv)))
    if a.num_shards > 1:
        subset = [row for i, row in enumerate(subset) if i % a.num_shards == a.shard_index]
    if a.limit:
        subset = subset[:a.limit]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        for i, row in enumerate(subset):
            uid = row["study_uid2"]
            rec = imap.get(uid, {})
            paths = [rec[m] for m in MODS
                     if isinstance(rec.get(m), str) and rec.get(m) and os.path.exists(rec[m])]
            if not paths:
                print(f"  [{i+1}/{len(subset)}] {uid} SKIP (no volumes)"); continue
            try:
                batch = preproc.load_study(paths, modality="mri")
                # FIX released-pipeline bug: load_study emits study_cu_seqlens in SERIES-count
                # convention ([0, n_series]), but model.generate expects TOKEN-count boundaries
                # that must appear in series_cu_seqlens. For one study span all series' tokens.
                batch["study_cu_seqlens"] = torch.tensor(
                    [0, int(batch["series_cu_seqlens"][-1])], dtype=torch.int32)
                report = generator(batch, clinical_context=None)
            except Exception as e:
                print(f"  [{i+1}/{len(subset)}] {uid} ERROR: {e}"); continue
            f.write(json.dumps({"study_uid2": uid, "hyp_current": report, "report2": row.get("report2", "")}) + "\n")
            f.flush()
            print(f"  [{i+1}/{len(subset)}] {uid}: {report[:100].replace(chr(10),' | ')}")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
